"""EMA weights and best.pt selection between raw and EMA (#57)."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_simple.py"
if "train_simple" in sys.modules:
    train_simple = sys.modules["train_simple"]
else:
    _spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    train_simple = importlib.util.module_from_spec(_spec)
    sys.modules["train_simple"] = train_simple
    _spec.loader.exec_module(train_simple)


def _net(seed: int):
    torch.manual_seed(seed)
    return train_simple.build_simple_network(action_space_size=23)


def test_ema_step_blends_toward_the_live_weights() -> None:
    ema, live = _net(0), _net(1)
    before = {k: v.clone() for k, v in ema.state_dict().items()}
    train_simple.update_ema(ema, live, 0.9)
    for name, value in ema.state_dict().items():
        if value.dtype.is_floating_point:
            expected = 0.9 * before[name] + 0.1 * live.state_dict()[name]
            assert torch.allclose(value, expected, atol=1e-6)


def test_ema_with_zero_decay_copies_the_live_weights() -> None:
    ema, live = _net(0), _net(1)
    train_simple.update_ema(ema, live, 0.0)
    for name, value in ema.state_dict().items():
        assert torch.allclose(value.float(), live.state_dict()[name].float())


def _result(i, raw, ema=None):
    return train_simple.IterationResult(
        iteration=i,
        win_rate=0.5,
        mean_loss=1.0,
        examples=10,
        mean_turns=5.0,
        anchor_score=raw,
        ema_anchor_score=ema,
    )


def test_best_picks_ema_when_it_scores_higher() -> None:
    results = [_result(5, 0.40, 0.45), _result(10, 0.50, 0.55)]
    assert train_simple.best_anchor_weights(results) == (10, 0.55, "ema")
    assert train_simple.best_anchor(results) == (10, 0.55)


def test_raw_wins_a_tie_and_earlier_checks_win_ties() -> None:
    results = [_result(5, 0.50, 0.50), _result(10, 0.50, 0.50)]
    assert train_simple.best_anchor_weights(results) == (5, 0.50, "raw")


def test_without_ema_selection_is_unchanged() -> None:
    results = [_result(5, 0.40), _result(10, 0.30)]
    assert train_simple.best_anchor_weights(results) == (5, 0.40, "raw")


def test_checkpoint_stores_ema_and_records_weights(tmp_path) -> None:
    net, ema = _net(0), _net(1)
    opt = torch.optim.Adam(net.parameters())
    path = tmp_path / "best.pt"
    train_simple.save_checkpoint(
        path,
        ema,
        opt,
        iteration=5,
        seed=0,
        action_space_size=23,
        result=_result(5, 0.4, 0.5),
        ema=ema,
        weights="ema",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["weights"] == "ema"
    assert "ema_network" in payload
    loaded = train_simple.load_checkpoint(path)
    for name, value in loaded.state_dict().items():
        assert torch.allclose(value.float(), ema.state_dict()[name].float())


def test_bad_decay_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_simple.train(
            iterations=1,
            games=1,
            eval_games=1,
            simulations=1,
            seed=0,
            ema_decay=1.0,
        )


def test_load_checkpoint_can_pick_the_ema_weights(tmp_path) -> None:
    raw, ema = _net(0), _net(1)
    opt = torch.optim.Adam(raw.parameters())
    path = tmp_path / "iter_005.pt"
    train_simple.save_checkpoint(
        path,
        raw,
        opt,
        iteration=5,
        seed=0,
        action_space_size=23,
        result=_result(5, 0.4, 0.5),
        ema=ema,
    )
    as_saved = train_simple.load_checkpoint(path)
    as_ema = train_simple.load_checkpoint(path, weights="ema")
    for name, value in as_saved.state_dict().items():
        assert torch.allclose(value.float(), raw.state_dict()[name].float())
    for name, value in as_ema.state_dict().items():
        assert torch.allclose(value.float(), ema.state_dict()[name].float())


def test_load_checkpoint_without_ema_refuses_ema(tmp_path) -> None:
    raw = _net(0)
    opt = torch.optim.Adam(raw.parameters())
    path = tmp_path / "iter_001.pt"
    train_simple.save_checkpoint(
        path,
        raw,
        opt,
        iteration=1,
        seed=0,
        action_space_size=23,
        result=_result(1, 0.4, None),
    )
    with pytest.raises(ValueError, match="no EMA"):
        train_simple.load_checkpoint(path, weights="ema")
    with pytest.raises(ValueError):
        train_simple.load_checkpoint(path, weights="bogus")
