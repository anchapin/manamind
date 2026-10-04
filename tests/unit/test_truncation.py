"""Truncated games (MAX_STEPS hit) are flagged, not scored as draws (#54)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from manamind.core.agent import MCTSAgent, RandomAgent

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_simple.py"
if "train_simple" in sys.modules:
    train_simple = sys.modules["train_simple"]
else:
    _spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    train_simple = importlib.util.module_from_spec(_spec)
    sys.modules["train_simple"] = train_simple
    _spec.loader.exec_module(train_simple)


def _randoms():
    return {0: RandomAgent(0, seed=1), 1: RandomAgent(1, seed=2)}


def test_capped_game_is_flagged_truncated(monkeypatch) -> None:
    monkeypatch.setattr(train_simple, "MAX_STEPS", 3)
    winner, _, _, truncated = train_simple.play_game_outcome(
        _randoms(), seed=7
    )
    assert truncated is True
    assert winner is None


def test_finished_game_is_not_truncated() -> None:
    winner, _, _, truncated = train_simple.play_game_outcome(
        _randoms(), seed=7
    )
    assert truncated is False
    assert winner in (0, 1, None)


def test_play_game_keeps_its_three_value_shape(monkeypatch) -> None:
    monkeypatch.setattr(train_simple, "MAX_STEPS", 3)
    out = train_simple.play_game(_randoms(), seed=7)
    assert len(out) == 3


def test_unknown_truncated_value_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_simple.play_game_outcome(
            _randoms(), seed=7, truncated_value="zero"
        )


def test_score_counts_truncations_but_keeps_the_half_point() -> None:
    jobs = [train_simple.GameJob("random", g, seat=0) for g in range(4)]
    results = [
        (0, 10, [], False),
        (1, 10, [], False),
        (None, 400, [], True),
        (None, 12, [], False),
    ]
    stats = {}
    score = train_simple._score(jobs, results, stats)
    assert score == pytest.approx((1.0 + 0.5 + 0.5) / 4)
    assert stats["truncated"] == 1


def _selfplay(truncated_value: str, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(train_simple, "MAX_STEPS", 6)
    probe = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    torch.manual_seed(0)
    net = train_simple.build_simple_network(
        action_space_size=len(probe.action_space.action_to_id)
    )
    job = train_simple.GameJob(
        "selfplay", 3, simulations=2, truncated_value=truncated_value
    )
    return train_simple._play_job(job, {"net": net}, reseed=True)


def test_draw_mode_trains_truncated_positions_toward_zero(monkeypatch) -> None:
    _, _, examples, truncated = _selfplay("draw", monkeypatch)
    assert truncated is True
    assert examples and all(e.value == 0.0 for e in examples)


def test_bootstrap_mode_uses_the_search_value(monkeypatch) -> None:
    _, _, examples, truncated = _selfplay("bootstrap", monkeypatch)
    assert truncated is True
    assert examples
    assert all(-1.0 <= e.value <= 1.0 for e in examples)
    assert any(e.value != 0.0 for e in examples)
