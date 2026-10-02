"""Learning-rate schedule, probe stats and training knobs (#49)."""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_simple.py"
_spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
assert _spec is not None and _spec.loader is not None
train_simple = importlib.util.module_from_spec(_spec)
sys.modules["train_simple"] = train_simple
_spec.loader.exec_module(train_simple)


def test_constant_schedule_keeps_the_rate() -> None:
    for it in (1, 25, 50):
        assert train_simple.lr_at(it, 50, 1e-3) == 1e-3


def test_cosine_runs_from_lr_to_lr_min() -> None:
    lr_at = train_simple.lr_at
    assert lr_at(1, 50, 1e-3, 1e-4, "cosine") == pytest.approx(1e-3)
    assert lr_at(50, 50, 1e-3, 1e-4, "cosine") == pytest.approx(1e-4)
    mid = lr_at(25, 49, 1e-3, 1e-4, "cosine")
    assert mid == pytest.approx(5.5e-4)
    rates = [lr_at(i, 50, 1e-3, 1e-4, "cosine") for i in range(1, 51)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_cosine_clamps_past_the_cap() -> None:
    assert train_simple.lr_at(60, 50, 1e-3, 1e-4, "cosine") == pytest.approx(
        1e-4
    )


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="schedule"):
        train_simple.lr_at(2, 10, 1e-3, schedule="step")


def _examples(count: int):
    state = train_simple.create_simple_game_start(seed=0)
    return [
        train_simple.Example(state, np.ones(4) / 4, 1.0 if i % 2 else -1.0)
        for i in range(count)
    ]


def test_probe_is_deterministic_and_leaves_global_rng_alone() -> None:
    examples = _examples(20)
    random.seed(123)
    before = random.getstate()
    a = train_simple.build_probe(examples, 5, seed=7)
    assert random.getstate() == before
    b = train_simple.build_probe(examples, 5, seed=7)
    assert [id(e) for e in a] == [id(e) for e in b]
    assert len(train_simple.build_probe(examples, 50, seed=7)) == 20
    assert train_simple.build_probe(examples, 0, seed=7) == []


def test_probe_stats_reports_entropy_and_value() -> None:
    probe = train_simple.MCTSAgent(player_id=0, simulations=1)
    size = len(probe.action_space.action_to_id)
    net = train_simple.build_simple_network(action_space_size=size)
    net.train()
    stats = train_simple.probe_stats(net, _examples(6))
    assert net.training, "probe_stats must restore train mode"
    assert set(stats) == {
        "probe_entropy",
        "probe_value_mean",
        "probe_value_std",
        "probe_value_mse",
        "probe_saturated",
    }
    assert 0.0 <= stats["probe_entropy"] <= math.log(size) + 1e-4
    assert 0.0 <= stats["probe_saturated"] <= 1.0
    assert train_simple.probe_stats(net, []) == {}


def test_run_logs_diagnostics_and_config(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    train_simple.train(
        iterations=2,
        games=1,
        eval_games=1,
        simulations=2,
        seed=0,
        out=out,
        ref_eval_games=0,
        lr=2e-3,
        lr_schedule="cosine",
        lr_min=2e-4,
        train_batches=3,
        batch_size=4,
        buffer_size=500,
        probe_size=8,
    )
    payload = json.loads(out.read_text())
    assert payload["lr_schedule"] == "cosine"
    assert payload["train_batches"] == 3
    assert payload["buffer_size"] == 500
    first, last = payload["results"]
    assert first["lr"] == pytest.approx(2e-3)
    assert last["lr"] == pytest.approx(2e-4)
    for r in payload["results"]:
        assert r["examples"] <= 500
        assert r["probe_entropy"] is not None
        assert r["grad_norm"] is not None
        assert r["samples_per_example"] == pytest.approx(3 * 4 / r["examples"])


def test_old_results_without_diagnostics_still_load() -> None:
    old = {
        "iteration": 1,
        "win_rate": 0.5,
        "mean_loss": 1.0,
        "examples": 10,
        "mean_turns": 20.0,
    }
    r = train_simple.IterationResult(**old)
    assert r.lr is None and r.as_dict()["probe_entropy"] is None


def test_resume_keeps_the_probe(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    common = dict(
        games=1,
        eval_games=1,
        simulations=2,
        seed=0,
        checkpoint_dir=ckpt,
        ref_eval_games=0,
        probe_size=8,
    )
    straight = train_simple.train(iterations=2, **common)
    payload = torch.load(
        ckpt / "iter_001.pt", map_location="cpu", weights_only=False
    )
    assert len(payload["resume"]["probe"]) == 8
    (ckpt / "iter_002.pt").unlink()
    resumed = train_simple.train(iterations=2, resume=True, **common)
    assert resumed[-1].probe_entropy == pytest.approx(
        straight[-1].probe_entropy
    )
