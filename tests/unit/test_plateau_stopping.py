"""Plateau stopping against a frozen anchor checkpoint (#39)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_simple.py"
_spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
assert _spec is not None and _spec.loader is not None
train_simple = importlib.util.module_from_spec(_spec)
sys.modules["train_simple"] = train_simple
_spec.loader.exec_module(train_simple)


def test_no_plateau_without_enough_checks() -> None:
    assert not train_simple.plateau_reached([0.2, 0.1, 0.1], 3)


def test_plateau_when_recent_checks_never_beat_the_best() -> None:
    assert train_simple.plateau_reached([0.2, 0.5, 0.4, 0.5, 0.3], 3)


def test_no_plateau_while_still_improving() -> None:
    assert not train_simple.plateau_reached([0.2, 0.5, 0.4, 0.45, 0.55], 3)


def test_plateau_off_when_patience_is_zero() -> None:
    assert not train_simple.plateau_reached([0.9, 0.1, 0.1, 0.1], 0)


def test_wilson_matches_the_head_to_head_numbers() -> None:
    # 63/100 was reported as 0.532-0.718 on #35.
    lo, hi = train_simple.wilson_interval(0.63, 100)
    assert lo == pytest.approx(0.532, abs=1e-3)
    assert hi == pytest.approx(0.718, abs=1e-3)


def test_best_anchor_keeps_the_first_best() -> None:
    R = train_simple.IterationResult
    results = [
        R(1, 0.5, 1.0, 10, 20.0),
        R(2, 0.5, 1.0, 10, 20.0, anchor_score=0.4),
        R(3, 0.5, 1.0, 10, 20.0, anchor_score=0.6),
        R(4, 0.5, 1.0, 10, 20.0, anchor_score=0.6),
    ]
    assert train_simple.best_anchor(results) == (3, 0.6)
    assert train_simple.best_anchor(results[:1]) is None


def test_plateau_needs_an_anchor() -> None:
    with pytest.raises(ValueError, match="--anchor"):
        train_simple.train(
            iterations=1,
            games=1,
            eval_games=1,
            simulations=1,
            seed=0,
            plateau=2,
        )


def test_anchor_run_writes_scores_and_best(tmp_path: Path) -> None:
    probe = train_simple.MCTSAgent(player_id=0, simulations=1)
    size = len(probe.action_space.action_to_id)
    anchor_net = train_simple.build_simple_network(action_space_size=size)
    anchor = tmp_path / "anchor.pt"
    torch.save(
        {"network": anchor_net.state_dict(), "action_space_size": size}, anchor
    )
    out = tmp_path / "run.json"
    ckpt = tmp_path / "ckpt"

    results = train_simple.train(
        iterations=3,
        games=1,
        eval_games=1,
        simulations=2,
        seed=0,
        out=out,
        checkpoint_dir=ckpt,
        ref_eval_games=0,
        anchor=anchor,
        anchor_every=1,
        anchor_games=2,
        plateau=1,
    )

    payload = json.loads(out.read_text())
    scores = [r["anchor_score"] for r in payload["results"]]
    assert all(s is not None for s in scores)
    assert payload["best_anchor_score"] == max(scores)
    assert (ckpt / "best.pt").exists()
    # A plateau of 1 stops at the first check that doesn't improve; that
    # can be the last iteration, so stopped_at may equal the cap.
    if payload["stopped_at"] is None:
        assert len(results) == 3
    else:
        assert payload["stopped_at"] == len(results) <= 3
        assert scores[-1] <= max(scores[:-1])
