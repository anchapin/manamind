"""Deck-pair cycling in the imitation phase (#96 data axis)."""

import gzip
import json
from pathlib import Path

import pytest
import torch

import manamind.training.train_forge as tf
from manamind.forge_interface import EXPERT_REPLY
from manamind.forge_interface.forge_env import StepResult


class StubNet:
    """Just enough ForgePointerNet for a no-decision run."""

    def __init__(self, card_dim: int = 64, state_dim: int = 128) -> None:
        self.card_dim = card_dim
        self.state_dim = state_dim
        self.p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return [self.p]

    def state_dict(self):
        return {"p": self.p.detach()}


def make_fake_env(registry: list):
    class FakeEnv:
        def __init__(self, command, cwd=None):
            self.command = list(command)
            self.cwd = cwd
            self.summary = None
            self.served = 0
            self.finished = False
            registry.append(self)

        def reset(self):
            self.served += 1
            return StepResult(None, True, 1.0, {"result": "win", "turns": 4})

        def step(self, reply):
            assert reply == EXPERT_REPLY
            return StepResult(None, True, 1.0, {"result": "win", "turns": 4})

        def close(self):
            self.closed = True

    return FakeEnv


@pytest.fixture()
def patched(monkeypatch):
    registry: list = []
    monkeypatch.setattr(tf, "ForgeEnv", make_fake_env(registry))
    monkeypatch.setattr(tf, "ForgePointerNet", StubNet)
    return registry


def _read_log(out_dir: Path) -> list:
    return [
        json.loads(line)
        for line in (out_dir / "log.jsonl").read_text().splitlines()
    ]


def test_two_pairs_cycle_in_blocks(patched, tmp_path):
    out = tmp_path / "run"
    meta = tf.train(
        command=["rl"],
        cwd=None,
        games=4,
        out_dir=out,
        expert_commands=[["bridgeA"], ["bridgeB"]],
        expert_games=4,
        save_every=0,
    )
    assert len(patched) == 2
    assert [e.command for e in patched] == [["bridgeA"], ["bridgeB"]]
    assert [e.served for e in patched] == [2, 2]
    rows = _read_log(out)
    imitate = [r for r in rows if r["phase"] == "imitate"]
    assert [r["pair"] for r in imitate] == [0, 0, 1, 1]
    with gzip.open(out / tf.EXPERT_DATA, "rt") as f:
        data = f.read().splitlines()
    assert len(data) == 4
    assert meta["games"] == 4 and meta["bc_done"] is True
    assert (out / "bc.pt").exists()


def test_single_command_uses_one_bridge(patched, tmp_path):
    out = tmp_path / "run"
    meta = tf.train(
        command=["rl"],
        cwd=None,
        games=2,
        out_dir=out,
        expert_command=["bridgeX"],
        expert_games=2,
        save_every=0,
    )
    assert len(patched) == 1
    assert patched[0].command == ["bridgeX"]
    assert patched[0].served == 2
    imitate = [r for r in _read_log(out) if r["phase"] == "imitate"]
    assert [r["pair"] for r in imitate] == [0, 0]
    assert meta["bc_done"] is True


def test_no_expert_command_raises():
    with pytest.raises(ValueError):
        tf.train(
            command=["rl"],
            cwd=None,
            games=2,
            out_dir=Path("/tmp/unused"),
            expert_games=2,
        )


def test_parse_deck_pairs():
    pairs = tf._parse_deck_pairs("a,b c,d")
    assert pairs == [(Path("a"), Path("b")), (Path("c"), Path("d"))]
    with pytest.raises(ValueError):
        tf._parse_deck_pairs("a,b,c")
    with pytest.raises(ValueError):
        tf._parse_deck_pairs("  ")
