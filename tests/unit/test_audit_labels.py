"""Read-only label audit for self-play buffers (#96)."""

import gzip
import json
from pathlib import Path

import pytest
import torch

from manamind.models.forge_pointer import ForgePointerNet
from manamind.training.audit_labels import (
    audit_buffer,
    audit_decision,
    kl_divergence,
    pearson,
)

BEAR = {
    "name": "Grizzly Bears",
    "type": "Creature - Bear",
    "cost": "{1}{G}",
    "cmc": 2,
    "creature": True,
    "land": False,
    "power": 2,
    "toughness": 2,
    "tapped": False,
    "sick": False,
}
FOREST = {**BEAR, "name": "Forest", "type": "Basic Land - Forest"}
FOREST.update(creature=False, land=True, power=0, toughness=0, cmc=0)
VIEW = {
    "turn": 3,
    "phase": "MAIN1",
    "active": True,
    "life": [18, 20],
    "hand": [BEAR, FOREST],
    "opp_hand_size": 5,
    "battlefield": [BEAR],
    "opp_battlefield": [BEAR],
    "graveyard": [],
    "opp_graveyard": [],
    "library": [50, 50],
}
PRIORITY = {
    **VIEW,
    "t": "priority",
    "options": [
        {"card": BEAR, "spell": True, "land": False},
        {"card": FOREST, "spell": False, "land": True},
    ],
    "seat": 0,
    "pi": [0.1, 0.8, 0.1],
}
ATTACK = {**VIEW, "t": "attack", "options": [BEAR, BEAR], "pi": [0.9, 0.2]}
BLOCK = {
    **VIEW,
    "t": "block",
    "blockers": [BEAR],
    "attackers": [BEAR, BEAR],
    "pi": [[0.7, 0.1, 0.2]],
}


def test_kl_divergence_basics() -> None:
    p = torch.tensor([[0.5, 0.5]])
    assert kl_divergence(p, p).item() == pytest.approx(0.0, abs=1e-6)
    uniform = torch.tensor([[0.5, 0.5]])
    point = torch.tensor([[1.0, 0.0]])
    assert kl_divergence(point, uniform).item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item(), abs=1e-6
    )
    assert kl_divergence(uniform, point).item() > 10  # zero in q is clamped


def test_pearson() -> None:
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0], [1.0]) is None


def test_audit_decision_each_head() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet().eval()
    for d in (PRIORITY, BLOCK):
        m = audit_decision(net, d)
        assert m is not None and m["kl"] >= 0
        assert 0.0 <= m["top1_agree"] <= 1.0
        assert -1.0 <= m["value"] <= 1.0
    a = audit_decision(net, ATTACK)
    assert a is not None and a["prior_bce"] >= 0 and a["abs_diff"] >= 0
    assert audit_decision(net, {**PRIORITY, "pi": [1.0]}) is None
    assert audit_decision(net, {**ATTACK, "pi": [0.5]}) is None
    assert audit_decision(net, {**BLOCK, "pi": [[1.0, 0.0]]}) is None
    assert (
        audit_decision(net, {k: v for k, v in PRIORITY.items() if k != "pi"})
        is None
    )
    assert audit_decision(net, {**PRIORITY, "t": "mulligan"}) is None


def test_audit_buffer_reads_gz(tmp_path: Path) -> None:
    torch.manual_seed(1)
    net = ForgePointerNet().eval()
    game = {
        "seed": 0,
        "decisions": [PRIORITY, ATTACK, BLOCK],
        "returns": [1.0, -1.0, 1.0],
        "winner": 0,
    }
    path = tmp_path / "round_0001.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps(game) + "\n")
        fh.write(json.dumps(game) + "\n")
    report = audit_buffer(net, path, max_games=0)
    assert report["games"] == 2
    assert report["kinds"]["priority"]["n"] == 2
    assert report["kinds"]["attack"]["n"] == 2
    assert report["kinds"]["block"]["n"] == 2
    assert report["value"]["n"] == 6
    assert report["value"]["floor_mse"] is not None
    assert report["value"]["mse"] >= 0
    with gzip.open(path, "rt") as fh:
        lines = fh.readlines()
    assert len(lines) == 2
    assert audit_buffer(net, path, max_games=1)["games"] == 1
