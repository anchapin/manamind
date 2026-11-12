"""Opponent-deck probe: history-only deck identification (manamind #44)."""

from typing import Optional

import torch

from manamind.models.forge_pointer import (
    DECK_BUCKETS,
    ForgePointerNet,
    OppDeckProbe,
    deck_bucket,
    opp_deck_eval,
    opp_deck_labels,
    opp_deck_loss,
)
from manamind.training.train_selfplay import train_epochs


def _decision(turn: int = 0, deck: Optional[str] = None) -> dict:
    d = {
        "turn": turn,
        "opp_history": {
            "buckets": [
                {"landsPlayed": 1 if deck == "red deck" else 0}
                for _ in range(4)
            ],
            "totals": {"observations": 1},
            "oppManaPool": {},
        },
    }
    if deck is not None:
        d["opp_deck"] = deck
    return d


def _forced_probe(deck: str) -> OppDeckProbe:
    """A probe whose logits always argmax ``deck``'s bucket."""
    probe = OppDeckProbe()
    probe.mlp[0].weight.data.zero_()
    probe.mlp[0].bias.data.zero_()
    probe.mlp[2].weight.data.zero_()
    bias = torch.full((DECK_BUCKETS,), -1e9)
    bias[deck_bucket(deck)] = 1e9
    probe.mlp[2].bias.data = bias
    probe.eval()
    return probe


def test_logits_shape_and_missing_history() -> None:
    probe = OppDeckProbe()
    logits = probe.logits([_decision(), _decision(2, "red deck")])
    assert logits.shape == (2, DECK_BUCKETS)
    assert bool(torch.isfinite(logits).all())


def test_labels_mask_unknown_deck() -> None:
    decisions = [_decision(0, "red deck"), _decision(1), _decision(3, "x")]
    labels = opp_deck_labels(decisions)
    assert labels == [deck_bucket("red deck"), None, deck_bucket("x")]


def test_loss_masks_unlabeled_rows() -> None:
    probe = OppDeckProbe()
    loss, stats = opp_deck_loss(probe, [_decision(), _decision()])
    assert loss is None
    assert stats["opp_deck_labeled"] == 0.0
    loss, stats = opp_deck_loss(
        probe, [_decision(0, "red deck"), _decision(1)]
    )
    assert loss is not None
    assert stats["opp_deck_labeled"] == 1.0
    loss.backward()
    for p in ForgePointerNet().parameters():
        assert p.grad is None


def test_probe_reaches_the_right_deck_by_turn() -> None:
    probe = _forced_probe("red deck")
    decisions = [_decision(0, "red deck"), _decision(2, "red deck")]
    _, stats = opp_deck_loss(probe, decisions)
    assert stats["opp_deck_acc"] == 1.0
    assert stats["opp_deck_acc_t01"] == 1.0
    assert stats["opp_deck_acc_t23"] == 1.0
    assert stats["opp_deck_acc_t4p"] == 0.0
    assert opp_deck_eval(probe, decisions) == {
        k: v for k, v in stats.items() if k != "opp_deck_loss"
    }


def test_eval_reports_mixed_buckets() -> None:
    probe = _forced_probe("red deck")
    decisions = [
        _decision(0, "red deck"),
        _decision(4, "red deck"),
        _decision(4, "other deck"),
    ]
    stats = opp_deck_eval(probe, decisions)
    assert stats["opp_deck_labeled"] == 3.0
    assert abs(stats["opp_deck_acc"] - 2.0 / 3.0) < 1e-6
    assert stats["opp_deck_acc_t01"] == 1.0
    assert abs(stats["opp_deck_acc_t4p"] - 0.5) < 1e-6


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    probe = OppDeckProbe()
    decisions = [
        _decision(t, "red deck" if t % 2 else "green deck") for t in range(4)
    ]
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    first = None
    for _ in range(300):
        loss, _ = opp_deck_loss(probe, decisions)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert first is not None and float(loss.detach()) < first * 0.5


def test_train_epochs_untouched_by_probe() -> None:
    view = {
        "turn": 3,
        "phase": "MAIN1",
        "active": True,
        "life": [18, 20],
        "hand": [],
        "opp_hand_size": 5,
        "battlefield": [],
        "opp_battlefield": [],
        "graveyard": [],
        "opp_graveyard": [],
        "library": [50, 50],
        "t": "priority",
        "options": [],
        "seat": 0,
        "pi": [0.5, 0.5],
        "deck": "red deck",
        "opp_deck": "green deck",
    }
    samples = [(dict(view), 1.0), (dict(view), 0.0)] * 3
    torch.manual_seed(7)
    base = ForgePointerNet()
    base_opt = torch.optim.Adam(base.parameters(), lr=0.01)
    train_epochs(base, base_opt, samples, epochs=2, batch=2, seed=3)
    torch.manual_seed(7)
    probed = ForgePointerNet()
    probed_opt = torch.optim.Adam(probed.parameters(), lr=0.01)
    probe = OppDeckProbe()
    probe_opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    train_epochs(
        probed,
        probed_opt,
        samples,
        epochs=2,
        batch=2,
        seed=3,
        probe=probe,
        probe_opt=probe_opt,
    )
    for (k, a), (_, b) in zip(
        base.state_dict().items(), probed.state_dict().items()
    ):
        assert k == _
        assert torch.equal(a, b), k
