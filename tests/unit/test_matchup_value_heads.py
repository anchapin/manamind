"""Per-matchup value heads (manamind #96, deck identity v2)."""

import argparse

import pytest
import torch

from manamind.models.forge_pointer import (
    ForgePointerNet,
    deck_bucket,
    load_pointer_net,
)
from manamind.models.forge_pointer_onnx import export, pack, unpack


def decision(deck="red", opp="green"):
    return {
        "t": "block",
        "hand": [{}, {}],
        "battlefield": [{}],
        "blockers": [{}, {}],
        "attackers": [{}],
        "deck": deck,
        "opp_deck": opp,
    }


def test_default_net_has_no_matchup_heads():
    net = ForgePointerNet()
    assert net.matchup_decks == ()
    assert not hasattr(net, "matchup_value_heads")
    d = decision()
    state = net.encode_state(d)
    own, opp = net.deck_ids(d)
    assert torch.equal(
        net.value(state, own, opp),
        torch.tanh(net.value_head(state)).squeeze(-1),
    )


def test_matchup_heads_equal_base_at_init():
    net = ForgePointerNet(matchup_decks=("red", "green"))
    assert len(net.matchup_value_heads) == 4
    state = net.encode_state(decision())
    own, opp = net.deck_ids(decision())
    assert torch.equal(net.value(state, own, opp), net.value(state))


def test_matchup_heads_diverge_per_combo():
    net = ForgePointerNet(matchup_decks=("red", "green"))
    red, green = deck_bucket("red"), deck_bucket("green")
    with torch.no_grad():
        net.matchup_value_heads[0][0].weight.add_(1.0)
        net.matchup_value_heads[1][2].bias.add_(0.5)
    state = net.encode_state(decision())
    base = net.value(state)
    assert not torch.equal(net.value(state, red, red), base)
    assert not torch.equal(net.value(state, red, green), base)
    # untouched heads still agree with the base head
    assert torch.equal(net.value(state, green, green), base)
    # pairings outside the configured decks fall to the base head
    assert torch.equal(net.value(state, 7, 9), base)


def test_unknown_decks_fall_to_base_head():
    net = ForgePointerNet(matchup_decks=("red", "green"))
    d = decision(deck="", opp="")
    state = net.encode_state(d)
    own, opp = net.deck_ids(d)
    assert (own, opp) == (0, 0)
    assert torch.equal(net.value(state, own, opp), net.value(state))


def test_v1_checkpoint_loads_bit_identical(tmp_path):
    plain = ForgePointerNet()
    path = tmp_path / "v1.pt"
    torch.save(
        {
            "network": plain.state_dict(),
            "config": {
                "card_dim": plain.card_dim,
                "state_dim": plain.state_dim,
            },
            "meta": {},
        },
        path,
    )
    net, _ = load_pointer_net(str(path), matchup_decks=("red", "green"))
    assert net.matchup_decks == ("red", "green")
    for dk, ok in (
        ("red", "green"),
        ("green", "red"),
        ("red", "red"),
        ("green", "green"),
    ):
        d = decision(deck=dk, opp=ok)
        state = net.encode_state(d)
        own, opp = net.deck_ids(d)
        plain_state = plain.encode_state(d)
        assert torch.equal(net.encode_state(d), plain_state)
        assert torch.equal(net.value(state, own, opp), plain.value(plain_state))


def test_matchup_checkpoint_resume_from_config(tmp_path):
    net = ForgePointerNet(matchup_decks=("red", "green"))
    path = tmp_path / "m.pt"
    torch.save(
        {
            "network": net.state_dict(),
            "config": {
                "card_dim": net.card_dim,
                "state_dim": net.state_dim,
                "matchup_decks": ["red", "green"],
            },
            "meta": {},
        },
        path,
    )
    net2, _ = load_pointer_net(str(path))
    assert net2.matchup_decks == ("red", "green")
    d = decision()
    assert torch.equal(
        net2.encode_state(d), net.encode_state(d)
    )
    own, opp = net.deck_ids(d)
    assert torch.equal(
        net2.value(net2.encode_state(d), own, opp),
        net.value(net.encode_state(d), own, opp),
    )


def test_onnx_matchup_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort  # type: ignore[import-untyped]

    net = ForgePointerNet(matchup_decks=("red", "green"))
    with torch.no_grad():
        net.matchup_value_heads[2][0].weight.add_(0.75)
    path = export(net, tmp_path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for dk, ok in (("red", "green"), ("green", "green"), ("blue", "red")):
        d = decision(deck=dk, opp=ok)
        outputs = sess.run(None, pack(d))
        got = float(unpack(d, outputs)["value"])
        state = net.encode_state(d)
        own, opp = net.deck_ids(d)
        want = float(net.value(state, own, opp))
        assert got == pytest.approx(want, abs=1e-4)


def test_loop_matchup_flags():
    from manamind.training.selfplay_loop import matchup_flags

    args = argparse.Namespace(
        matchup_value_heads=True, deck_a="red", deck_b="green"
    )
    assert matchup_flags(args) == ["--matchup-decks", "red,green"]
    off = argparse.Namespace(
        matchup_value_heads=False, deck_a="red", deck_b="green"
    )
    assert matchup_flags(off) == []
