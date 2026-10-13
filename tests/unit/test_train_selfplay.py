"""Training ForgePointerNet on Planar Nexus self-play records (#87)."""

import gzip
import json
from pathlib import Path

import pytest
import torch

from manamind.models.forge_pointer import (
    ForgePointerNet,
    distill_loss,
    load_pointer_net,
)
from manamind.training.train_selfplay import (
    build_parser,
    carve_holdout,
    flatten,
    holdout_loss,
    load_selfplay,
    reset_value_head,
    run,
    train_epochs,
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


def _game(seed: int, n: int = 1) -> dict:
    decisions = [PRIORITY, ATTACK, BLOCK] * n
    return {
        "seed": seed,
        "decisions": decisions,
        "returns": [1.0, -1.0, 1.0] * n,
        "winner": 0,
    }


def test_distill_uses_each_head_and_rejects_bad_shapes() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet()
    for d in (PRIORITY, ATTACK, BLOCK):
        out = net.distill(d)
        assert out.used and torch.isfinite(out.policy) and out.policy > 0
    assert not net.distill({**PRIORITY, "pi": [1.0]}).used
    assert not net.distill({**ATTACK, "pi": [0.5]}).used
    assert not net.distill({**BLOCK, "pi": [[1.0, 0.0]]}).used
    assert not net.distill(
        {k: v for k, v in PRIORITY.items() if k != "pi"}
    ).used
    steps = [net.distill(d) for d in (PRIORITY, ATTACK, BLOCK)]
    loss, stats = distill_loss(steps, [1.0, -1.0, 0.0])
    loss.backward()
    assert stats["used_frac"] == 1.0
    assert net.priority_head[0].weight.grad is not None
    with pytest.raises(ValueError):
        distill_loss(steps, [1.0])


def test_training_moves_policy_toward_search_target() -> None:
    torch.manual_seed(1)
    net = ForgePointerNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    samples = flatten([_game(0)])
    before = train_epochs(net, opt, samples, epochs=1, batch=3)
    after = train_epochs(net, opt, samples, epochs=40, batch=3)
    assert after["policy"] < before["policy"]
    assert after["value"] < before["value"]
    state = net.encode_state(PRIORITY)
    probs = torch.softmax(net.priority_logits(state, PRIORITY["options"]), -1)
    assert int(probs.argmax()) == 1


def test_replay_buffer_keeps_newest_games(tmp_path: Path) -> None:
    old = tmp_path / "a.jsonl"
    old.write_text("".join(json.dumps(_game(s)) + "\n" for s in range(5)))
    new = tmp_path / "b.jsonl.gz"
    with gzip.open(new, "wt") as fh:
        fh.write("".join(json.dumps(_game(s)) + "\n" for s in range(5, 8)))
    import os

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    games = load_selfplay([tmp_path], buffer_games=4)
    assert [g["seed"] for g in games] == [4, 5, 6, 7]
    bad = tmp_path / "c.jsonl"
    bad.write_text(json.dumps({"decisions": [PRIORITY], "returns": []}))
    with pytest.raises(ValueError):
        load_selfplay([bad])


def test_cli_writes_loadable_checkpoint(tmp_path: Path) -> None:
    data = tmp_path / "round0.jsonl"
    data.write_text(json.dumps(_game(0, 2)) + "\n")
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            ["--data", str(data), "--out", str(out), "--epochs", "2"]
        )
    )
    assert meta["decisions"] == 6 and meta["selfplay_games_total"] == 1
    net, saved = load_pointer_net(str(out / "last.pt"))
    assert saved is not None and saved["phase"] == "selfplay"
    assert len(saved["history"]) == 2
    meta2 = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(tmp_path / "round2"),
                "--init",
                str(out / "last.pt"),
            ]
        )
    )
    assert meta2["selfplay_games_total"] == 2


def test_cli_exports_onnx(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    data = tmp_path / "g.jsonl"
    data.write_text(json.dumps(_game(0)) + "\n")
    out = tmp_path / "r"
    meta = run(
        build_parser().parse_args(
            ["--data", str(data), "--out", str(out), "--onnx"]
        )
    )
    assert Path(meta["onnx"]).exists()
    assert (out / "forge_pointer.schema.json").exists()


def test_reset_value_head_changes_head_and_keeps_trunk() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet()
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    reset_value_head(net)
    after = dict(net.named_parameters())
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    assert changed
    assert all("value_head" in n for n in changed)
    kept = [n for n in before if "value_head" not in n]
    assert all(torch.equal(before[n], after[n]) for n in kept)


def test_carve_holdout_deterministic_and_capped() -> None:
    samples = [({"t": "priority"}, 1.0)] * 100
    train, hold = carve_holdout(samples, 0.1, 1)
    assert len(hold) == 10 and len(train) == 90
    train2, hold2 = carve_holdout(samples, 0.1, 1)
    assert hold == hold2 and train == train2
    one_train, one_hold = carve_holdout(samples[:1], 0.1, 1)
    assert one_hold == [] and one_train == samples[:1]


def test_holdout_loss_evaluates_without_gradients() -> None:
    net = ForgePointerNet()
    samples = [({**PRIORITY, "t": "priority"}, 1.0) for _ in range(4)]
    stats = holdout_loss(net, samples, batch=2)
    assert stats is not None
    assert set(stats) == {"loss", "policy", "value", "used_frac"}
    assert stats["used_frac"] == 1.0
    assert holdout_loss(net, []) is None


def test_run_carves_holdout_and_records_it(tmp_path: Path) -> None:
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(json.dumps(_game(s, 2)) + "\n" for s in range(3)))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0.2",
            ]
        )
    )
    assert meta["holdout_examples"] > 0
    assert set(meta["holdout"]) == {"loss", "policy", "value", "used_frac"}
    assert (out / "selfplay_train.json").exists()
    meta2 = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(tmp_path / "round2"),
                "--holdout-frac",
                "0.2",
            ]
        )
    )
    assert meta2["holdout_examples"] == meta["holdout_examples"]
    meta3 = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(tmp_path / "round3"),
                "--holdout-frac",
                "0",
            ]
        )
    )
    assert meta3["holdout_examples"] == 0 and "holdout" not in meta3
    assert meta3["decisions"] == meta["decisions"]
