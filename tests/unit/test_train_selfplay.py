"""Training ForgePointerNet on Planar Nexus self-play records (#87)."""

import copy
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
    apply_expert_weight,
    blend_value_targets,
    build_parser,
    carve_holdout,
    flatten,
    holdout_loss,
    kl_weights,
    load_selfplay,
    mix_expert_rows,
    reset_value_head,
    run,
    sample_source,
    search_value_targets,
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


def test_mix_expert_rows_replays_to_target_share() -> None:
    samples = [({"source": "net"}, 1.0)] * 70 + [
        ({"source": "expert"}, -1.0)
    ] * 10
    mixed, fill, meta = mix_expert_rows(samples, 0.3)
    assert len(mixed) == 100 and len(fill) == 20
    share = sum(1 for d, _ in mixed if sample_source(d) == "expert") / len(
        mixed
    )
    assert share == pytest.approx(0.3)
    assert meta["expert_rows_base"] == 10
    assert meta["expert_rows_added"] == 20
    assert meta["expert_rows_final"] == 30


def test_mix_expert_rows_passthrough_at_default() -> None:
    samples = [({"source": "net"}, 1.0)] * 3
    mixed, fill, meta = mix_expert_rows(samples, 0.0)
    assert mixed is samples and fill == [] and meta == {}


def test_mix_expert_rows_skips_without_expert_rows() -> None:
    samples = [({"source": "net"}, 1.0)] * 3
    mixed, fill, meta = mix_expert_rows(samples, 0.3)
    assert mixed == samples and fill == []
    assert meta["expert_mix_skipped"] == "no expert rows"


def test_mix_expert_rows_no_add_at_target_share() -> None:
    samples = [({"source": "net"}, 1.0)] * 70 + [
        ({"source": "expert"}, -1.0)
    ] * 30
    mixed, fill, meta = mix_expert_rows(samples, 0.3)
    assert len(mixed) == 100 and fill == []
    assert meta["expert_rows_added"] == 0


def test_mix_expert_rows_rejects_full_share() -> None:
    with pytest.raises(ValueError):
        mix_expert_rows([({"source": "net"}, 1.0)], 1.0)


def test_run_mixes_expert_rows_into_train_pool(tmp_path: Path) -> None:
    rows = [json.dumps(_game(s)) + "\n" for s in range(8)]
    rows += [json.dumps(_expert_game(s)) + "\n" for s in range(2)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0",
                "--expert-mix-frac",
                "0.5",
            ]
        )
    )
    assert meta["decisions"] == 26
    assert meta["expert_mix_frac"] == 0.5
    assert meta["expert_rows_base"] == 2
    assert meta["expert_rows_added"] == 22
    assert meta["expert_samples"] == 24


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
    assert set(stats) == {
        "loss",
        "policy",
        "value",
        "used_frac",
        "value_by_turn",
        "value_by_turn_n",
    }
    assert stats["used_frac"] == 1.0
    assert stats["value_by_turn_n"] == {"1-3": 4}
    assert abs(stats["value_by_turn"]["1-3"] - stats["value"]) < 1e-4
    assert holdout_loss(net, []) is None


def test_holdout_loss_buckets_value_error_by_turn() -> None:
    net = ForgePointerNet()
    samples = [
        ({**PRIORITY, "t": "priority", "turn": n}, 1.0) for n in (1, 5, 20)
    ] + [({**PRIORITY, "t": "priority", "turn": 0}, 1.0)]
    stats = holdout_loss(net, samples, batch=4)
    assert stats is not None
    assert set(stats["value_by_turn"]) == {"1-3", "4-8", "17+", "unknown"}
    assert stats["value_by_turn_n"] == {
        "1-3": 1,
        "4-8": 1,
        "17+": 1,
        "unknown": 1,
    }
    assert (
        abs(sum(stats["value_by_turn"].values()) / 4 - stats["value"]) < 1e-4
    )


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
    assert set(meta["holdout"]) == {
        "loss",
        "policy",
        "value",
        "used_frac",
        "value_by_turn",
        "value_by_turn_n",
    }
    assert set(meta["holdout"]["value_by_turn"]) <= {
        "1-3",
        "4-8",
        "9-16",
        "17+",
        "unknown",
    }
    assert sum(meta["holdout"]["value_by_turn_n"].values()) == (
        meta["holdout_examples"]
    )
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


def test_kl_weights_mean_one_floor_and_deterministic() -> None:
    torch.manual_seed(2)
    net = ForgePointerNet()
    samples = flatten([_game(0, n=3)])
    weights = kl_weights(net, samples, eps=0.01)
    assert len(weights) == len(samples)
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-6
    # unusable decisions (bad shape) weight exactly 1
    weights_bad = kl_weights(
        net, samples + [({**ATTACK, "pi": [0.5]}, 1.0)], eps=0.01
    )
    assert weights_bad[-1] == 1.0
    # a bigger eps floor shrinks the spread toward uniform
    assert max(kl_weights(net, samples, eps=10.0)) < max(weights)
    # deterministic on the same net
    assert kl_weights(net, samples, eps=0.01) == weights


def test_train_epochs_weighted_resamples_and_rejects_mismatch() -> None:
    samples = flatten([_game(0, n=2)])
    weights = [1.0] * len(samples)
    torch.manual_seed(4)
    net = ForgePointerNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    a = train_epochs(
        net, opt, samples, epochs=1, batch=2, seed=6, weights=weights
    )
    torch.manual_seed(4)
    net2 = ForgePointerNet()
    opt2 = torch.optim.Adam(net2.parameters(), lr=3e-4)
    b = train_epochs(
        net2, opt2, samples, epochs=1, batch=2, seed=6, weights=weights
    )
    assert a["used_frac"] == b["used_frac"] == 1.0
    with pytest.raises(ValueError):
        train_epochs(net, opt, samples, epochs=1, weights=[1.0])


def test_sample_source_defaults_to_net() -> None:
    assert sample_source({"source": "expert"}) == "expert"
    assert sample_source({}) == "net"


def test_apply_expert_weight_scales_expert_samples() -> None:
    samples = [
        ({"source": "net"}, 1.0),
        ({"source": "expert"}, -1.0),
        ({}, 0.0),
    ]
    assert apply_expert_weight(None, samples, 1.0) is None
    assert apply_expert_weight(None, samples, 2.0) == [1.0, 2.0, 1.0]
    assert apply_expert_weight([3.0, 3.0, 3.0], samples, 0.5) == [
        3.0,
        1.5,
        3.0,
    ]


def _expert_game(seed: int) -> dict:
    return {
        "seed": seed,
        "decisions": [{**PRIORITY, "source": "expert"}],
        "returns": [1.0],
        "winner": 0,
    }


def test_imitation_phase_trains_on_expert_rows_only(tmp_path: Path) -> None:
    rows = [json.dumps(_expert_game(s)) + "\n" for s in range(8)]
    rows += [json.dumps(_game(s)) + "\n" for s in range(8)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0.25",
                "--imitation-epochs",
                "2",
            ]
        )
    )
    samples = flatten(load_selfplay([data], 20000))
    train, holdout = carve_holdout(samples, 0.25, 1)
    n_train = sum(1 for d, _ in train if sample_source(d) == "expert")
    n_hold = sum(1 for d, _ in holdout if sample_source(d) == "expert")
    im = meta["imitation"]
    assert im["epochs"] == 2 and im["samples"] == n_train
    assert "skipped" not in im
    assert meta["expert_samples"] == n_train
    if n_hold:
        assert im["holdout_expert_pre"] is not None
        assert set(meta["holdout_expert"]) == {
            "loss",
            "policy",
            "value",
            "used_frac",
            "value_by_turn",
            "value_by_turn_n",
        }
    assert len(im["history"]) == 2


def test_imitation_phase_skips_without_expert_rows(tmp_path: Path) -> None:
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(json.dumps(_game(s)) + "\n" for s in range(3)))
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(tmp_path / "r1"),
                "--imitation-epochs",
                "2",
            ]
        )
    )
    assert meta["imitation"] == {
        "epochs": 2,
        "skipped": "no expert rows",
    }
    plain = run(
        build_parser().parse_args(
            ["--data", str(data), "--out", str(tmp_path / "r2")]
        )
    )
    assert "imitation" not in plain


def test_imitation_lr_scale_recorded_and_restored(tmp_path: Path) -> None:
    rows = [json.dumps(_expert_game(s)) + "\n" for s in range(8)]
    rows += [json.dumps(_game(s)) + "\n" for s in range(8)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "r1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0.25",
                "--imitation-epochs",
                "2",
                "--imitation-lr-scale",
                "0.3",
            ]
        )
    )
    assert meta["imitation"]["lr_scale"] == 0.3
    ckpt = torch.load(out / "last.pt", weights_only=False)
    assert ckpt["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-4)


def test_imitation_lr_scale_default_is_one(tmp_path: Path) -> None:
    rows = [json.dumps(_expert_game(s)) + "\n" for s in range(8)]
    rows += [json.dumps(_game(s)) + "\n" for s in range(8)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(tmp_path / "r2"),
                "--holdout-frac",
                "0.25",
                "--imitation-epochs",
                "2",
            ]
        )
    )
    assert meta["imitation"]["lr_scale"] == 1.0


def test_blend_value_targets_default_passthrough() -> None:
    net = ForgePointerNet()
    samples = [(dict(PRIORITY), 1.0), (dict(ATTACK), -1.0)]
    assert blend_value_targets(net, samples, 0.0) is None


def test_blend_value_targets_rejects_full_blend() -> None:
    net = ForgePointerNet()
    with pytest.raises(ValueError):
        blend_value_targets(net, [(dict(PRIORITY), 1.0)], 1.0)


def test_blend_value_targets_blends_selfplay_keeps_expert() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet()
    expert = {**PRIORITY, "source": "expert"}
    samples = [(dict(PRIORITY), 1.0), (expert, -1.0)]
    out = blend_value_targets(net, samples, 0.5)
    assert out is not None
    blended, meta = out
    frozen = copy.deepcopy(net).eval()
    with torch.no_grad():
        v = float(frozen.distill(dict(PRIORITY)).value)
    assert blended[0][1] == pytest.approx(0.5 + 0.5 * v, abs=1e-6)
    assert blended[1][1] == -1.0
    assert meta["value_target_blend"] == 0.5
    assert meta["value_target_rows"] == 1


def test_search_value_targets_default_passthrough() -> None:
    samples = [(dict(PRIORITY), 1.0), (dict(ATTACK), -1.0)]
    assert search_value_targets(samples, False) is None


def test_search_value_targets_swaps_net_keeps_expert_and_fallback() -> None:
    expert = {**PRIORITY, "source": "expert"}
    net_row = {**PRIORITY, "root_value": 0.4}
    samples = [(net_row, 1.0), (dict(ATTACK), -1.0), (expert, 1.0)]
    out = search_value_targets(samples, True)
    assert out is not None
    swapped, meta = out
    assert swapped[0][1] == pytest.approx(0.4)
    assert swapped[1][1] == -1.0
    assert swapped[2][1] == 1.0
    assert meta == {
        "value_source": "search",
        "value_search_rows": 1,
        "value_source_fallback": 1,
    }


def test_search_value_targets_accepts_planar_nexus_rootvalue() -> None:
    net_row = {**PRIORITY, "rootValue": 0.37}
    expert = {**PRIORITY, "source": "expert", "rootValue": 0.9}
    out = search_value_targets([(net_row, 1.0), (expert, 1.0)], True)
    assert out is not None
    swapped, meta = out
    assert swapped[0][1] == pytest.approx(0.37)
    assert swapped[1][1] == 1.0
    assert meta == {
        "value_source": "search",
        "value_search_rows": 1,
        "value_source_fallback": 0,
    }


def test_run_uses_planar_nexus_camelcase_rootvalue(tmp_path: Path) -> None:
    rows = []
    for s in range(3):
        g = _game(s)
        for d in g["decisions"]:
            d["rootValue"] = 0.25
        rows.append(json.dumps(g) + "\n")
    rows += [json.dumps(_expert_game(s)) + "\n" for s in range(2)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0",
                "--value-source",
                "search",
            ]
        )
    )
    assert meta["value_source"] == "search"
    assert meta["value_search_rows"] == 9
    assert meta["value_source_fallback"] == 0


def test_run_uses_search_targets_and_records_it(tmp_path: Path) -> None:
    rows = []
    for s in range(3):
        g = _game(s)
        for d in g["decisions"]:
            d["root_value"] = 0.25
        rows.append(json.dumps(g) + "\n")
    rows += [json.dumps(_expert_game(s)) + "\n" for s in range(2)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0",
                "--value-source",
                "search",
            ]
        )
    )
    assert meta["value_source"] == "search"
    assert meta["value_search_rows"] == 9
    assert meta["value_source_fallback"] == 0


def test_run_default_adds_no_value_source_keys(tmp_path: Path) -> None:
    data = tmp_path / "round0.jsonl"
    data.write_text(json.dumps(_game(0)) + "\n")
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(["--data", str(data), "--out", str(out)])
    )
    assert "value_source" not in meta
    assert "value_search_rows" not in meta


def test_run_blends_targets_and_keeps_holdout_outcomes(tmp_path: Path) -> None:
    rows = [json.dumps(_game(s)) + "\n" for s in range(6)]
    rows += [json.dumps(_expert_game(s)) + "\n" for s in range(2)]
    data = tmp_path / "round0.jsonl"
    data.write_text("".join(rows))
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                "--holdout-frac",
                "0",
                "--value-target-blend",
                "0.5",
            ]
        )
    )
    assert meta["value_target_blend"] == 0.5
    assert meta["value_target_rows"] == 18


def test_run_default_adds_no_value_target_keys(tmp_path: Path) -> None:
    data = tmp_path / "round0.jsonl"
    data.write_text(json.dumps(_game(0)) + "\n")
    out = tmp_path / "round1"
    meta = run(
        build_parser().parse_args(["--data", str(data), "--out", str(out)])
    )
    assert "value_target_blend" not in meta
    assert "value_target_rows" not in meta
