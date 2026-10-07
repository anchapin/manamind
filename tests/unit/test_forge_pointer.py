"""ForgePointerNet encoding, heads, replies and one training step (#76)."""

import json
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from manamind.models.forge_pointer import (
    CARD_FEATURES,
    GLOBAL_FEATURES,
    ActOutput,
    ForgePointerNet,
    actor_critic_loss,
    card_features,
    global_features,
    imitation_loss,
    life_potential,
    load_pointer_net,
    shaped_returns,
)
from manamind.training.train_forge import (
    EXPERT_DATA,
    evaluate,
    load_expert_data,
    train,
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
FOREST = {
    "name": "Forest",
    "type": "Basic Land - Forest",
    "cost": "no cost",
    "cmc": 0,
    "creature": False,
    "land": True,
    "power": 0,
    "toughness": 0,
    "tapped": False,
    "sick": False,
}
VIEW = {
    "turn": 3,
    "phase": "MAIN1",
    "active": True,
    "life": [18, 20],
    "hand": [BEAR, FOREST],
    "opp_hand_size": 5,
    "battlefield": [FOREST, BEAR],
    "opp_battlefield": [BEAR],
    "graveyard": ["Shock"],
    "opp_graveyard": [],
    "library": [50, 51],
}
PRIORITY = dict(
    VIEW,
    t="priority",
    options=[
        {
            "text": "Forest | play",
            "land": True,
            "spell": False,
            "card": FOREST,
        },
        {"text": "Bears | cast", "land": False, "spell": True, "card": BEAR},
    ],
)
ATTACK = dict(VIEW, t="attack", phase="COMBAT_DECLARE_ATTACKERS")
ATTACK["options"] = [BEAR, BEAR]
BLOCK = dict(
    VIEW,
    t="block",
    phase="COMBAT_DECLARE_BLOCKERS",
    attackers=[BEAR],
    blockers=[BEAR, BEAR, BEAR],
)


def test_feature_sizes_and_pips() -> None:
    f = card_features(BEAR)
    assert len(f) == CARD_FEATURES
    assert f[1] == 1.0  # creature
    assert len(global_features(PRIORITY)) == GLOBAL_FEATURES
    hybrid = card_features(dict(BEAR, cost="{2}{G/W}{X}"))
    assert len(hybrid) == CARD_FEATURES


def test_heads_shapes() -> None:
    net = ForgePointerNet()
    s = net.encode_state(PRIORITY)
    assert s.shape == (net.state_dim,)
    assert net.priority_logits(s, PRIORITY["options"]).shape == (3,)
    assert net.priority_logits(s, []).shape == (1,)
    assert net.attack_logits(s, ATTACK["options"]).shape == (2,)
    assert net.block_logits(
        s, BLOCK["blockers"], BLOCK["attackers"]
    ).shape == (
        3,
        2,
    )
    empty = net.encode_state({"t": "priority"})
    assert empty.shape == (net.state_dim,)


def test_replies_are_valid() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet()
    for _ in range(20):
        r = net.act(PRIORITY).reply
        assert r in {"0", "1", "-1"}
        r = net.act(ATTACK).reply
        assert all(int(i) in (0, 1) for i in r.split())
        r = net.act(BLOCK).reply
        for pair in r.split():
            b, a = map(int, pair.split(":"))
            assert 0 <= b < 3 and a == 0
    g = net.act(PRIORITY, greedy=True)
    assert g.reply == net.act(PRIORITY, greedy=True).reply


def test_loss_backprops() -> None:
    net = ForgePointerNet()
    steps = [net.act(PRIORITY), net.act(ATTACK), net.act(BLOCK)]
    loss, stats = actor_critic_loss(steps, 1.0)
    loss.backward()
    assert net.pass_token.grad is not None
    assert {"loss", "policy", "value", "entropy"} <= set(stats)


_FAKE_SRC = """
    import json, sys
    VIEW = json.loads(sys.argv[1])
    def out(m):
        print("@@MM " + json.dumps(m), flush=True)
    out({"t": "ready"})
    n = 0
    for g in range(4):
        for kind in ("priority", "attack", "block"):
            out(VIEW[kind]); sys.stdin.readline(); n += 1
        out({"t": "game_over", "game": g,
             "result": "win" if g % 2 else "loss", "turns": 5})
    out({"t": "done", "decisions": n, "fallbacks": 0, "errors": 0})
    """
FAKE = textwrap.dedent(_FAKE_SRC)


def test_train_against_fake_bridge(tmp_path: Path) -> None:
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    views = json.dumps(
        {"priority": PRIORITY, "attack": ATTACK, "block": BLOCK}
    )
    out = tmp_path / "run"
    meta = train(
        [sys.executable, str(script), views],
        None,
        games=4,
        out_dir=out,
        update_every=2,
    )
    assert meta["games"] == 4 and meta["wins"] == 2
    lines = (out / "log.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert sum("updated" in json.loads(x) for x in lines) == 2
    net, m = load_pointer_net(str(out / "last.pt"))
    assert m["games"] == 4
    assert net.act(PRIORITY).reply in {"0", "1", "-1"}


def test_evaluate_scores_checkpoints_without_training(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    views = json.dumps(
        {"priority": PRIORITY, "attack": ATTACK, "block": BLOCK}
    )
    net = ForgePointerNet()
    ckpt = tmp_path / "bc.pt"
    torch.save(
        {
            "network": net.state_dict(),
            "config": {"card_dim": 64, "state_dim": 128},
            "meta": {"games": 1000},
        },
        ckpt,
    )
    before = {k: v.clone() for k, v in net.state_dict().items()}
    out = tmp_path / "eval"
    summary = evaluate(
        [sys.executable, str(script), views],
        None,
        [ckpt],
        games=4,
        out_dir=out,
        labels=["run/bc.pt"],
    )
    rows = summary["eval"]
    assert [r["mode"] for r in rows] == ["greedy", "sample"]
    for r in rows:
        assert r["ckpt"] == "run/bc.pt" and r["trained_games"] == 1000
        assert r["games"] == 4 and r["wins"] == 2 and r["win_rate"] == 0.5
        assert r["avg_turns"] == 5.0
    lines = [
        json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 8 and all(x["phase"] == "eval" for x in lines)
    assert (out / "eval_summary.json").exists()
    assert not (out / "last.pt").exists()
    saved = torch.load(ckpt, weights_only=False)["network"]
    assert all(torch.equal(saved[k], before[k]) for k in before)


def test_evaluate_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate([], None, [], 1, tmp_path, modes=["argmax"])


def test_shaped_returns_default_is_game_result() -> None:
    assert shaped_returns(-1.0, [0.0, 0.0, 0.0]) == [-1.0, -1.0, -1.0]


def test_shaped_returns_credit_life_swings() -> None:
    # lead goes 0 -> +0.5 -> 0; result is a loss
    out = shaped_returns(-1.0, [0.0, 0.5, 0.0], gamma=0.5, shaping_coef=1.0)
    # r = [+0.5, -0.5, -1.0]
    assert out[2] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(-0.5 + 0.5 * -1.0)
    assert out[0] == pytest.approx(0.5 + 0.5 * out[1])
    assert out[0] > out[1]


def test_life_potential() -> None:
    assert life_potential({"life": [18, 20]}) == pytest.approx(-0.1)
    assert life_potential({}) == 0.0


def test_loss_skips_single_choice_steps() -> None:
    v = torch.zeros((), requires_grad=True)
    lp = torch.zeros((), requires_grad=True)
    forced = ActOutput("-1", lp * 1.0, torch.zeros(()), v * 1.0)
    real = ActOutput("0", lp * 1.0 - 0.7, torch.tensor(0.69), v * 1.0)
    _, stats = actor_critic_loss(
        [forced, real],
        -1.0,
        potentials=[0.0, 0.1],
        gamma=0.9,
        shaping_coef=1.0,
    )
    assert stats["choice_frac"] == pytest.approx(0.5)
    assert stats["entropy"] == pytest.approx(0.69, abs=1e-4)
    assert stats["adv_abs"] > 0


# -- imitation warm start (#79) ---------------------------------------------
def _labelled() -> list:
    # ATTACK and BLOCK list identical bears, so only labels that treat them
    # alike are learnable: attack with both, block with none.
    return [
        dict(PRIORITY, expert=1),
        dict(ATTACK, expert=[0, 1]),
        dict(BLOCK, expert=[]),
    ]


def test_imitate_scores_expert_choice() -> None:
    net = ForgePointerNet()
    outs = [net.imitate(d) for d in _labelled()]
    assert all(o.labelled and o.choice for o in outs)
    assert all(float(o.log_prob.detach()) <= 0.0 for o in outs)
    # priority: log-prob of option 1 under the softmax over 2 options + pass
    state = net.encode_state(PRIORITY)
    ref = torch.log_softmax(
        net.priority_logits(state, PRIORITY["options"]), -1
    )[1]
    assert torch.allclose(outs[0].log_prob, ref)


def test_imitate_skips_unusable_labels() -> None:
    net = ForgePointerNet()
    assert not net.imitate(PRIORITY).labelled  # no expert field
    assert not net.imitate(dict(PRIORITY, expert=-2)).labelled
    passing = net.imitate(dict(PRIORITY, options=[], expert=0))
    assert passing.labelled and not passing.choice


def test_imitation_loss_backprops_and_learns() -> None:
    torch.manual_seed(0)
    net = ForgePointerNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    data = _labelled()
    first = None
    for _ in range(60):
        loss, stats = imitation_loss(
            [net.imitate(d) for d in data], [1.0, 1.0, 1.0]
        )
        first = stats["bc_nll"] if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert stats["bc_nll"] < 0.1 * first
    assert stats["bc_acc"] == 1.0 and stats["labelled_frac"] == 1.0
    assert net.act(data[0], greedy=True).reply == "1"
    assert net.act(data[1], greedy=True).reply == "0 1"
    assert net.act(data[2], greedy=True).reply == ""


_EXPERT_FAKE_SRC = """
    import json, sys
    VIEW = json.loads(sys.argv[1])
    expert = "-Dmanamind.expert=true" in sys.argv
    def out(m):
        print("@@MM " + json.dumps(m), flush=True)
    out({"t": "ready"})
    n = 0
    for g in range(6):
        for kind in ("priority", "attack", "block"):
            m = dict(VIEW[kind])
            if not expert:
                m.pop("expert", None)
            out(m)
            reply = sys.stdin.readline().strip()
            assert (reply == "e") == expert, reply
            n += 1
        out({"t": "game_over", "game": g,
             "result": "win" if g % 2 else "loss", "turns": 5})
    out({"t": "done", "decisions": n, "fallbacks": 0, "errors": 0})
    """
EXPERT_FAKE = textwrap.dedent(_EXPERT_FAKE_SRC)


def test_train_imitates_then_switches_to_rl(tmp_path: Path) -> None:
    script = tmp_path / "fake.py"
    script.write_text(EXPERT_FAKE)
    p, a, b = _labelled()
    views = json.dumps({"priority": p, "attack": a, "block": b})
    cmd = [sys.executable, str(script), views]
    out = tmp_path / "run"
    meta = train(
        cmd,
        None,
        games=6,
        out_dir=out,
        update_every=2,
        expert_command=cmd + ["-Dmanamind.expert=true"],
        expert_games=4,
        bc_epoch_count=2,
        bc_batch=4,
    )
    assert meta["games"] == 6 and meta["bc_done"]
    assert meta["rl_games_this_run"] == 2 and meta["wins"] == 1
    recs = [
        json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()
    ]
    phases = [r.get("phase") for r in recs if "game" in r]
    assert phases == ["imitate"] * 4 + ["rl"] * 2
    epochs = [r for r in recs if "bc_epoch" in r]
    assert [r["bc_epoch"] for r in epochs] == [1, 2]
    assert epochs[0]["train_rows"] == 12 and "val_rows" not in epochs[0]
    assert (out / "bc.pt").exists()
    games = load_expert_data(out / EXPERT_DATA)
    assert len(games) == 4 and len(games[0]["decisions"]) == 3
    assert games[1]["returns"] == [1.0, 1.0, 1.0]


def test_resume_past_imitation_goes_straight_to_rl(tmp_path: Path) -> None:
    script = tmp_path / "fake.py"
    script.write_text(EXPERT_FAKE)
    p, a, b = _labelled()
    cmd = [
        sys.executable,
        str(script),
        json.dumps({"priority": p, "attack": a, "block": b}),
    ]
    out = tmp_path / "run"
    train(
        cmd,
        None,
        games=2,
        out_dir=out,
        expert_command=cmd + ["-Dmanamind.expert=true"],
        expert_games=2,
    )
    meta = train(
        cmd,
        None,
        games=2,
        out_dir=out,
        resume=out / "last.pt",
        expert_command=cmd + ["-Dmanamind.expert=true"],
        expert_games=2,
    )
    assert meta["games"] == 4 and meta["rl_games_this_run"] == 2
