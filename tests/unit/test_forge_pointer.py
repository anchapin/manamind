"""ForgePointerNet encoding, heads, replies and one training step (#76)."""

import json
import sys
import textwrap
from pathlib import Path

import torch

from manamind.models.forge_pointer import (
    CARD_FEATURES,
    GLOBAL_FEATURES,
    ForgePointerNet,
    actor_critic_loss,
    card_features,
    global_features,
    load_pointer_net,
)
from manamind.training.train_forge import train

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
    assert set(stats) == {"loss", "policy", "value", "entropy"}


FAKE = textwrap.dedent("""
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
    """)


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
