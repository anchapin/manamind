"""ForgeEnv against a fake bridge that speaks the PipeBench protocol."""

import sys
import textwrap
from pathlib import Path

import pytest

from manamind.forge_interface import (
    ForgeBridgeError,
    ForgeEnv,
    attack_reply,
    block_reply,
    priority_reply,
)

_FAKE_SRC = """
    import json, sys

    def out(m):
        print("Forge log noise", flush=True)
        print("@@MM " + json.dumps(m), flush=True)

    out({"t": "ready"})
    replies = []
    # game 0: priority, attack, block, then win
    out({"t": "priority", "turn": 1, "options": [{"text": "Forest"}]})
    replies.append(sys.stdin.readline().strip())
    out({"t": "attack", "turn": 3, "options": [{"name": "Bear"}]})
    replies.append(sys.stdin.readline().strip())
    out({"t": "block", "turn": 4, "attackers": [{}], "blockers": [{}, {}]})
    replies.append(sys.stdin.readline().strip())
    out({"t": "game_over", "game": 0, "result": "win", "replies": replies})
    # game 1: one priority then loss
    out({"t": "priority", "turn": 1, "options": []})
    replies.append(sys.stdin.readline().strip())
    out({"t": "game_over", "game": 1, "result": "loss", "replies": replies})
    out({"t": "done", "decisions": 4, "fallbacks": 0, "errors": 0})
    """
FAKE = textwrap.dedent(_FAKE_SRC)


@pytest.fixture
def fake_cmd(tmp_path: Path) -> list:
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE)
    return [sys.executable, str(script)]


def test_replies_encode() -> None:
    assert priority_reply(None) == "-1"
    assert priority_reply(2) == "2"
    assert attack_reply([0, 2]) == "0 2"
    assert attack_reply([]) == ""
    assert block_reply([(1, 0), (0, 0)]) == "1:0 0:0"


def test_two_games_end_to_end(fake_cmd: list) -> None:
    with ForgeEnv(fake_cmd) as env:
        r = env.reset()
        assert r.decision is not None and r.decision["t"] == "priority"
        r = env.step(priority_reply(0))
        assert r.decision["t"] == "attack"
        r = env.step(attack_reply([0]))
        assert r.decision["t"] == "block"
        r = env.step(block_reply([(1, 0)]))
        assert r.done and r.reward == 1.0
        assert r.info["replies"] == ["0", "0", "1:0"]

        r = env.reset()
        assert r.decision["t"] == "priority"
        r = env.step(priority_reply(None))
        assert r.done and r.reward == -1.0

        r = env.reset()
        assert r.done and r.decision is None and env.finished
        assert env.summary["decisions"] == 4
        assert [g["result"] for g in env.results] == ["win", "loss"]


def test_step_without_decision_raises(fake_cmd: list) -> None:
    with ForgeEnv(fake_cmd) as env:
        with pytest.raises(ForgeBridgeError):
            env.step("0")


def test_reset_mid_game_raises(fake_cmd: list) -> None:
    with ForgeEnv(fake_cmd) as env:
        env.reset()
        with pytest.raises(ForgeBridgeError):
            env.reset()


def test_process_exit_is_an_error(tmp_path: Path) -> None:
    script = tmp_path / "dies.py"
    script.write_text('print(\'@@MM {"t": "ready"}\', flush=True)\n')
    with ForgeEnv([sys.executable, str(script)]) as env:
        with pytest.raises(ForgeBridgeError):
            env.reset()
