"""Step-wise environment over the Forge bridge's JSON pipe (#74).

Forge drives the game loop, so the environment is decision-driven: each
``step`` answers the pending decision and returns the next one. One JVM
plays many games; after a game ends, ``reset`` returns the first decision
of the next game.

The Java side is ``tools/forge-bridge`` (``manamind.forge.PipeBench``).
Every decision carries only what the piped seat can see.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

TAG = "@@MM "
DECISION_TYPES = ("priority", "attack", "block")
Message = Dict[str, Any]


class ForgeBridgeError(RuntimeError):
    """The bridge process exited or sent something unexpected."""


@dataclass
class StepResult:
    """What the environment returns after ``reset`` or ``step``."""

    decision: Optional[Message]
    done: bool
    reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


def priority_reply(option: Optional[int]) -> str:
    """Encode a priority choice; ``None`` passes priority."""
    return "-1" if option is None else str(option)


def attack_reply(attackers: Sequence[int]) -> str:
    """Encode the indices of creatures to attack with."""
    return " ".join(str(i) for i in attackers)


def block_reply(pairs: Sequence[tuple[int, int]]) -> str:
    """Encode ``(blocker_index, attacker_index)`` pairs."""
    return " ".join(f"{b}:{a}" for b, a in pairs)


def bridge_command(
    forge_dir: Path,
    bridge_out: Path,
    games: int,
    deck_a: Path,
    deck_b: Path,
    java: str = "java",
    xmx: str = "1500m",
) -> List[str]:
    """Build the JVM command line for ``PipeBench``."""
    jars = sorted(
        forge_dir.glob("forge-gui-desktop-*-jar-with-dependencies.jar")
    )
    if not jars:
        raise ForgeBridgeError(f"no Forge desktop jar in {forge_dir}")
    return [
        java,
        f"-Xmx{xmx}",
        f"-Duser.home={forge_dir / 'home'}",
        "-cp",
        f"{bridge_out}:{jars[0]}",
        "manamind.forge.PipeBench",
        str(games),
        str(deck_a),
        str(deck_b),
    ]


class ForgeEnv:
    """Decision-driven environment over one bridge process.

    Pass ``command`` to run something other than the real JVM (tests use a
    fake driver that speaks the same protocol).
    """

    def __init__(
        self, command: Sequence[str], cwd: Optional[Path] = None
    ) -> None:
        self._command = list(command)
        self._cwd = cwd
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lines: Optional[Iterator[str]] = None
        self._pending: Optional[Message] = None
        self.results: List[Message] = []
        self.summary: Optional[Message] = None

    # -- process plumbing -------------------------------------------------
    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        self._lines = iter(self._proc.stdout)
        first = self._next_message()
        if first.get("t") != "ready":
            raise ForgeBridgeError(f"expected ready, got {first}")

    def _next_message(self) -> Message:
        assert self._lines is not None
        for line in self._lines:
            if line.startswith(TAG):
                try:
                    msg: Message = json.loads(line[len(TAG) :])
                except json.JSONDecodeError as e:
                    raise ForgeBridgeError(
                        f"bad protocol line: {line!r}"
                    ) from e
                return msg
        raise ForgeBridgeError("bridge process exited")

    def _send(self, reply: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(reply + "\n")
        self._proc.stdin.flush()

    def _advance(self) -> StepResult:
        msg = self._next_message()
        t = msg.get("t")
        if t in DECISION_TYPES:
            self._pending = msg
            return StepResult(decision=msg, done=False)
        self._pending = None
        if t == "game_over":
            self.results.append(msg)
            reward = {"win": 1.0, "loss": -1.0}.get(msg.get("result"), 0.0)
            return StepResult(
                decision=None, done=True, reward=reward, info=msg
            )
        if t == "done":
            self.summary = msg
            return StepResult(decision=None, done=True, info=msg)
        raise ForgeBridgeError(f"unexpected message {msg}")

    # -- public API -------------------------------------------------------
    @property
    def finished(self) -> bool:
        """True once the bridge has played all its games."""
        return self.summary is not None

    def reset(self) -> StepResult:
        """Return the first decision of the next game.

        Returns a ``done`` result with no decision if the game ended before
        the piped seat had to decide anything, or if all games are played.
        """
        if self._proc is None:
            self._start()
        if self._pending is not None:
            raise ForgeBridgeError("reset() called mid-game; answer it first")
        if self.finished:
            return StepResult(
                decision=None, done=True, info=self.summary or {}
            )
        return self._advance()

    def step(self, reply: str) -> StepResult:
        """Answer the pending decision and return what comes next."""
        if self._pending is None:
            raise ForgeBridgeError("no pending decision; call reset()")
        self._send(reply)
        return self._advance()

    def close(self) -> None:
        """Stop the bridge process."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def __enter__(self) -> "ForgeEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
