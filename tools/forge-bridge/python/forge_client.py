"""Drive a Forge seat from Python over the PipeBench JSON protocol (#74).

Usage:
    python forge_client.py --forge-dir /path/to/forge --games 20 rg.dck ub.dck
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

TAG = "@@MM "
Decision = Dict[str, Any]


def random_policy(rng: random.Random) -> Callable[[Decision], str]:
    """Uniform random over the protocol's legal choices."""

    def act(d: Decision) -> str:
        if d["t"] == "priority":
            return str(rng.randrange(len(d["options"]) + 1))
        if d["t"] == "attack":
            picks = [i for i in range(len(d["options"])) if rng.random() < 0.5]
            return " ".join(map(str, picks))
        if d["t"] == "block":
            pairs = []
            for b in range(len(d["blockers"])):
                a = rng.randrange(len(d["attackers"]) + 1)
                if a < len(d["attackers"]):
                    pairs.append(f"{b}:{a}")
            return " ".join(pairs)
        raise ValueError(f"unknown decision {d['t']}")

    return act


class ForgeSession:
    """One long-lived JVM running ``games`` games with one piped seat."""

    def __init__(
        self,
        forge_dir: Path,
        bridge_out: Path,
        games: int,
        deck_a: Path,
        deck_b: Path,
        java: str = "java",
        xmx: str = "1500m",
    ) -> None:
        jar = next(
            forge_dir.glob("forge-gui-desktop-*-jar-with-dependencies.jar")
        )
        cmd = [
            java,
            f"-Xmx{xmx}",
            f"-Duser.home={forge_dir / 'home'}",
            "-cp",
            f"{bridge_out}:{jar}",
            "manamind.forge.PipeBench",
            str(games),
            str(deck_a),
            str(deck_b),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=forge_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def messages(self) -> Iterator[Decision]:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.startswith(TAG):
                yield json.loads(line[len(TAG) :])

    def send(self, reply: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(reply + "\n")
        self.proc.stdin.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge-dir", type=Path, required=True)
    ap.add_argument(
        "--bridge-out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out",
    )
    ap.add_argument("--java", default="java")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("deck_a", type=Path)
    ap.add_argument("deck_b", type=Path)
    args = ap.parse_args()

    policy = random_policy(random.Random(args.seed))
    s = ForgeSession(
        args.forge_dir,
        args.bridge_out,
        args.games,
        args.deck_a,
        args.deck_b,
        java=args.java,
    )
    results: List[Decision] = []
    decisions = 0
    py_secs = 0.0
    t_start = None
    for msg in s.messages():
        t = msg["t"]
        if t == "ready":
            t_start = time.time()
        elif t in ("priority", "attack", "block"):
            t0 = time.time()
            reply = policy(msg)
            py_secs += time.time() - t0
            decisions += 1
            s.send(reply)
        elif t == "game_over":
            results.append(msg)
            print(json.dumps(msg), flush=True)
        elif t == "done":
            wall = time.time() - (t_start or time.time())
            wins = sum(r["result"] == "win" for r in results)
            crashes = sum(r["result"] == "crash" for r in results)
            print(
                f"SUMMARY games={len(results)} wins={wins} crashes={crashes} "
                f"decisions={decisions} fallbacks={msg['fallbacks']} "
                f"errors={msg['errors']} wall={wall:.1f}s "
                f"decisions/s={decisions / wall:.0f} "
                f"policy_secs={py_secs:.2f}",
                flush=True,
            )
    s.proc.wait()


if __name__ == "__main__":
    main()
