"""Self-play round loop for ForgePointerNet (#87 step 4).

Each round:

1. Planar Nexus plays ``--games`` self-play games with the champion
   (``scripts/selfplay-forge.ts``) into ``<run>/buffer/round_NNNN.jsonl.gz``.
2. ``train_selfplay`` trains a candidate, starting from the champion, on the
   newest ``--buffer-games`` games (default 20,000) and exports ONNX.
3. ``scripts/gate-forge.ts`` plays the candidate against the champion.
4. The candidate becomes champion when the gate promotes it. The ladder
   Elo then rises by the gate's Elo difference; otherwise it stays flat.

Kill switch: the loop stops for good once the Elo has been flat for
``--flat-rounds`` rounds in a row (default 3). State lives in
``<run>/loop.json``, so dispatching again resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from manamind.models.forge_pointer import ForgePointerNet, load_pointer_net
from manamind.training.train_selfplay import BUFFER_GAMES
from manamind.training.train_selfplay import build_parser as train_parser
from manamind.training.train_selfplay import run as train_run
from manamind.training.train_selfplay import save_checkpoint

Runner = Callable[[List[str], Path], None]
Trainer = Callable[[List[str]], Dict[str, Any]]

GATE_SEED_BASE = 1_000_000


def run_cmd(cmd: List[str], cwd: Path) -> None:
    """Run one Planar Nexus script; raise if it fails."""
    subprocess.run(cmd, cwd=cwd, check=True)


def train_cmd(argv: List[str]) -> Dict[str, Any]:
    """Train one candidate in-process with ``train_selfplay``."""
    return train_run(train_parser().parse_args(argv))


def elo_diff(score: float, games: int) -> float:
    """Elo gap implied by a score; clamped half a game from 0 and 1."""
    if games <= 0:
        return 0.0
    eps = 0.5 / games
    s = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s))


def new_state() -> Dict[str, Any]:
    return {
        "round": 0,
        "champion": "rounds/0000",
        "elo": 0.0,
        "flat": 0,
        "rises": 0,
        "games_total": 0,
        "stopped": None,
        "history": [],
    }


def load_state(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "loop.json"
    if not path.exists():
        return None
    state: Dict[str, Any] = json.loads(path.read_text())
    return state


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    tmp = run_dir / "loop.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(run_dir / "loop.json")


def bootstrap(run_dir: Path, init: Optional[Path]) -> Path:
    """Make ``rounds/0000`` (last.pt + ONNX) from ``init`` or a fresh net."""
    from manamind.models.forge_pointer_onnx import export

    out = run_dir / "rounds" / "0000"
    out.mkdir(parents=True, exist_ok=True)
    if init is not None:
        net, _ = load_pointer_net(str(init))
        shutil.copyfile(init, out / "last.pt")
    else:
        net = ForgePointerNet()
        opt = torch.optim.Adam(net.parameters())
        save_checkpoint(net, opt, out, {"phase": "selfplay", "init": None})
    export(net, out)
    return out


def table(state: Dict[str, Any]) -> str:
    """Markdown results table, one row per round."""
    rows = [
        "| Round | Games | Gate score | 95% CI | Result | Elo |",
        "|---:|---:|---:|:---:|:---|---:|",
    ]
    for h in state["history"]:
        lo, hi = h["ci95"]
        result = "promoted" if h["promote"] else "kept champion"
        rows.append(
            f"| {h['round']} | {h['games_total']:,} | {h['score']:.3f} "
            f"| [{lo:.3f}, {hi:.3f}] | {result} | {h['elo']:+.0f} |"
        )
    tail = (
        f"\nElo rose in {state['rises']} round(s); "
        f"flat streak {state['flat']}."
    )
    if state["stopped"]:
        tail += f" Stopped: {state['stopped']}."
    return "\n".join(rows) + tail + "\n"


def run_round(
    args: argparse.Namespace,
    state: Dict[str, Any],
    runner: Runner,
    trainer: Trainer,
) -> Dict[str, Any]:
    """Play, train, gate and promote-or-keep for one round."""
    run_dir: Path = args.run_dir
    r = state["round"] + 1
    champ = run_dir / state["champion"]
    cand = run_dir / "rounds" / f"{r:04d}"
    buffer = run_dir / "buffer"
    buffer.mkdir(parents=True, exist_ok=True)
    pn: Path = args.pn_dir
    tsx = ["npx", "tsx"]
    runner(
        tsx
        + [
            "scripts/selfplay-forge.ts",
            "--model",
            str(champ / "forge_pointer.onnx"),
            "--games",
            str(args.games),
            "--seed",
            str(args.seed + (r - 1) * args.games + 1),
            "--sims",
            str(args.sims),
            "--out",
            str(buffer / f"round_{r:04d}.jsonl.gz"),
        ],
        pn,
    )
    trainer(
        [
            "--data",
            str(buffer),
            "--init",
            str(champ / "last.pt"),
            "--out",
            str(cand),
            "--buffer-games",
            str(args.buffer_games),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed + r),
            "--onnx",
        ]
    )
    gate_path = cand / "gate.json"
    runner(
        tsx
        + [
            "scripts/gate-forge.ts",
            "--candidate",
            str(cand / "forge_pointer.onnx"),
            "--baseline",
            str(champ / "forge_pointer.onnx"),
            "--games",
            str(args.gate_games),
            "--seed",
            str(GATE_SEED_BASE + (r - 1) * args.gate_games + 1),
            "--sims",
            str(args.sims),
            "--threshold",
            str(args.threshold),
            "--out",
            str(gate_path),
        ],
        pn,
    )
    gate = json.loads(gate_path.read_text())
    state["round"] = r
    state["games_total"] += args.games
    if gate["promote"]:
        state["champion"] = f"rounds/{r:04d}"
        state["elo"] += elo_diff(gate["score"], gate["games"])
        state["rises"] += 1
        state["flat"] = 0
    else:
        state["flat"] += 1
    state["history"].append(
        {
            "round": r,
            "games_total": state["games_total"],
            "wins": gate["wins"],
            "losses": gate["losses"],
            "draws": gate["draws"],
            "score": gate["score"],
            "ci95": gate["ci95"],
            "promote": bool(gate["promote"]),
            "elo": round(state["elo"], 1),
            "champion": state["champion"],
        }
    )
    if state["flat"] >= args.flat_rounds:
        state["stopped"] = f"Elo flat for {state['flat']} rounds"
    return state


def run(
    args: argparse.Namespace,
    runner: Runner = run_cmd,
    trainer: Trainer = train_cmd,
) -> Dict[str, Any]:
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(run_dir)
    if state is None:
        bootstrap(run_dir, args.init)
        state = new_state()
        save_state(run_dir, state)
    if state["stopped"] and args.force:
        state["stopped"] = None
        state["flat"] = 0
    for _ in range(args.rounds):
        if state["stopped"]:
            break
        state = run_round(args, state, runner, trainer)
        save_state(run_dir, state)
        (run_dir / "loop.md").write_text(table(state))
    (run_dir / "loop.md").write_text(table(state))
    return state


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--pn-dir", type=Path, required=True, help="planar-nexus checkout"
    )
    ap.add_argument(
        "--init", type=Path, default=None, help=".pt for round 0 (new runs)"
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--gate-games", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--buffer-games", type=int, default=BUFFER_GAMES)
    ap.add_argument("--flat-rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="resume after the kill switch"
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    state = run(build_parser().parse_args(argv))
    print(table(state))


if __name__ == "__main__":
    main()
