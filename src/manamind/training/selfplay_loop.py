"""Self-play round loop for ForgePointerNet (#87 step 4).

Each round:

1. Planar Nexus plays ``--games`` self-play games with the champion
   (``scripts/selfplay-forge.ts``) into ``<run>/buffer/round_NNNN.jsonl.gz``.
2. ``train_selfplay`` trains a candidate, starting from the champion, on the
   newest ``--buffer-games`` games (default 20,000) and exports ONNX.
3. ``scripts/gate-forge.ts`` plays the candidate against the champion.
4. The candidate becomes champion when the gate promotes it. The ladder
   Elo then rises by the gate's Elo difference; otherwise it stays flat.

Expert anchor (``--anchor-games N``, off at 0): head-to-head gains alone can
overfit to the champion's own lineage (round 9 of ``selfplay_rg`` beat
bc.pt 0.725 head to head yet lost ground to Forge and the Planar Nexus
Expert AI, #84). With the anchor on, a candidate the gate would promote also
plays ``N`` games against the Expert AI (``scripts/yardstick-pn.ts``, the
same fixed seeds every time, so each comparison sees the same deals). It is
promoted only if its Expert score is at least the best champion's Expert
score minus ``--anchor-margin``; otherwise the round counts as flat.

Expert games (``--expert-games N``, off at 0, #96): each round also plays
``N`` champion-vs-Expert games (``selfplay-forge.ts --opponent expert``,
only the net's decisions recorded) into
``<run>/buffer/round_NNNN_expert.jsonl.gz``, so training keeps seeing an
outside opponent instead of only its own lineage.

Decks (``--deck-a`` / ``--deck-b``): passed to the self-play, Expert-game
and gate scripts. ``red`` / ``green`` are the #2614 Mono-Red Aggro and
Mono-Green Landfall decks the yardsticks measure; left unset, the scripts
use the simulator's vanilla ``aggro`` / ``midrange`` decks.

Move pick (``--pick``, #96): ``sample`` (default) draws each move from the
search policy; ``gumbel`` plays the search's own pick under Gumbel root
noise (Gumbel MuZero), in self-play, Expert games and the gate. Sampling
the search policy played about as well as the raw policy (0.535 vs bc.pt's
prior, against 0.760 for the search's pick).

Kill switch: the loop stops for good once the Elo has been flat for
``--flat-rounds`` rounds in a row (default 3). State lives in
``<run>/loop.json``, so dispatching again resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
ANCHOR_SEED = 2_000_001
# Champion-vs-Expert training games: seeds clear of self-play and gates.
EXPERT_SEED_BASE = 3_000_000
DECK_NAME = re.compile(r"^[a-z][a-z-]*$")


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


def _result(h: Dict[str, Any]) -> str:
    if h["promote"]:
        return "promoted"
    if h.get("vetoed"):
        return "kept champion (Expert regression)"
    return "kept champion"


def table(state: Dict[str, Any]) -> str:
    """Markdown results table, one row per round."""
    anchored = any("expert" in h for h in state["history"])
    head = "| Round | Games | Gate score | 95% CI | Result | Elo |"
    rule = "|---:|---:|---:|:---:|:---|---:|"
    if anchored:
        head += " Expert |"
        rule += "---:|"
    rows = [head, rule]
    for h in state["history"]:
        lo, hi = h["ci95"]
        row = (
            f"| {h['round']} | {h['games_total']:,} | {h['score']:.3f} "
            f"| [{lo:.3f}, {hi:.3f}] | {_result(h)} | {h['elo']:+.0f} |"
        )
        if anchored:
            ex = h.get("expert")
            row += f" {ex:.3f} |" if ex is not None else " - |"
        rows.append(row)
    tail = (
        f"\nElo rose in {state['rises']} round(s); "
        f"flat streak {state['flat']}."
    )
    best = state.get("anchor_best")
    if best is not None:
        tail += f" Best champion Expert score {best:.3f}."
    if state["stopped"]:
        tail += f" Stopped: {state['stopped']}."
    return "\n".join(rows) + tail + "\n"


def expert_score(
    args: argparse.Namespace, model_dir: Path, runner: Runner
) -> float:
    """Expert AI score for ``model_dir``'s ONNX; cached in anchor.json."""
    out = model_dir / "anchor.json"
    if not out.exists():
        cmd = [
            "npx",
            "tsx",
            "scripts/yardstick-pn.ts",
            "--agent",
            "model",
            "--model",
            str((model_dir / "forge_pointer.onnx").resolve()),
            "--games",
            str(args.anchor_games),
            "--seed",
            str(ANCHOR_SEED),
            "--sims",
            str(args.sims),
            "--out",
            str(out.resolve()),
        ]
        try:
            runner(cmd, args.pn_dir)
        except subprocess.CalledProcessError:
            # yardstick-pn.ts exits 1 when some games errored but still
            # writes its results; a missing file is a real failure.
            if not out.exists():
                raise
    result: Dict[str, Any] = json.loads(out.read_text())
    if int(result["games"]) != args.anchor_games:
        raise SystemExit(
            f"{out} has {result['games']} games, not --anchor-games "
            f"{args.anchor_games}; delete it to re-measure"
        )
    return float(result["score"])


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
    # A round cut short and re-run must not reuse a stale Expert score.
    (cand / "anchor.json").unlink(missing_ok=True)
    pn: Path = args.pn_dir
    tsx = ["npx", "tsx"]
    decks: List[str] = []
    if args.deck_a:
        decks += ["--deck-a", args.deck_a]
    if args.deck_b:
        decks += ["--deck-b", args.deck_b]
    # Gumbel root sampling (#96): play the search's own pick, varied by
    # root noise, in self-play, Expert games and the gate.
    pick = ["--pick", "gumbel"] if args.pick == "gumbel" else []
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
        ]
        + decks
        + pick,
        pn,
    )
    if args.expert_games > 0:
        runner(
            tsx
            + [
                "scripts/selfplay-forge.ts",
                "--model",
                str(champ / "forge_pointer.onnx"),
                "--opponent",
                "expert",
                "--games",
                str(args.expert_games),
                "--seed",
                str(EXPERT_SEED_BASE + (r - 1) * args.expert_games + 1),
                "--sims",
                str(args.sims),
                "--out",
                str(buffer / f"round_{r:04d}_expert.jsonl.gz"),
            ]
            + decks
            + pick,
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
        ]
        + decks
        + pick,
        pn,
    )
    gate = json.loads(gate_path.read_text())
    promote = bool(gate["promote"])
    expert: Optional[float] = None
    vetoed = False
    if args.anchor_games > 0 and promote:
        best = state.get("anchor_best")
        if best is None:
            best = expert_score(args, champ, runner)
            state["anchor_best"] = best
        expert = expert_score(args, cand, runner)
        if expert < best - args.anchor_margin:
            promote = False
            vetoed = True
        else:
            state["anchor_best"] = max(best, expert)
    state["round"] = r
    state["games_total"] += args.games
    if promote:
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
            "promote": promote,
            "elo": round(state["elo"], 1),
            "champion": state["champion"],
        }
    )
    if args.anchor_games > 0:
        state["history"][-1]["expert"] = expert
        state["history"][-1]["vetoed"] = vetoed
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
    ap.add_argument(
        "--anchor-games",
        type=int,
        default=0,
        help="Expert AI games to check a promotion (even; 0 = off)",
    )
    ap.add_argument(
        "--anchor-margin",
        type=float,
        default=0.05,
        help="allowed drop below the best champion's Expert score",
    )
    ap.add_argument(
        "--expert-games",
        type=int,
        default=0,
        help="champion-vs-Expert training games per round (0 = off)",
    )
    ap.add_argument(
        "--deck-a", default=None, help="e.g. red (#2614); default aggro"
    )
    ap.add_argument(
        "--deck-b", default=None, help="e.g. green (#2614); default midrange"
    )
    ap.add_argument(
        "--pick",
        choices=["sample", "gumbel"],
        default="sample",
        help="move pick in self-play, Expert games and the gate: sample "
        "the search policy, or the search's pick under Gumbel root noise",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="resume after the kill switch"
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.anchor_games < 0 or args.anchor_games % 2:
        raise SystemExit("--anchor-games must be 0 or a positive even number")
    if args.expert_games < 0:
        raise SystemExit("--expert-games must be 0 or positive")
    for deck in (args.deck_a, args.deck_b):
        if deck is not None and not DECK_NAME.match(deck):
            raise SystemExit(f"bad deck name {deck!r}")
    state = run(args)
    print(table(state))


if __name__ == "__main__":
    main()
