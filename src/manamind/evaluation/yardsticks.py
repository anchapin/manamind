"""Fixed yardsticks that score every checkpoint the same way (#85).

``forge`` plays each checkpoint against stock Forge AI with the eval-only
mode of ``train_forge`` (#83): ``--games`` games per mode (greedy and
sampled) on the training deck pair, with no updates. The bridge alternates
seats and decks game by game, so an even game count is balanced. Results
go to ``<out>/forge.json`` and ``<out>/forge.md``: win rate, a Wilson 95%
interval, and where that interval sits against the 18% imitation baseline.

``--summary`` re-renders an existing ``eval_summary.json`` without playing,
which the wrapper and the tests use.

Example::

    PYTHONPATH=src python -m manamind.evaluation.yardsticks forge \\
        --forge-dir ~/forge-2.0.15 --java ~/forge-jdk17/bin/java \\
        --runs-dir ~/manamind-runs --ckpts forge_bc/bc.pt,forge_rl/last.pt \\
        --games 200 --out results/yardstick_forge
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORGE_BASELINE = 0.18
BRIDGE = Path(__file__).resolve().parents[3] / "tools" / "forge-bridge"


def wilson_interval(
    wins: int, games: int, z: float = 1.96
) -> Tuple[float, float]:
    """95% Wilson interval for ``wins`` out of ``games``."""
    if games <= 0:
        return float("nan"), float("nan")
    p = wins / games
    centre = p + z * z / (2 * games)
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    denom = 1 + z * z / games
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def versus_baseline(lo: float, hi: float, baseline: float) -> str:
    """Where a 95% interval sits against a baseline win rate."""
    if math.isnan(lo):
        return "no games"
    if lo > baseline:
        return "above"
    if hi < baseline:
        return "below"
    return "within CI"


def forge_rows(
    summary: Dict[str, Any], baseline: float = FORGE_BASELINE
) -> List[Dict[str, Any]]:
    """Add the interval and baseline verdict to ``evaluate``'s rows."""
    rows = []
    for r in summary.get("eval", []):
        games = int(r.get("games") or 0)
        wins = int(r.get("wins") or 0)
        lo, hi = wilson_interval(wins, games)
        rows.append(
            dict(
                r,
                ci_low=None if math.isnan(lo) else round(lo, 4),
                ci_high=None if math.isnan(hi) else round(hi, 4),
                vs_baseline=versus_baseline(lo, hi, baseline),
            )
        )
    return rows


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def forge_table(
    rows: Sequence[Dict[str, Any]], baseline: float = FORGE_BASELINE
) -> str:
    """Markdown table of the Forge leg."""
    lines = [
        f"**vs Forge AI** (baseline {_pct(baseline)}, imitation #83)",
        "",
        "| checkpoint | trained games | mode | wins / games | win rate "
        "| 95% CI | vs baseline | avg turns |",
        "|---|---:|---|---:|---:|---|---|---:|",
    ]
    for r in rows:
        ci = f"{_pct(r.get('ci_low'))} to {_pct(r.get('ci_high'))}"
        trained = r.get("trained_games")
        lines.append(
            "| `{}` | {} | {} | {} / {} | {} | {} | {} | {} |".format(
                r.get("ckpt"),
                "?" if trained is None else trained,
                r.get("mode"),
                r.get("wins"),
                r.get("games"),
                _pct(r.get("win_rate")),
                ci,
                r.get("vs_baseline"),
                "n/a" if r.get("avg_turns") is None else r["avg_turns"],
            )
        )
    return "\n".join(lines) + "\n"


def resolve_ckpts(names: Sequence[str], runs_dir: Path) -> List[Path]:
    """Resolve checkpoint names; relative ones live under ``runs_dir``."""
    paths = []
    for name in names:
        p = Path(name).expanduser()
        if not p.is_absolute():
            if ".." in p.parts:
                raise ValueError(f"checkpoint {name!r} leaves --runs-dir")
            p = runs_dir / p
        paths.append(p)
    return paths


def write_forge(
    summary: Dict[str, Any], out: Path, baseline: float = FORGE_BASELINE
) -> str:
    """Write forge.json and forge.md under ``out``; return the table."""
    rows = forge_rows(summary, baseline)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forge.json").write_text(
        json.dumps(
            {"baseline": baseline, "rows": rows, "secs": summary.get("secs")},
            indent=2,
        )
    )
    table = forge_table(rows, baseline)
    (out / "forge.md").write_text(table)
    return table


def run_forge(args: argparse.Namespace) -> str:
    if args.summary:
        summary = json.loads(Path(args.summary).read_text())
        return write_forge(summary, args.out, args.baseline)
    if not args.forge_dir or not args.ckpts:
        raise SystemExit("--forge-dir and --ckpts are needed to play games")
    # Imported here so --summary works without torch or a Forge install.
    from manamind.forge_interface.forge_env import bridge_command
    from manamind.training.train_forge import evaluate

    names = [c for c in args.ckpts.split(",") if c]
    paths = resolve_ckpts(names, args.runs_dir)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)}")
    cmd = bridge_command(
        args.forge_dir,
        args.bridge_out,
        args.games,
        args.deck_a.resolve(),
        args.deck_b.resolve(),
        java=args.java,
    )
    summary = evaluate(
        cmd,
        args.forge_dir,
        paths,
        args.games,
        args.out / "forge_eval",
        modes=[m for m in args.modes.split(",") if m],
        seed=args.seed,
        labels=names,
    )
    return write_forge(summary, args.out, args.baseline)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="leg", required=True)
    f = sub.add_parser("forge", help="score checkpoints vs Forge AI")
    f.add_argument("--forge-dir", type=Path)
    f.add_argument("--java", default="java")
    f.add_argument("--bridge-out", type=Path, default=BRIDGE / "out")
    f.add_argument("--deck-a", type=Path, default=BRIDGE / "decks" / "rg.dck")
    f.add_argument("--deck-b", type=Path, default=BRIDGE / "decks" / "ub.dck")
    f.add_argument("--ckpts", help="comma list of checkpoints")
    f.add_argument("--runs-dir", type=Path, default=Path("."))
    f.add_argument("--games", type=int, default=200)
    f.add_argument("--modes", default="greedy,sample")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--baseline", type=float, default=FORGE_BASELINE)
    f.add_argument(
        "--summary",
        type=Path,
        help="render an existing eval_summary.json instead of playing",
    )
    f.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.leg == "forge":
        print(run_forge(args), end="")


if __name__ == "__main__":
    main()
