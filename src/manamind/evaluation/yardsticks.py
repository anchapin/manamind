"""Fixed yardsticks that score every checkpoint the same way (#85).

``forge`` plays each checkpoint against stock Forge AI with the eval-only
mode of ``train_forge`` (#83): ``--games`` games per mode (greedy and
sampled) on the training deck pair, with no updates. The bridge alternates
seats and decks game by game, so an even game count is balanced. Results
go to ``<out>/forge.json`` and ``<out>/forge.md``: win rate, a Wilson 95%
interval, and where that interval sits against the 18% imitation baseline.

``--summary`` re-renders an existing ``eval_summary.json`` without playing,
which the wrapper and the tests use.

``elo`` ranks checkpoints against each other in Planar Nexus: every pair
plays ``--games`` games through ``scripts/gate-forge.ts`` (seats alternate
every game, decks swap every two), and a Bradley-Terry fit (draws count
half, one virtual draw per pair so a clean sweep stays finite) turns the
results into Elo, anchored at the first checkpoint = 0. A checkpoint is an
ONNX file, a directory holding ``forge_pointer.onnx``, or a ``.pt`` that is
exported to ONNX first. Results go to ``<out>/elo.json`` and ``elo.md``.

``pn`` plays each checkpoint against the Planar Nexus Expert AI through
``scripts/yardstick-pn.ts --agent model`` (the #2614 Mono-Red vs
Mono-Green pair, decks swapped per seed): ``--games`` games per mode, the
search's greedy pick and a draw from its policy. Results go to
``<out>/pn.json`` and ``pn.md``: score (a draw counts half), Wilson 95%
interval, and the score with each deck.

Example::

    PYTHONPATH=src python -m manamind.evaluation.yardsticks forge \\
        --forge-dir ~/forge-2.0.15 --java ~/forge-jdk17/bin/java \\
        --runs-dir ~/manamind-runs --ckpts forge_bc/bc.pt,forge_rl/last.pt \\
        --games 200 --out results/yardstick_forge

    PYTHONPATH=src python -m manamind.evaluation.yardsticks elo \\
        --pn-dir ../planar-nexus --runs-dir ~/manamind-runs \\
        --ckpts forge_bc/bc.pt,selfplay_rg/rounds/0003 --games 40 \\
        --out results/yardstick_elo
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


Runner = Callable[[List[str], Path], None]


def run_cmd(cmd: List[str], cwd: Path) -> None:
    """Run one Planar Nexus script; raise if it fails."""
    subprocess.run(cmd, cwd=cwd, check=True)


def fit_elo(
    n_players: int,
    pairs: Sequence[Tuple[int, int, int, float]],
    prior: float = 0.5,
    iters: int = 5000,
) -> List[float]:
    """Bradley-Terry Elo from ``(i, j, games, score_of_j)`` results.

    Draws count half (they are already in the score). ``prior`` adds that
    many virtual wins to each side of every pair, so a sweep stays finite.
    Player 0 is anchored at 0 Elo.
    """
    wins = [0.0] * n_players
    games: Dict[Tuple[int, int], float] = {}
    for i, j, n, score in pairs:
        wins[j] += score * n + prior
        wins[i] += (1.0 - score) * n + prior
        key = (min(i, j), max(i, j))
        games[key] = games.get(key, 0.0) + n + 2 * prior
    gamma = [1.0] * n_players
    for _ in range(iters):
        new = []
        for k in range(n_players):
            denom = sum(
                n / (gamma[a] + gamma[b])
                for (a, b), n in games.items()
                if k in (a, b)
            )
            new.append(wins[k] / denom if denom else gamma[k])
        new = [g / new[0] for g in new]
        done = max(abs(a - b) for a, b in zip(new, gamma)) < 1e-10
        gamma = new
        if done:
            break
    return [400.0 * math.log10(g) for g in gamma]


def onnx_for(ckpt: Path, scratch: Path, index: int) -> Path:
    """ONNX path for a checkpoint, exporting a ``.pt`` when needed."""
    if ckpt.is_dir():
        return ckpt / "forge_pointer.onnx"
    if ckpt.suffix == ".onnx":
        return ckpt
    from manamind.models.forge_pointer import load_pointer_net
    from manamind.models.forge_pointer_onnx import export

    net, _ = load_pointer_net(str(ckpt))
    return export(net, scratch / f"{index:02d}")


def elo_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Checkpoint | Elo | Games | Score |",
        "|---|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: -float(r["elo"])):
        lines.append(
            f"| `{r['ckpt']}` | {r['elo']:+.0f} | {r['games']} "
            f"| {r['score']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_elo(args: argparse.Namespace, runner: Runner = run_cmd) -> str:
    names = [c for c in (args.ckpts or "").split(",") if c]
    if len(names) < 2:
        raise SystemExit("--ckpts needs at least two checkpoints")
    paths = resolve_ckpts(names, args.runs_dir)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)}")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    models = [
        onnx_for(p, out / "onnx", k).resolve() for k, p in enumerate(paths)
    ]
    pairs: List[Tuple[int, int, int, float]] = []
    results: List[Dict[str, Any]] = []
    for n, (i, j) in enumerate(combinations(range(len(paths)), 2)):
        pair_out = (out / "pairs" / f"{i:02d}_{j:02d}.json").resolve()
        pair_out.parent.mkdir(parents=True, exist_ok=True)
        runner(
            [
                "npx",
                "tsx",
                "scripts/gate-forge.ts",
                "--candidate",
                str(models[j]),
                "--baseline",
                str(models[i]),
                "--games",
                str(args.games),
                "--seed",
                str(args.seed + n * args.games + 1),
                "--sims",
                str(args.sims),
                "--threshold",
                "0.5",
                "--out",
                str(pair_out),
            ],
            args.pn_dir,
        )
        g = json.loads(pair_out.read_text())
        pairs.append((i, j, int(g["games"]), float(g["score"])))
        results.append(
            {"a": names[i], "b": names[j], **{k: g[k] for k in RESULT_KEYS}}
        )
    elo = fit_elo(len(paths), pairs)
    rows = []
    for k, name in enumerate(names):
        played = [p for p in pairs if k in (p[0], p[1])]
        n_k = sum(p[2] for p in played)
        pts = sum(p[2] * (p[3] if p[1] == k else 1 - p[3]) for p in played)
        rows.append(
            {
                "ckpt": name,
                "elo": round(elo[k], 1),
                "games": n_k,
                "score": round(pts / n_k, 4) if n_k else 0.0,
            }
        )
    (out / "elo.json").write_text(
        json.dumps({"rows": rows, "pairs": results}, indent=2) + "\n"
    )
    table = elo_table(rows)
    (out / "elo.md").write_text(table)
    return table


RESULT_KEYS = ("games", "wins", "losses", "draws", "score", "ci95")


def _deck_score(d: Dict[str, int]) -> Optional[float]:
    n = d.get("win", 0) + d.get("loss", 0) + d.get("draw", 0)
    return (d.get("win", 0) + d.get("draw", 0) / 2) / n if n else None


def pn_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Checkpoint | Mode | Games | W-L-D | Score | 95% CI "
        "| as Red | as Green | Errors |",
        "|---|---|---:|:---:|---:|:---:|---:|---:|---:|",
    ]
    for r in rows:
        lo, hi = r["ci95"]
        lines.append(
            f"| `{r['ckpt']}` | {r['mode']} | {r['games']} "
            f"| {r['win']}-{r['loss']}-{r['draw']} | {_pct(r['score'])} "
            f"| {_pct(lo)}-{_pct(hi)} | {_pct(r['red'])} "
            f"| {_pct(r['green'])} | {r['errors']} |"
        )
    return "\n".join(lines) + "\n"


def run_pn(args: argparse.Namespace, runner: Runner = run_cmd) -> str:
    names = [c for c in (args.ckpts or "").split(",") if c]
    if not names:
        raise SystemExit("--ckpts is needed")
    if args.games <= 0 or args.games % 2:
        raise SystemExit("--games must be a positive even number")
    modes = [m for m in args.modes.split(",") if m]
    if not modes or any(m not in ("greedy", "sample") for m in modes):
        raise SystemExit("--modes takes greedy and/or sample")
    paths = resolve_ckpts(names, args.runs_dir)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)}")
    out: Path = args.out
    (out / "pn_runs").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for k, (name, path) in enumerate(zip(names, paths)):
        model = onnx_for(path, out / "onnx", k).resolve()
        for mode in modes:
            res_path = (out / "pn_runs" / f"{k:02d}_{mode}.json").resolve()
            cmd = [
                "npx",
                "tsx",
                "scripts/yardstick-pn.ts",
                "--agent",
                "model",
                "--model",
                str(model),
                "--games",
                str(args.games),
                "--seed",
                str(args.seed),
                "--sims",
                str(args.sims),
                "--out",
                str(res_path),
            ]
            if mode == "sample":
                cmd.insert(-2, "--sample")
            try:
                runner(cmd, args.pn_dir)
            except subprocess.CalledProcessError:
                # The script exits 1 when some games errored but still
                # writes its results; a missing file is a real failure.
                if not res_path.exists():
                    raise
            r = json.loads(res_path.read_text())
            by_deck = r.get("byDeck", {})
            rows.append(
                {
                    "ckpt": name,
                    "mode": mode,
                    "games": r["games"],
                    "win": r["win"],
                    "loss": r["loss"],
                    "draw": r["draw"],
                    "score": r["score"],
                    "ci95": r["ci95"],
                    "red": _deck_score(by_deck.get("red", {})),
                    "green": _deck_score(by_deck.get("green", {})),
                    "errors": r.get("errorCount", 0),
                }
            )
    (out / "pn.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    table = pn_table(rows)
    (out / "pn.md").write_text(table)
    return table


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
    e = sub.add_parser("elo", help="Elo ladder between checkpoints (PN)")
    e.add_argument("--pn-dir", type=Path, required=True)
    e.add_argument("--ckpts", help="comma list: .onnx, .pt or run dirs")
    e.add_argument("--runs-dir", type=Path, default=Path("."))
    e.add_argument("--games", type=int, default=40, help="games per pair")
    e.add_argument("--sims", type=int, default=16)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pn", help="score checkpoints vs PN Expert AI")
    p.add_argument("--pn-dir", type=Path, required=True)
    p.add_argument("--ckpts", help="comma list: .onnx, .pt or run dirs")
    p.add_argument("--runs-dir", type=Path, default=Path("."))
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--modes", default="greedy,sample")
    p.add_argument("--sims", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.leg == "forge":
        print(run_forge(args), end="")
    elif args.leg == "elo":
        print(run_elo(args), end="")
    elif args.leg == "pn":
        print(run_pn(args), end="")


if __name__ == "__main__":
    main()
