"""Preflight a ``train_simple.py`` run before it ties up the runner (#68).

Parses exactly the args ``train_simple.py`` takes, then:

1. rejects anything outside hard bounds (exit 2) and warns on anything
   outside the recommended band;
2. times a tiny slice on this machine (one self-play game, one game vs
   RandomAgent, one training step) and projects wall time for the whole
   run; exits 3 when the projection exceeds ``--timeout-minutes``.

Usage::

    python scripts/preflight.py [--timeout-minutes 1440] \\
        [--json-out preflight.json] -- <train_simple args>

The projection assumes games scale linearly with ``--workers``, so it is
a lower bound when workers > 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT = Path(__file__).resolve().parent / "train_simple.py"


def _load_train_simple() -> Any:
    loaded = sys.modules.get("train_simple")
    if loaded is not None and Path(
        getattr(loaded, "__file__", "") or ""
    ).resolve() == (_SCRIPT):
        # Reuse it: worker processes pickle its functions by module name.
        return loaded
    spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_simple"] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Bound:
    """Hard limits (reject) and a recommended band (warn)."""

    hard_min: float
    hard_max: float
    rec_min: float
    rec_max: float


# Recommended bands come from the runs that produced the shipped tiers
# (#49/#57/#46: 24 games, 40 sims, 60 iterations, 40 eval games).
BOUNDS: Dict[str, Bound] = {
    "iterations": Bound(1, 500, 6, 100),
    "games": Bound(2, 1000, 12, 64),
    "eval_games": Bound(2, 2000, 20, 160),
    "simulations": Bound(1, 2000, 10, 200),
    "workers": Bound(1, 16, 1, 4),
    "lr": Bound(1e-6, 0.1, 1e-4, 3e-3),
    "lr_min": Bound(0.0, 0.1, 0.0, 1e-3),
    "train_batches": Bound(1, 500, 4, 32),
    "batch_size": Bound(1, 4096, 16, 256),
    # 20,000 is the agreed cap; don't raise it without Alex.
    "buffer_size": Bound(32, 20000, 1000, 20000),
    "anchor_games": Bound(2, 2000, 40, 160),
}


def check_bounds(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for parsed train_simple args."""
    errors: List[str] = []
    warnings: List[str] = []
    for name, b in BOUNDS.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if not b.hard_min <= value <= b.hard_max:
            errors.append(
                f"{flag}={value} is outside the hard bounds "
                f"[{b.hard_min:g}, {b.hard_max:g}]"
            )
        elif not b.rec_min <= value <= b.rec_max:
            warnings.append(
                f"{flag}={value} is outside the recommended band "
                f"[{b.rec_min:g}, {b.rec_max:g}]"
            )
    for name in ("games", "eval_games", "anchor_games"):
        value = getattr(args, name, None)
        if value is not None and value % 2:
            warnings.append(
                f"--{name.replace('_', '-')}={value} is odd; seats alternate, "
                "so one seat gets an extra game"
            )
    if getattr(args, "batch_size", 0) > getattr(args, "buffer_size", 1e9):
        errors.append("--batch-size is larger than --buffer-size")
    if (
        getattr(args, "plateau", 0) > 0
        and getattr(args, "anchor", None) is None
    ):
        errors.append("--plateau needs --anchor")
    ema = getattr(args, "ema_decay", 0.0)
    if not 0.0 <= ema < 1.0:
        errors.append(f"--ema-decay={ema} must be in [0, 1)")
    return errors, warnings


@dataclass
class SliceTimes:
    """Seconds for one of each unit of work, measured on this machine."""

    selfplay_game: float
    random_game: float
    train_step: float


def project_seconds(
    args: argparse.Namespace, t: SliceTimes, done: int = 0
) -> float:
    """Projected wall time for the iterations still to run.

    ``done`` is the last iteration a resumable checkpoint already holds,
    so a resumed run is projected for what is left, not from scratch.
    """
    workers = max(1, int(args.workers))
    n_ref = (
        args.eval_games
        if args.ref_eval_games is None
        else (args.ref_eval_games)
    )
    per_iter = (
        args.games * t.selfplay_game
        + args.eval_games * t.random_game
        # reference games: both seats search, same cost as self-play
        + n_ref * t.selfplay_game
    ) / workers + args.train_batches * t.train_step
    done = min(max(0, int(done)), int(args.iterations))
    total = (args.iterations - done) * per_iter
    if args.anchor is not None and args.anchor_every > 0:
        rounds = (
            args.iterations // args.anchor_every - done // args.anchor_every
        )
        total += rounds * args.anchor_games * t.selfplay_game / workers
    return total


def done_iterations(ts: Any, checkpoint_dir: Optional[Path]) -> int:
    """Last iteration the newest resumable checkpoint holds (0 if none).

    Uses train_simple's own ``latest_resumable`` so the preflight resumes
    from the same checkpoint ``--resume`` would.
    """
    path = ts.latest_resumable(checkpoint_dir)
    if path is None:
        return 0
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["iteration"])


def time_slice(ts: Any, args: argparse.Namespace) -> SliceTimes:
    """Play one self-play game, one RandomAgent game, one train step."""
    import torch

    from manamind.core.agent import MCTSAgent
    from manamind.models.policy_value_network import PolicyValueLoss
    from manamind.rules.simple import build_simple_network

    ts.seed_everything(args.seed)
    probe = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    size = len(probe.action_space.action_to_id)
    net = build_simple_network(action_space_size=size)
    nets = {"net": net}

    def one(kind: str) -> Tuple[float, Any]:
        job = ts.GameJob(
            kind,
            args.seed,
            0,
            args.simulations,
            args.fpu_reduction,
            args.search,
            args.value_mix,
        )
        start = time.perf_counter()
        out = ts.run_games([job], nets, workers=1)
        return time.perf_counter() - start, out[0]

    t_self, outcome = one("selfplay")
    t_rand, _ = one("random")

    # Outcome = (winner, turns, examples, truncated)
    examples = list(outcome[2])
    if not examples:
        raise RuntimeError("self-play slice produced no training examples")
    while len(examples) < args.batch_size:
        examples = examples + examples
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = PolicyValueLoss(value_weight=1.0, l2_reg=1e-4)
    start = time.perf_counter()
    ts.train_on_buffer(
        net, opt, loss_fn, examples, batch_size=args.batch_size, epochs=1
    )
    t_step = time.perf_counter() - start
    return SliceTimes(t_self, t_rand, t_step)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        own, train_args = argv[:split], argv[split + 1 :]
    else:
        own, train_args = [], argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout-minutes", type=float, default=1440.0)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="the run's checkpoint dir; on resume, project only what is left",
    )
    ap.add_argument(
        "--no-timing",
        action="store_true",
        help="bounds check only (no games played)",
    )
    mine = ap.parse_args(own)

    ts = _load_train_simple()
    args = ts.build_parser().parse_args(train_args)
    errors, warnings = check_bounds(args)
    report: Dict[str, Any] = {
        "args": train_args,
        "errors": errors,
        "warnings": warnings,
        "timeout_minutes": mine.timeout_minutes,
    }
    for w in warnings:
        print(f"preflight warning: {w}")
    for e in errors:
        print(f"preflight ERROR: {e}")
    code = 2 if errors else 0
    done = 0 if errors else done_iterations(ts, mine.checkpoint_dir)
    report["resumed_from_iteration"] = done
    if done:
        print(
            f"preflight: resuming after iteration {done} of "
            f"{args.iterations}; projecting the remaining "
            f"{max(0, args.iterations - done)}"
        )
    if not errors and not mine.no_timing:
        t = time_slice(ts, args)
        projected = project_seconds(args, t, done)
        report.update(
            {
                "slice_seconds": {
                    "selfplay_game": round(t.selfplay_game, 3),
                    "random_game": round(t.random_game, 3),
                    "train_step": round(t.train_step, 4),
                },
                "projected_minutes": round(projected / 60.0, 1),
            }
        )
        print(
            f"preflight: one self-play game {t.selfplay_game:.2f}s, "
            f"one random game {t.random_game:.2f}s, "
            f"one train step {t.train_step * 1000:.0f}ms"
        )
        print(
            f"preflight: projected {projected / 60:.0f} min for the "
            f"{'remaining' if done else 'full'} run "
            f"(limit {mine.timeout_minutes:.0f} min; lower bound with "
            f"--workers {args.workers})"
        )
        if projected / 60.0 > mine.timeout_minutes:
            print("preflight ERROR: projection exceeds the job timeout")
            code = 3
    if mine.json_out is not None:
        mine.json_out.parent.mkdir(parents=True, exist_ok=True)
        mine.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
