"""Re-evaluate saved simple-mode checkpoints against RandomAgent.

Separates "the network is weak" from "the search budget is too small":
run the same weights at several simulation counts, with and without
first-play urgency.

    PYTHONPATH=src python scripts/eval_checkpoints.py \\
        checkpoints/seed0/iter_006.pt --sims 10 40 100 --fpu none 0.0 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_simple import (  # noqa: E402
    evaluate,
    load_checkpoint,
    seed_everything,
)


def _fpu(value: str):
    return None if value == "none" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--sims", nargs="+", type=int, default=[10, 40])
    parser.add_argument("--fpu", nargs="+", type=_fpu, default=[None])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in args.checkpoints:
        network = load_checkpoint(path)
        for sims in args.sims:
            for fpu in args.fpu:
                seed_everything(args.seed)
                rate = evaluate(
                    network,
                    games=args.games,
                    simulations=sims,
                    seed=args.seed,
                    fpu_reduction=fpu,
                )
                row = {
                    "checkpoint": str(path),
                    "simulations": sims,
                    "fpu_reduction": fpu,
                    "games": args.games,
                    "win_rate": rate,
                }
                rows.append(row)
                print(
                    f"{path.name}  sims {sims:>4}  fpu {str(fpu):>5}  "
                    f"win_rate {rate:.3f}",
                    flush=True,
                )
                # Write after every cell so a killed run keeps its results.
                if args.out is not None:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
