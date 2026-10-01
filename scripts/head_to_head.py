"""Play two trained checkpoints against each other in simple mode.

Each side keeps its own root search (PUCT or Gumbel). Games come in pairs on
the same deal with seats swapped, so neither side gets the first-player edge
more often than the other. Evaluation is greedy: no Dirichlet or Gumbel noise.

Example:
    PYTHONPATH=src python scripts/head_to_head.py \
        --a runs/v7/iter_010.pt --a-search puct \
        --b runs/g1/iter_010.pt --b-search gumbel --games 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_simple import (  # noqa: E402
    MCTSAgent,
    load_checkpoint,
    play_game,
    seed_everything,
)


def resolve(path: Path) -> Path:
    """Accept a checkpoint file, or a directory and take its latest iter."""
    if path.is_dir():
        found = sorted(path.glob("iter_*.pt"))
        if not found:
            raise SystemExit(f"no iter_*.pt checkpoints in {path}")
        return found[-1]
    return path


def match(
    a: Path,
    a_search: str,
    b: Path,
    b_search: str,
    games: int,
    simulations: int,
    seed: int,
) -> Dict[str, object]:
    net_a, net_b = load_checkpoint(a), load_checkpoint(b)
    a_wins = b_wins = draws = 0
    turns = 0
    for game in range(games):
        deal = seed + game // 2
        a_seat = game % 2
        sides = {a_seat: (net_a, a_search), 1 - a_seat: (net_b, b_search)}
        agents = {
            pid: MCTSAgent(
                player_id=pid,
                policy_network=net,
                value_network=net,
                simulations=simulations,
                simulation_time=30.0,
                search=search,
            )
            for pid, (net, search) in sides.items()
        }
        winner: Optional[int]
        winner, game_turns, _ = play_game(agents, seed=deal)
        turns += game_turns
        if winner is None:
            draws += 1
        elif winner == a_seat:
            a_wins += 1
        else:
            b_wins += 1
        print(
            f"game {game + 1:3d}/{games}  A {a_wins}  B {b_wins}  "
            f"draws {draws}",
            flush=True,
        )

    score = (a_wins + 0.5 * draws) / games
    # Wilson interval: stays sensible at 0% or 100%, unlike the normal one.
    z = 1.96
    denom = 1 + z * z / games
    centre = (score + z * z / (2 * games)) / denom
    half = (
        z
        * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games))
        / denom
    )
    return {
        "a": str(a),
        "a_search": a_search,
        "b": str(b),
        "b_search": b_search,
        "games": games,
        "simulations": simulations,
        "seed": seed,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": score,
        "a_score_95ci": [max(0.0, centre - half), min(1.0, centre + half)],
        "mean_turns": turns / games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument(
        "--a-search", choices=("puct", "gumbel"), default="puct"
    )
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument(
        "--b-search", choices=("puct", "gumbel"), default="puct"
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.games % 2:
        parser.error("--games must be even so each deal is played both ways")

    seed_everything(args.seed)
    a, b = resolve(args.a), resolve(args.b)
    print(f"A = {a} ({args.a_search})\nB = {b} ({args.b_search})", flush=True)
    result = match(
        a,
        args.a_search,
        b,
        args.b_search,
        args.games,
        args.simulations,
        args.seed,
    )
    lo, hi = result["a_score_95ci"]
    print(
        f"A score {result['a_score']:.3f}  (95% CI {lo:.3f}-{hi:.3f})  "
        f"A {result['a_wins']}  B {result['b_wins']}  "
        f"draws {result['draws']}  "
        f"mean_turns {result['mean_turns']:.1f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
