"""Prediction-guided move selection spike (#106).

Three modes, all eval-only (no training):

``collect``
    Play games with a checkpoint, snapshot mid-game decisions, and for each
    one score the search's top candidate moves by the value head applied to
    engine-simulated successor states. The rules engine is the world model:
    applying a candidate is exact, so the predictor only has to score the
    successor. Rows land in a JSONL file with the eventual game outcome
    backfilled, which makes the retrospective question answerable offline:
    does the value head's favourite disagree with the policy's, and does
    that disagreement track who actually won?

``analyze``
    Summarise a snapshots JSONL: override rate, win rate split by whether
    the value head's best move was the one played, and mean regret
    (value-best minus value-chosen).

``h2h``
    Play a re-ranking agent (the checkpoint's own search proposes the top-k
    candidates; the value head picks among them) against the plain search
    agent, paired deals with seats swapped, greedy.

Value convention: the simple-mode value head is trained from the priority
player's perspective on that player's observation, and the simple encoder
only reads public per-player features, so encoding a full state is the same
vector as encoding its observation. For a successor state s' the mover's
value is v(s') if s'.priority_player == mover else -v(s'), and exactly
+1/-1/0 for terminal successors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_simple import (  # noqa: E402, F401
    MAX_STEPS,
    load_checkpoint,
)
from train_simple import play_game as _play_game  # noqa: E402
from train_simple import seed_everything  # noqa: E402

from manamind.core.agent import Agent, MCTSAgent  # noqa: E402
from manamind.core.game_state import GameState  # noqa: E402
from manamind.core.observation import observe  # noqa: E402
from manamind.rules.simple import create_simple_game_start  # noqa: E402


def action_key(action: Any) -> str:
    """Compact identity for a candidate action within one snapshot."""
    kind = getattr(getattr(action, "action_type", None), "value", None)
    if kind is None:
        return str(action)
    card = getattr(action, "card", None)
    name = getattr(card, "name", None) if card is not None else None
    parts = [kind]
    if name:
        parts.append(str(name))
    return ":".join(parts)


def mover_value(net: Any, successor: GameState, mover: int) -> float:
    """Value of ``successor`` for the player who just moved, in [-1, 1]."""
    if successor.is_game_over():
        winner = successor.winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == mover else -1.0
    with __import__("torch").no_grad():
        value = float(net.evaluate_position(successor).flatten()[0])
    to_move = successor.priority_player
    return value if to_move == mover else -value


def score_candidates(
    net: Any, state: GameState, mover: int, candidates: List[Tuple[Any, int]]
) -> List[Dict[str, Any]]:
    """Engine-apply each candidate on a copy and value-score the successor."""
    scored: List[Dict[str, Any]] = []
    for cand_action, visits in candidates:
        successor = state.copy()
        successor = cand_action.execute(successor)
        scored.append(
            {
                "action": action_key(cand_action),
                "visits": visits,
                "value": round(mover_value(net, successor, mover), 6),
            }
        )
    return scored


def search_candidates(agent: MCTSAgent, top_k: int) -> List[Tuple[Any, int]]:
    """Top-k actions from the agent's most recent root search, by visits."""
    root = agent._last_root
    if root is None or not root.children:
        return []
    pairs = sorted(
        root.children,
        key=lambda action_child: action_child[1].visits,
        reverse=True,
    )
    return [(action, child.visits) for action, child in pairs[:top_k]]


def make_agent(
    net: Any, player_id: int, args: argparse.Namespace
) -> MCTSAgent:
    return MCTSAgent(
        player_id=player_id,
        policy_network=net,
        value_network=net,
        simulations=args.sims,
        simulation_time=args.sim_time,
        search=args.search,
        temperature=0.0,
    )


def collect(args: argparse.Namespace) -> None:
    net = load_checkpoint(Path(args.checkpoint), weights=args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    started = time.time()
    decisions = 0
    for game in range(args.games):
        deal = args.seed + game
        state = create_simple_game_start(deal)
        agents = {pid: make_agent(net, pid, args) for pid in (0, 1)}
        game_rows: List[Dict[str, Any]] = []
        for step in range(MAX_STEPS):
            if state.is_game_over():
                break
            actor = state.priority_player
            agent = agents[actor]
            seen = observe(state, actor)
            action = agent.select_action(seen)
            if (
                agent._last_forced is None
                and args.turn_min <= state.turn_number <= args.turn_max
            ):
                candidates = search_candidates(agent, args.top_k)
                if len(candidates) >= 2:
                    scored = score_candidates(net, state, actor, candidates)
                    best = max(scored, key=lambda row: row["value"])
                    chosen_key = action_key(action)
                    game_rows.append(
                        {
                            "game": game,
                            "seed": deal,
                            "turn": state.turn_number,
                            "step": step,
                            "seat": actor,
                            "chosen": chosen_key,
                            "candidates": scored,
                            "override": best["action"] != chosen_key,
                        }
                    )
                    decisions += 1
            state = action.execute(state)
        winner = state.winner()
        truncated = not state.is_game_over()
        for row in game_rows:
            if truncated:
                row["outcome"] = None
            elif winner is None:
                row["outcome"] = 0
            else:
                row["outcome"] = 1 if winner == row["seat"] else -1
            rows.append(row)
    elapsed = time.time() - started
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "games": args.games,
        "sims": args.sims,
        "top_k": args.top_k,
        "checkpoint": str(args.checkpoint),
        "decisions": decisions,
        "elapsed_seconds": round(elapsed, 1),
        "out": str(out_path),
    }
    print(json.dumps(summary))


def analyze(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for line in Path(args.snapshots).read_text().splitlines()
        if line.strip()
    ]
    decided = [row for row in rows if row.get("outcome") is not None]
    overrides = [row for row in decided if row["override"]]
    non_overrides = [row for row in decided if not row["override"]]

    def win_rate(sample: List[Dict[str, Any]]) -> Optional[float]:
        if not sample:
            return None
        return sum(1 for row in sample if row["outcome"] == 1) / len(sample)

    def mean_regret(sample: List[Dict[str, Any]]) -> Optional[float]:
        if not sample:
            return None
        regrets = []
        for row in sample:
            values = {
                candidate["action"]: candidate["value"]
                for candidate in row["candidates"]
            }
            chosen = values.get(row["chosen"])
            if chosen is None:
                # The played move can fall outside the top-k candidates
                # (a gumbel search picks by g + logits + sigma(q), not
                # visits), in which case there is no regret to measure.
                continue
            best = max(values.values())
            regrets.append(best - chosen)
        if not regrets:
            return None
        return sum(regrets) / len(regrets)

    def by_seat(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for seat in (0, 1):
            part = [row for row in sample if row["seat"] == seat]
            result[f"seat{seat}"] = {
                "n": len(part),
                "override_rate": (
                    round(sum(1 for r in part if r["override"]) / len(part), 4)
                    if part
                    else None
                ),
                "win_rate": win_rate(part),
            }
        return result

    result = {
        "snapshots": str(args.snapshots),
        "decisions": len(rows),
        "decided": len(decided),
        "override_rate": (
            round(sum(1 for r in rows if r["override"]) / len(rows), 4)
            if rows
            else None
        ),
        "win_rate_when_value_agrees": win_rate(non_overrides),
        "win_rate_when_value_disagrees": win_rate(overrides),
        "mean_regret": mean_regret(rows),
        "by_seat": by_seat(decided),
    }
    out_path = Path(args.out) if args.out else None
    text = json.dumps(result, indent=2)
    print(text)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


class RerankAgent(Agent):
    """Search for candidates, then let the value head pick among them."""

    def __init__(
        self,
        player_id: int,
        net: Any,
        args: argparse.Namespace,
    ) -> None:
        super().__init__(player_id)
        self.inner = make_agent(net, player_id, args)
        self.net = net
        self.top_k = args.top_k
        self.last_was_forced = False

    def update_from_game(
        self, game_history: List[Tuple[GameState, Any, float]]
    ) -> None:
        """Nothing to learn at eval time."""

    def select_action(self, game_state: GameState) -> Any:
        action = self.inner.select_action(game_state)
        if self.inner._last_forced is not None:
            self.last_was_forced = True
            return action
        self.last_was_forced = False
        root = self.inner._last_root
        if root is None:
            return action
        candidates = search_candidates(self.inner, self.top_k)
        if len(candidates) < 2:
            return action
        scored = score_candidates(
            self.net, root.game_state, self.player_id, candidates
        )
        best = max(scored, key=lambda row: row["value"])
        for cand_action, _ in candidates:
            if action_key(cand_action) == best["action"]:
                return cand_action
        return action


def match(
    a_factory: Any,
    b_factory: Any,
    games: int,
    seed: int,
) -> Dict[str, Any]:
    """Paired-deal match: same deal both seats, greedy (see head_to_head)."""
    a_wins = b_wins = draws = 0
    turns = 0
    for game in range(games):
        deal = seed + game // 2
        a_seat = game % 2
        agents = {a_seat: a_factory(), 1 - a_seat: b_factory()}
        winner, game_turns, _ = _play_game(agents, seed=deal)
        turns += game_turns
        if winner is None:
            draws += 1
        elif winner == a_seat:
            a_wins += 1
        else:
            b_wins += 1
    decided = games - draws
    rate = a_wins / decided if decided else 0.5
    ci = 1.96 * (rate * (1 - rate) / decided) ** 0.5 if decided else 0.0
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": round(rate, 4),
        "ci95": round(ci, 4),
        "mean_turns": round(turns / games, 2) if games else None,
    }


def h2h_mode(args: argparse.Namespace) -> None:
    net = load_checkpoint(Path(args.checkpoint), weights=args.weights)
    started = time.time()
    result = match(
        lambda: RerankAgent(0, net, args),
        lambda: MCTSAgent(
            player_id=1,
            policy_network=net,
            value_network=net,
            simulations=args.sims,
            simulation_time=args.sim_time,
            search=args.search,
            temperature=0.0,
        ),
        games=args.games,
        seed=args.seed,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "sims": args.sims,
            "top_k": args.top_k,
            "games": args.games,
            "elapsed_seconds": round(time.time() - started, 1),
        }
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    collect_p = sub.add_parser("collect", help="snapshot decisions + values")
    collect_p.add_argument("--checkpoint", required=True)
    collect_p.add_argument("--weights", default="network")
    collect_p.add_argument("--games", type=int, default=200)
    collect_p.add_argument("--sims", type=int, default=32)
    collect_p.add_argument("--sim-time", type=float, default=30.0)
    collect_p.add_argument(
        "--search", default="puct", choices=("puct", "gumbel")
    )
    collect_p.add_argument("--top-k", type=int, default=8)
    collect_p.add_argument("--turn-min", type=int, default=2)
    collect_p.add_argument("--turn-max", type=int, default=8)
    collect_p.add_argument("--seed", type=int, default=20261110)
    collect_p.add_argument("--out", required=True)
    collect_p.set_defaults(func=collect)

    analyze_p = sub.add_parser("analyze", help="summarise a snapshots JSONL")
    analyze_p.add_argument("--snapshots", required=True)
    analyze_p.add_argument("--out")
    analyze_p.set_defaults(func=analyze)

    h2h_p = sub.add_parser("h2h", help="rerank agent vs plain search agent")
    h2h_p.add_argument("--checkpoint", required=True)
    h2h_p.add_argument("--weights", default="network")
    h2h_p.add_argument("--games", type=int, default=400)
    h2h_p.add_argument("--sims", type=int, default=32)
    h2h_p.add_argument("--sim-time", type=float, default=30.0)
    h2h_p.add_argument("--search", default="puct", choices=("puct", "gumbel"))
    h2h_p.add_argument("--top-k", type=int, default=8)
    h2h_p.add_argument("--seed", type=int, default=20261110)
    h2h_p.add_argument("--out")
    h2h_p.set_defaults(func=h2h_mode)

    args = parser.parse_args()
    if getattr(args, "seed", None) is not None:
        seed_everything(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
