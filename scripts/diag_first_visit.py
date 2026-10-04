"""Diagnose why first-visit child values hurt gumbel search (#58).

Plays games between two copies of a checkpoint (gumbel search, current
``main`` behaviour) and, at every real decision, scores each legal action
two ways with the value head:

* ``direct``: the position right after the action (what PR #59 used)
* ``then_pass``: the action followed by the last legal action at the child,
  which is always PASS_PRIORITY (what ``main`` uses on a first visit)

Each record also notes who holds priority at the child and the eventual
result for the mover. The summary splits by child priority holder:

* A sign/perspective bug shows as negative correlation with the result
  (and with the root value) in one group only.
* A calibration problem shows as positive but weak correlation, with a
  bias against the root value.

    python scripts/diag_first_visit.py CHECKPOINT [--games 40]
        [--simulations 40] [--out diag.json]

``CHECKPOINT`` may be ``random`` for a smoke test with an untrained net.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_simple import (  # noqa: E402
    MAX_STEPS,
    MCTSAgent,
    known_deck_lists,
    load_checkpoint,
    seed_everything,
)

from manamind.core.action import ActionSpace  # noqa: E402
from manamind.core.observation import observe  # noqa: E402
from manamind.rules.simple import (  # noqa: E402
    build_simple_network,
    create_simple_game_start,
)

GROUPS = ("mover", "opponent", "terminal")


def corr(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def score_actions(state, mover: int, evaluator, space) -> Dict:
    """Value every legal action at ``state`` both ways."""
    root_value = evaluator._evaluate_position(state)
    rows = []
    for action in space.get_legal_actions(state):
        child = action.execute(state)
        if child.is_game_over():
            group = "terminal"
            direct = then_pass = evaluator._evaluate_position(child)
        else:
            group = "mover" if child.priority_player == mover else "opponent"
            direct = evaluator._evaluate_position(child)
            follow = space.get_legal_actions(child)[-1]
            then_pass = evaluator._evaluate_position(follow.execute(child))
        rows.append(
            {
                "action": action.action_type.name,
                "group": group,
                "direct": direct,
                "then_pass": then_pass,
            }
        )
    return {"root_value": root_value, "actions": rows}


def play_and_record(net, seed: int, simulations: int) -> List[Dict]:
    space = ActionSpace()
    agents = {
        pid: MCTSAgent(
            player_id=pid,
            policy_network=net,
            value_network=net,
            simulations=simulations,
            simulation_time=30.0,
            deck_lists=known_deck_lists(),
            search="gumbel",
        )
        for pid in (0, 1)
    }
    evaluators = {
        pid: MCTSAgent(player_id=pid, policy_network=net, value_network=net)
        for pid in (0, 1)
    }
    state = create_simple_game_start(seed)
    decisions: List[Dict] = []
    for _ in range(MAX_STEPS):
        if state.is_game_over():
            break
        mover = state.priority_player
        if len(space.get_legal_actions(state)) > 1:
            record = score_actions(state, mover, evaluators[mover], space)
            record["mover"] = mover
            decisions.append(record)
        action = agents[mover].select_action(observe(state, mover))
        if decisions and decisions[-1]["mover"] == mover:
            decisions[-1].setdefault("chosen", action.action_type.name)
        state = action.execute(state)
    winner = state.winner()
    for record in decisions:
        if winner is None:
            record["z"] = 0.0
        else:
            record["z"] = 1.0 if winner == record["mover"] else -1.0
    return decisions


def summarize(decisions: List[Dict]) -> Dict:
    summary: Dict = {
        "decisions": len(decisions),
        "root_value_vs_result": corr(
            [d["root_value"] for d in decisions], [d["z"] for d in decisions]
        ),
        "groups": {},
    }
    for group in GROUPS:
        rows = [
            (d, a)
            for d in decisions
            for a in d["actions"]
            if a["group"] == group
        ]
        entry: Dict = {"n": len(rows)}
        for method in ("direct", "then_pass"):
            values = [a[method] for _, a in rows]
            entry[method] = {
                "mean_minus_root": (
                    sum(a[method] - d["root_value"] for d, a in rows)
                    / len(rows)
                    if rows
                    else None
                ),
                "corr_with_result": corr(values, [d["z"] for d, _ in rows]),
                "corr_with_root": corr(
                    values, [d["root_value"] for d, _ in rows]
                ),
            }
        summary["groups"][group] = entry
    for method in ("direct", "then_pass"):
        agree = [
            max(d["actions"], key=lambda a: a[method])["action"] == d["chosen"]
            for d in decisions
            if "chosen" in d
        ]
        summary[f"{method}_argmax_matches_search"] = (
            sum(agree) / len(agree) if agree else None
        )
    return summary


def fmt(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.checkpoint == "random":
        net = build_simple_network(
            action_space_size=len(ActionSpace().action_to_id)
        )
        net.eval()
    else:
        net = load_checkpoint(Path(args.checkpoint))

    decisions: List[Dict] = []
    for game in range(args.games):
        decisions.extend(
            play_and_record(net, args.seed * 1000 + game, args.simulations)
        )
        print(f"game {game + 1}/{args.games}: {len(decisions)} decisions")
    summary = summarize(decisions)

    print(f"\ndecisions: {summary['decisions']}")
    print(f"root value vs result: {fmt(summary['root_value_vs_result'])}")
    for group, entry in summary["groups"].items():
        print(f"\nchild priority = {group} (n={entry['n']})")
        for method in ("direct", "then_pass"):
            m = entry[method]
            print(
                f"  {method:9s} mean-root {fmt(m['mean_minus_root'])}  "
                f"corr(result) {fmt(m['corr_with_result'])}  "
                f"corr(root) {fmt(m['corr_with_root'])}"
            )
    for method in ("direct", "then_pass"):
        key = f"{method}_argmax_matches_search"
        print(f"{method} argmax matches search choice: {fmt(summary[key])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "decisions": decisions}, indent=1)
        )


if __name__ == "__main__":
    main()
