"""Write simple-mode replay fixtures for host ports (planar-nexus#2378).

A host that reimplements the simple ruleset (for example the TypeScript port in
planar-nexus) needs proof that it plays the same game the network was trained
on. This script plays seeded games with uniformly random moves and records,
for every decision: the legal moves in engine order, the observation vector,
and the move that was played. A host replays the fixture from the recorded
opening deal and must reproduce every legal-move list and observation.

Optionally (``--model-dir``) it also records the exported network's logits
and ``legal_priors`` for the first decisions of each game, so the host can
check its prior computation without running a model in its unit tests.

With ``--search-sims`` (e.g. ``10,40``) it also runs Gumbel search
(``MCTSAgent(search="gumbel")``, no root noise, the setting behind the
Medium, Hard and Expert tiers) on every ``--search-every``-th real decision
of each game, up to ``--search-decisions`` per game. Search sees
``observe(state, seat)``, exactly as in the tier games, so the host must
reproduce the hidden-card placeholders too.
Each record holds the chosen move and the completed-Q search policy, both
indexed in legal-move order.

    PYTHONPATH=src python scripts/export_host_fixtures.py out.json --games 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manamind.core.action import Action, ActionType  # noqa: E402
from manamind.core.agent import MCTSAgent  # noqa: E402
from manamind.core.game_state import GameState, Player  # noqa: E402
from manamind.core.observation import observe  # noqa: E402
from manamind.rules.simple import (  # noqa: E402
    SimpleRules,
    SimpleStateEncoder,
    create_simple_game_start,
)

FIXTURE_VERSION = 1
MAX_STEPS = 400


def _names(cards: list) -> List[str]:
    return [c.name for c in cards]


def _deal(player: Player) -> Dict[str, List[str]]:
    return {
        "hand": _names(player.hand.cards),
        "library": _names(player.library.cards),
    }


def _move(action: Action) -> Dict[str, Any]:
    move: Dict[str, Any] = {"type": action.action_type.value}
    if action.action_type in (ActionType.PLAY_LAND, ActionType.CAST_SPELL):
        move["card"] = action.card.name if action.card else None
    elif action.action_type == ActionType.DECLARE_ATTACKERS:
        move["attackers"] = list(action.attackers)
    elif action.action_type == ActionType.DECLARE_BLOCKERS:
        move["blockers"] = {
            str(k): list(v) for k, v in sorted(action.blockers.items())
        }
    return move


def _round(values: List[float]) -> List[float]:
    return [round(float(v), 6) for v in values]


def _search_records(
    state: GameState,
    legal: List[Action],
    net: Any,
    sims_list: List[int],
) -> List[Dict[str, Any]]:
    """Gumbel decisions for the priority player, on their observation."""
    seat = state.priority_player
    seen = observe(state, seat)
    seen_legal = SimpleRules.legal_actions(seen)
    if [_move(a) for a in seen_legal] != [_move(a) for a in legal]:
        raise RuntimeError("legal moves differ on the observation")
    records = []
    for sims in sims_list:
        agent = MCTSAgent(
            player_id=seat,
            policy_network=net,
            value_network=net,
            simulations=sims,
            simulation_time=1e9,
            search="gumbel",
        )
        action = agent.select_action(seen)
        policy = [0.0] * len(legal)
        positions = {_key(a): i for i, a in enumerate(seen_legal)}
        for child_action, p in agent._last_gumbel_policy or []:
            policy[positions[_key(child_action)]] = p
        records.append(
            {
                "sims": sims,
                "chosen": positions[_key(action)],
                "policy": _round(policy),
            }
        )
    return records


def _key(action: Action) -> str:
    return json.dumps(_move(action), sort_keys=True)


def play_fixture_game(
    seed: int,
    model: Optional[Any] = None,
    prior_steps: int = 0,
    net: Optional[Any] = None,
    sims_list: Optional[List[int]] = None,
    search_decisions: int = 0,
    search_every: int = 1,
) -> Dict[str, Any]:
    state: GameState = create_simple_game_start(seed=seed)
    encoder = SimpleStateEncoder()
    rng = random.Random(10_000 + seed)
    game: Dict[str, Any] = {
        "seed": seed,
        "deal": [_deal(state.players[0]), _deal(state.players[1])],
        "steps": [],
    }
    searched = 0
    real_index = 0
    for step in range(MAX_STEPS):
        if state.is_game_over():
            break
        legal = SimpleRules.legal_actions(state)
        obs = encoder.features(state).tolist()
        chosen = rng.randrange(len(legal))
        record: Dict[str, Any] = {
            "priority": state.priority_player,
            "phase": state.phase,
            "turn": state.turn_number,
            "legal": [_move(a) for a in legal],
            "obs": _round(obs),
            "chosen": chosen,
        }
        if model is not None and step < prior_steps:
            logits, priors = model(state, legal)
            record["logits"] = _round(logits)
            record["priors"] = _round(priors)
        real = len(legal) > 1
        if (
            net is not None
            and real
            and searched < search_decisions
            and real_index % search_every == 0
        ):
            record["search"] = _search_records(
                state, legal, net, sims_list or []
            )
            searched += 1
        if real:
            real_index += 1
        game["steps"].append(record)
        state = SimpleRules.apply(state, legal[chosen])
    game["winner"] = state.winner()
    game["final_life"] = [state.players[0].life, state.players[1].life]
    return game


def _onnx_model(model_dir: Path):
    import numpy as np
    import onnxruntime as ort
    from onnx_host import legal_priors, load_schema, observation

    schema = load_schema(model_dir)
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )

    def run(state: GameState, legal: List[Action]):
        obs = observation(state, schema)[None, :]
        logits, _ = session.run(None, {"observation": obs.astype(np.float32)})
        priors = legal_priors(logits[0], legal, schema["actions"])
        return logits[0].tolist(), priors.tolist()

    def net(state: GameState):
        import torch

        obs = observation(state, schema)[None, :]
        logits, value = session.run(
            None, {"observation": obs.astype(np.float32)}
        )
        return torch.from_numpy(logits), torch.from_numpy(value)

    return run, schema, net


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prior-steps", type=int, default=40)
    parser.add_argument(
        "--search-sims",
        default="",
        help="comma-separated Gumbel simulation counts, e.g. 10,40",
    )
    parser.add_argument("--search-decisions", type=int, default=12)
    parser.add_argument(
        "--search-every",
        type=int,
        default=3,
        help="search every Nth real decision, so mid-game spots are covered",
    )
    args = parser.parse_args(argv)

    model = None
    net = None
    schema_version = None
    sims_list = [int(x) for x in args.search_sims.split(",") if x.strip()]
    if sims_list and args.model_dir is None:
        parser.error("--search-sims needs --model-dir")
    if args.model_dir is not None:
        model, schema, onnx_net = _onnx_model(args.model_dir)
        schema_version = schema["schema_version"]
        if sims_list:
            net = onnx_net

    games = [
        play_fixture_game(
            seed,
            model,
            args.prior_steps if model else 0,
            net,
            sims_list,
            args.search_decisions if net else 0,
            max(1, args.search_every),
        )
        for seed in range(args.first_seed, args.first_seed + args.games)
    ]
    fixture = {
        "fixture_version": FIXTURE_VERSION,
        "ruleset": "simple",
        "schema_version": schema_version,
        "model": str(args.model_dir) if args.model_dir else None,
        "games": games,
    }
    args.out.write_text(json.dumps(fixture, separators=(",", ":")))
    steps = sum(len(g["steps"]) for g in games)
    print(f"wrote {args.out}: {len(games)} games, {steps} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
