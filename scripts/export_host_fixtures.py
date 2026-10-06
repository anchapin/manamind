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
from manamind.core.game_state import GameState, Player  # noqa: E402
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


def play_fixture_game(
    seed: int, model: Optional[Any] = None, prior_steps: int = 0
) -> Dict[str, Any]:
    state: GameState = create_simple_game_start(seed=seed)
    encoder = SimpleStateEncoder()
    rng = random.Random(10_000 + seed)
    game: Dict[str, Any] = {
        "seed": seed,
        "deal": [_deal(state.players[0]), _deal(state.players[1])],
        "steps": [],
    }
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

    return run, schema


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prior-steps", type=int, default=40)
    args = parser.parse_args(argv)

    model = None
    schema_version = None
    if args.model_dir is not None:
        model, schema = _onnx_model(args.model_dir)
        schema_version = schema["schema_version"]

    games = [
        play_fixture_game(seed, model, args.prior_steps if model else 0)
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
