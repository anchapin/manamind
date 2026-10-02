"""Reference host: play simple-mode games through an exported model (#46).

Shows that ``model.onnx`` plus ``model.schema.json`` (from
``export_onnx.py``) is all a host engine needs. Observations are built
here with numpy from the schema's field list, and inference runs in
onnxruntime; the network and encoder are never touched. The simple
rules engine still comes from manamind, standing in for a host's own
rules (planar-nexus would use ``src/lib/game-state/``).

The agent is policy-only, with the difficulty knobs from #46:
``temperature`` (0 = always the top move) and ``blunder_rate`` (chance
of a uniformly random legal move).

    python scripts/onnx_host.py MODEL_DIR [--games N] [--temperature T]
        [--blunder-rate P] [--seed S]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manamind.core.action import Action, ActionSpace  # noqa: E402
from manamind.core.agent import RandomAgent  # noqa: E402
from manamind.core.game_state import GameState, Player  # noqa: E402

SUPPORTED_SCHEMAS = {"simple-v1"}


def _creatures(p: Player) -> list:
    return [c for c in p.battlefield.cards if c.is_creature()]


def _lands(p: Player) -> list:
    return [c for c in p.battlefield.cards if c.is_land()]


_PLAYER: Dict[str, Callable[[Player], float]] = {
    "life": lambda p: p.life / 20.0,
    "hand_size": lambda p: p.hand.size() / 7.0,
    "library_size": lambda p: p.library.size() / 40.0,
    "graveyard_size": lambda p: p.graveyard.size() / 10.0,
    "lands": lambda p: len(_lands(p)) / 10.0,
    "untapped_lands": lambda p: sum(1 for c in _lands(p) if not c.tapped)
    / 10.0,
    "creatures": lambda p: len(_creatures(p)) / 10.0,
    "total_power": lambda p: sum(c.current_power() or 0 for c in _creatures(p))
    / 20.0,
    "total_toughness": lambda p: sum(
        c.current_toughness() or 0 for c in _creatures(p)
    )
    / 20.0,
    "untapped_creatures": lambda p: sum(
        1 for c in _creatures(p) if not c.tapped
    )
    / 10.0,
}


def load_schema(model_dir: Path) -> dict:
    schema = json.loads((model_dir / "model.schema.json").read_text())
    version = schema.get("schema_version")
    if version not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"unsupported schema_version {version!r}; this host reads "
            f"{sorted(SUPPORTED_SCHEMAS)}"
        )
    return schema


def observation(state: GameState, schema: dict) -> np.ndarray:
    """The observation vector, built from the schema's field names."""
    mover = state.priority_player
    players = {
        "self": state.players[mover],
        "opponent": state.players[1 - mover],
    }
    values: List[float] = []
    for field in schema["observation"]:
        name = field["name"]
        who, _, attr = name.partition(".")
        if who in players:
            values.append(_PLAYER[attr](players[who]))
        elif who == "phase":
            values.append(1.0 if state.phase == attr else 0.0)
        elif name == "self_is_active":
            values.append(1.0 if state.active_player == mover else 0.0)
        elif name == "turn":
            values.append(min(state.turn_number, 60) / 60.0)
        else:
            raise ValueError(f"unknown observation field {name!r}")
    return np.asarray(values, dtype=np.float32)


def legal_priors(
    logits: np.ndarray, legal: List[Action], actions: List[str]
) -> np.ndarray:
    """Softmax over legal moves; moves sharing an action id split its mass.

    Matches how ``MCTSAgent`` turns logits into priors.
    """
    index = {name: i for i, name in enumerate(actions)}
    ids = [index.get(a.action_type.value) for a in legal]
    known = [i for i in ids if i is not None]
    if not known:
        return np.full(len(legal), 1.0 / len(legal))
    z = logits[known] - logits[known].max()
    mass = dict(zip(known, np.exp(z) / np.exp(z).sum()))
    shares: Dict[int, int] = {}
    for i in known:
        shares[i] = shares.get(i, 0) + 1
    priors = np.array([0.0 if i is None else mass[i] / shares[i] for i in ids])
    total = priors.sum()
    return priors / total if total > 0 else np.full(len(legal), 1 / len(legal))


class OnnxPolicyAgent:
    """Picks moves from the exported policy head, no search."""

    def __init__(
        self,
        model_dir: Path,
        player_id: int = 0,
        temperature: float = 0.0,
        blunder_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        import onnxruntime as ort

        self.player_id = player_id
        self.schema = load_schema(model_dir)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.action_space = ActionSpace()
        self.temperature = temperature
        self.blunder_rate = blunder_rate
        self.rng = random.Random(seed)
        self.decisions = 0
        self.seconds = 0.0
        self.last_value: Optional[float] = None

    def evaluate(self, state: GameState) -> tuple:
        obs = observation(state, self.schema)[None, :]
        logits, value = self.session.run(None, {"observation": obs})
        return logits[0], float(value[0, 0])

    def select_action(self, state: GameState) -> Action:
        legal = self.action_space.get_legal_actions(state)
        if not legal:
            raise ValueError("no legal actions")
        if len(legal) == 1:
            return legal[0]
        start = time.perf_counter()
        logits, self.last_value = self.evaluate(state)
        priors = legal_priors(logits, legal, self.schema["actions"])
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        if self.blunder_rate > 0 and self.rng.random() < self.blunder_rate:
            return self.rng.choice(legal)
        if self.temperature <= 0:
            return legal[int(np.argmax(priors))]
        weights = priors ** (1.0 / self.temperature)
        return self.rng.choices(legal, weights=list(weights))[0]


def main() -> None:
    from train_simple import play_game

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--blunder-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wins = draws = 0
    agent = None
    for g in range(args.games):
        seat = g % 2
        agent = OnnxPolicyAgent(
            args.model_dir,
            player_id=seat,
            temperature=args.temperature,
            blunder_rate=args.blunder_rate,
            seed=args.seed + g,
        )
        agents = {seat: agent, 1 - seat: RandomAgent(1 - seat, seed=g)}
        winner, _, _ = play_game(agents, seed=args.seed * 1000 + g)
        wins += winner == seat
        draws += winner is None
        if agent.decisions:
            ms = 1000 * agent.seconds / agent.decisions
            print(f"game {g}: winner {winner}, {ms:.2f} ms/decision")
    print(
        f"vs random: {wins}/{args.games} wins, {draws} draws "
        f"(score {(wins + 0.5 * draws) / args.games:.3f})"
    )


if __name__ == "__main__":
    main()
