"""Train on the simple subset and report whether the agent improves.

This is the experiment behind issue #20. It answers one question: does the
loop learn anything at all? Run it, read the curve, and believe nothing else
about manamind until this says yes.

    python scripts/train_simple.py --iterations 6 --games 6 --seed 0

Every source of randomness is seeded, so two runs with the same seed give
the same curve.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from manamind.core.agent import MCTSAgent, RandomAgent
from manamind.core.game_state import GameState
from manamind.core.observation import observe
from manamind.models.policy_value_network import (
    PolicyValueLoss,
    PolicyValueNetwork,
)
from manamind.rules.simple import (
    build_simple_deck,
    build_simple_network,
    create_simple_game_start,
)

# Self-play only; evaluation searches without noise.
ROOT_DIRICHLET_ALPHA = 0.3
SELF_PLAY_TEMPERATURE = 1.0


def known_deck_lists() -> Dict[int, List[object]]:
    """Both seats play the fixed simple-mode list, so both lists are known."""
    return {0: build_simple_deck(), 1: build_simple_deck()}


MAX_STEPS = 400


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Example:
    """One training position: what the search concluded, and who won."""

    state: GameState
    policy: np.ndarray
    value: float


@dataclass
class IterationResult:
    iteration: int
    win_rate: float
    mean_loss: float
    examples: int
    mean_turns: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "win_rate": self.win_rate,
            "mean_loss": self.mean_loss,
            "examples": self.examples,
            "mean_turns": self.mean_turns,
        }


def play_game(
    agents: Dict[int, object], seed: int, record: bool = False
) -> Tuple[Optional[int], int, List[Example]]:
    """Play one simple-mode game. Returns winner, turns, examples."""
    state = create_simple_game_start(seed)
    history: List[Tuple[GameState, np.ndarray, int]] = []

    for _ in range(MAX_STEPS):
        if state.is_game_over():
            break
        actor = agents[state.priority_player]
        # Agents see only their own observation; the full state stays here.
        seen = observe(state, state.priority_player)
        action = actor.select_action(seen)
        if record and isinstance(actor, MCTSAgent):
            history.append(
                (
                    seen,
                    actor.last_search_policy(
                        actor.policy_network.action_space_size
                    ),
                    state.priority_player,
                )
            )
        state = action.execute(state)

    winner = state.winner()
    examples: List[Example] = []
    if record:
        for snapshot, policy, mover in history:
            value = (
                0.0 if winner is None else (1.0 if winner == mover else -1.0)
            )
            examples.append(Example(snapshot, policy, value))
    return winner, state.turn_number, examples


def evaluate(
    network: PolicyValueNetwork, games: int, simulations: int, seed: int
) -> float:
    """Win rate against RandomAgent, seats alternating.

    Alternating seats matters: whoever moves first in this subset wins more
    often, so a fixed seat would read as skill.
    """
    score = 0.0
    for game in range(games):
        seat = game % 2
        agents: Dict[int, object] = {
            seat: MCTSAgent(
                player_id=seat,
                policy_network=network,
                value_network=network,
                simulations=simulations,
                simulation_time=30.0,
                deck_lists=known_deck_lists(),
            ),
            1 - seat: RandomAgent(1 - seat, seed=seed + game),
        }
        winner, _, _ = play_game(agents, seed=seed + game)
        if winner == seat:
            score += 1.0
        elif winner is None:
            score += 0.5
    return score / games


def train_on_buffer(
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: PolicyValueLoss,
    buffer: List[Example],
    batch_size: int = 32,
    epochs: int = 8,
) -> List[float]:
    if len(buffer) < batch_size:
        return []

    network.train()
    losses: List[float] = []
    width = network.action_space_size

    for _ in range(epochs):
        batch = random.sample(buffer, batch_size)
        states = torch.stack([network.state_encoder(ex.state) for ex in batch])

        targets = np.zeros((batch_size, width), dtype=np.float32)
        for row, ex in enumerate(batch):
            policy = np.asarray(ex.policy, dtype=np.float32)[:width]
            total = float(policy.sum())
            if total <= 0:
                targets[row, :] = 1.0 / width
            else:
                targets[row, : policy.shape[0]] = policy / total

        policy_logits, value_pred = network(states)
        total_loss, _ = loss_fn(
            policy_logits,
            value_pred,
            torch.from_numpy(targets),
            torch.tensor([[ex.value] for ex in batch], dtype=torch.float32),
            network,
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total_loss.detach()))

    return losses


def train(
    iterations: int,
    games: int,
    eval_games: int,
    simulations: int,
    seed: int,
    out: Optional[Path] = None,
) -> List[IterationResult]:
    seed_everything(seed)

    probe = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    action_space_size = len(probe.action_space.action_to_id)

    network = build_simple_network(action_space_size=action_space_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = PolicyValueLoss(value_weight=1.0, l2_reg=1e-4)

    buffer: List[Example] = []
    results: List[IterationResult] = []

    baseline = evaluate(
        network, games=eval_games, simulations=simulations, seed=seed * 7919
    )
    print(
        f"iter  0  win_rate {baseline:.3f}  (untrained baseline)", flush=True
    )

    for iteration in range(1, iterations + 1):
        turns = []
        for game in range(games):
            agents: Dict[int, object] = {
                pid: MCTSAgent(
                    player_id=pid,
                    policy_network=network,
                    value_network=network,
                    simulations=simulations,
                    simulation_time=30.0,
                    deck_lists=known_deck_lists(),
                    root_dirichlet_alpha=ROOT_DIRICHLET_ALPHA,
                    temperature=SELF_PLAY_TEMPERATURE,
                )
                for pid in (0, 1)
            }
            _, length, examples = play_game(
                agents,
                seed=seed * 1000 + iteration * 100 + game,
                record=True,
            )
            buffer.extend(examples)
            turns.append(length)

        buffer = buffer[-20000:]
        losses = train_on_buffer(network, optimizer, loss_fn, buffer)
        win_rate = evaluate(
            network,
            games=eval_games,
            simulations=simulations,
            seed=seed * 7919 + iteration,
        )

        result = IterationResult(
            iteration=iteration,
            win_rate=win_rate,
            mean_loss=float(np.mean(losses)) if losses else float("nan"),
            examples=len(buffer),
            mean_turns=float(np.mean(turns)),
        )
        results.append(result)
        print(
            f"iter {iteration:>2}  win_rate {win_rate:.3f}  "
            f"loss {result.mean_loss:.4f}  examples {len(buffer)}  "
            f"mean_turns {result.mean_turns:.1f}",
            flush=True,
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": seed,
            "iterations": iterations,
            "games_per_iteration": games,
            "eval_games": eval_games,
            "simulations": simulations,
            "baseline_win_rate": baseline,
            "results": [r.as_dict() for r in results],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games=args.games,
        eval_games=args.eval_games,
        simulations=args.simulations,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
