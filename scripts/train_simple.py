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
import copy
import json
import multiprocessing as mp
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
    # Score against the frozen untrained network with the same search.
    # None for runs from before this existed, or with --ref-eval-games 0.
    ref_win_rate: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "win_rate": self.win_rate,
            "ref_win_rate": self.ref_win_rate,
            "mean_loss": self.mean_loss,
            "examples": self.examples,
            "mean_turns": self.mean_turns,
        }


def value_target(z: float, q_root: Optional[float], value_mix: float) -> float:
    """Soft-Z value target: blend the game result with the search's view.

    ``z`` is the final result from the mover's side (+1, 0, -1) and
    ``q_root`` the root's mean value after search, from the same side.
    ``value_mix`` 0 trains on the result alone (plain AlphaZero); 1 trains
    on the search value alone. Without a search value, the result is used.
    """
    if q_root is None or value_mix == 0.0:
        return z
    return (1.0 - value_mix) * z + value_mix * q_root


def play_game(
    agents: Dict[int, object],
    seed: int,
    record: bool = False,
    value_mix: float = 0.0,
) -> Tuple[Optional[int], int, List[Example]]:
    """Play one simple-mode game. Returns winner, turns, examples."""
    state = create_simple_game_start(seed)
    history: List[Tuple[GameState, np.ndarray, int, Optional[float]]] = []

    for _ in range(MAX_STEPS):
        if state.is_game_over():
            break
        actor = agents[state.priority_player]
        # Agents see only their own observation; the full state stays here.
        seen = observe(state, state.priority_player)
        action = actor.select_action(seen)
        # Forced moves carry no decision, so they make no training example.
        if (
            record
            and isinstance(actor, MCTSAgent)
            and actor.last_was_forced is False
        ):
            history.append(
                (
                    seen,
                    actor.last_search_policy(
                        actor.policy_network.action_space_size
                    ),
                    state.priority_player,
                    actor.last_root_value(),
                )
            )
        state = action.execute(state)

    winner = state.winner()
    examples: List[Example] = []
    if record:
        for snapshot, policy, mover, q_root in history:
            z = 0.0 if winner is None else (1.0 if winner == mover else -1.0)
            examples.append(
                Example(snapshot, policy, value_target(z, q_root, value_mix))
            )
    return winner, state.turn_number, examples


@dataclass
class GameJob:
    """One game to play: who sits where, and with what search settings.

    ``kind`` is "selfplay" (network vs itself, noisy, recorded), "random"
    (network on ``seat`` vs RandomAgent) or "reference" (network on
    ``seat`` vs the same search driven by the frozen reference network).
    """

    kind: str
    seed: int
    seat: int = 0
    simulations: int = 10
    fpu_reduction: Optional[float] = None
    search: str = "puct"
    value_mix: float = 0.0


def _search_agent(
    job: GameJob, pid: int, net: PolicyValueNetwork, noisy: bool
) -> MCTSAgent:
    extra: Dict[str, object] = (
        {
            "root_dirichlet_alpha": ROOT_DIRICHLET_ALPHA,
            "temperature": SELF_PLAY_TEMPERATURE,
            "gumbel_noise": True,
        }
        if noisy
        else {}
    )
    return MCTSAgent(
        player_id=pid,
        policy_network=net,
        value_network=net,
        simulations=job.simulations,
        simulation_time=30.0,
        deck_lists=known_deck_lists(),
        fpu_reduction=job.fpu_reduction,
        search=job.search,
        **extra,
    )


def _play_job(
    job: GameJob, nets: Dict[str, PolicyValueNetwork], reseed: bool
) -> Tuple[Optional[int], int, List[Example]]:
    if reseed:
        # Parallel games each get their own RNG stream, so a run gives
        # the same games whatever the worker count.
        seed_everything(job.seed)
    net = nets["net"]
    seat = job.seat
    agents: Dict[int, object]
    if job.kind == "selfplay":
        agents = {pid: _search_agent(job, pid, net, True) for pid in (0, 1)}
    elif job.kind == "random":
        agents = {
            seat: _search_agent(job, seat, net, False),
            1 - seat: RandomAgent(1 - seat, seed=job.seed),
        }
    elif job.kind == "reference":
        agents = {
            seat: _search_agent(job, seat, net, False),
            1 - seat: _search_agent(job, 1 - seat, nets["ref"], False),
        }
    else:
        raise ValueError(f"unknown game kind {job.kind!r}")
    return play_game(
        agents,
        seed=job.seed,
        record=job.kind == "selfplay",
        value_mix=job.value_mix,
    )


_WORKER_NETS: Dict[str, PolicyValueNetwork] = {}


def _init_worker(
    action_space_size: int,
    state_dicts: Dict[str, Dict[str, torch.Tensor]],
    training: Dict[str, bool],
) -> None:
    # One core per worker: the speedup comes from games running side by
    # side, not from torch threads fighting over the same cores.
    torch.set_num_threads(1)
    for name, state_dict in state_dicts.items():
        net = build_simple_network(action_space_size=action_space_size)
        net.load_state_dict(state_dict)
        # Match the parent's train/eval mode: the network has dropout, and
        # search in the main process runs with whatever mode the network
        # is in. Forcing eval here made parallel games differ from inline.
        net.train(training[name])
        _WORKER_NETS[name] = net


def _worker_play(job: GameJob) -> Tuple[Optional[int], int, List[Example]]:
    return _play_job(job, _WORKER_NETS, reseed=True)


def run_games(
    jobs: List[GameJob],
    nets: Dict[str, PolicyValueNetwork],
    workers: int = 1,
) -> List[Tuple[Optional[int], int, List[Example]]]:
    """Play ``jobs`` in order, or across ``workers`` processes.

    Games are independent, so they parallelise cleanly; the two seats of
    one game cannot, because they move in turn. Every game reseeds from
    its own job seed, inline too, so any worker count plays exactly the
    same games. (Runs from before this change used one shared RNG stream
    and won't reproduce bit for bit.)
    """
    if workers <= 1 or len(jobs) <= 1:
        return [_play_job(job, nets, reseed=True) for job in jobs]
    state_dicts = {
        name: {k: v.detach().cpu() for k, v in net.state_dict().items()}
        for name, net in nets.items()
    }
    size = nets["net"].action_space_size
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        min(workers, len(jobs)),
        initializer=_init_worker,
        initargs=(
            size,
            state_dicts,
            {name: net.training for name, net in nets.items()},
        ),
    ) as pool:
        return pool.map(_worker_play, jobs, chunksize=1)


def _score(
    jobs: List[GameJob],
    results: List[Tuple[Optional[int], int, List[Example]]],
) -> float:
    score = 0.0
    for job, (winner, _, _) in zip(jobs, results):
        if winner == job.seat:
            score += 1.0
        elif winner is None:
            score += 0.5
    return score / len(jobs) if jobs else float("nan")


def evaluate(
    network: PolicyValueNetwork,
    games: int,
    simulations: int,
    seed: int,
    fpu_reduction: Optional[float] = None,
    search: str = "puct",
    workers: int = 1,
) -> float:
    """Win rate against RandomAgent, seats alternating.

    Alternating seats matters: whoever moves first in this subset wins more
    often, so a fixed seat would read as skill.
    """
    jobs = [
        GameJob("random", seed + g, g % 2, simulations, fpu_reduction, search)
        for g in range(games)
    ]
    return _score(jobs, run_games(jobs, {"net": network}, workers))


def evaluate_vs_reference(
    network: PolicyValueNetwork,
    reference: PolicyValueNetwork,
    games: int,
    simulations: int,
    seed: int,
    fpu_reduction: Optional[float] = None,
    search: str = "puct",
    workers: int = 1,
) -> float:
    """Score against the same search driven by a frozen reference network.

    Search alone beats RandomAgent most of the time before any training,
    so that metric saturates. Here both seats search with identical
    settings and only the networks differ, so a score above 0.5 is what
    training added. Seats alternate, greedy play, draws count half.
    """
    jobs = [
        GameJob(
            "reference", seed + g, g % 2, simulations, fpu_reduction, search
        )
        for g in range(games)
    ]
    nets = {"net": network, "ref": reference}
    return _score(jobs, run_games(jobs, nets, workers))


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


def save_checkpoint(
    path: Path,
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    seed: int,
    action_space_size: int,
    result: IterationResult,
    resume_state: Optional[dict] = None,
) -> None:
    """Write weights plus enough metadata to rebuild and re-evaluate.

    ``resume_state`` (replay buffer, results so far, baseline and RNG
    states) lets ``--resume`` continue a run a restart interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "seed": seed,
        "action_space_size": action_space_size,
        "result": result.as_dict(),
    }
    if resume_state is not None:
        payload["resume"] = resume_state
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def rng_state() -> dict:
    """Capture every RNG the training loop draws from."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def latest_resumable(checkpoint_dir: Optional[Path]) -> Optional[Path]:
    """Newest iter_NNN.pt in ``checkpoint_dir`` that carries resume state."""
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    for path in sorted(checkpoint_dir.glob("iter_*.pt"), reverse=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "resume" in payload:
            return path
    return None


def load_checkpoint(path: Path) -> PolicyValueNetwork:
    """Rebuild the simple-mode network from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    network = build_simple_network(
        action_space_size=payload["action_space_size"]
    )
    network.load_state_dict(payload["network"])
    network.eval()
    return network


def _write_results(
    out: Optional[Path],
    seed: int,
    iterations: int,
    games: int,
    eval_games: int,
    simulations: int,
    baseline: float,
    results: List[IterationResult],
    value_mix: float = 0.0,
) -> None:
    """Write the curve so far; called every iteration so a crash keeps it."""
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "iterations": iterations,
        "games_per_iteration": games,
        "eval_games": eval_games,
        "simulations": simulations,
        "value_mix": value_mix,
        "baseline_win_rate": baseline,
        "results": [r.as_dict() for r in results],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")


def train(
    iterations: int,
    games: int,
    eval_games: int,
    simulations: int,
    seed: int,
    out: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    fpu_reduction: Optional[float] = None,
    resume: bool = False,
    search: str = "puct",
    value_mix: float = 0.0,
    ref_eval_games: Optional[int] = None,
    workers: int = 1,
) -> List[IterationResult]:
    seed_everything(seed)

    probe = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    action_space_size = len(probe.action_space.action_to_id)

    network = build_simple_network(action_space_size=action_space_size)
    # Taken before any resume load, so a resumed run rebuilds the same
    # untrained weights from the seed.
    reference = copy.deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = PolicyValueLoss(value_weight=1.0, l2_reg=1e-4)

    buffer: List[Example] = []
    results: List[IterationResult] = []
    start = 1

    resume_from = latest_resumable(checkpoint_dir) if resume else None
    if resume_from is not None:
        payload = torch.load(
            resume_from, map_location="cpu", weights_only=False
        )
        if payload["seed"] != seed:
            raise ValueError(
                f"{resume_from} was trained with seed {payload['seed']}, "
                f"not {seed}"
            )
        network.load_state_dict(payload["network"])
        optimizer.load_state_dict(payload["optimizer"])
        state = payload["resume"]
        buffer = [Example(*item) for item in state["buffer"]]
        results = [IterationResult(**r) for r in state["results"]]
        baseline = state["baseline"]
        restore_rng_state(state["rng"])
        start = payload["iteration"] + 1
        print(
            f"resumed from {resume_from} (iteration {start - 1})", flush=True
        )
    else:
        baseline = evaluate(
            network,
            games=eval_games,
            simulations=simulations,
            seed=seed * 7919,
            fpu_reduction=fpu_reduction,
            search=search,
            workers=workers,
        )
        print(
            f"iter  0  win_rate {baseline:.3f}  (untrained baseline)",
            flush=True,
        )

    for iteration in range(start, iterations + 1):
        turns = []
        jobs = [
            GameJob(
                "selfplay",
                seed * 1000 + iteration * 100 + game,
                simulations=simulations,
                fpu_reduction=fpu_reduction,
                search=search,
                value_mix=value_mix,
            )
            for game in range(games)
        ]
        for _, length, examples in run_games(jobs, {"net": network}, workers):
            buffer.extend(examples)
            turns.append(length)

        buffer = buffer[-20000:]
        losses = train_on_buffer(network, optimizer, loss_fn, buffer)
        win_rate = evaluate(
            network,
            games=eval_games,
            simulations=simulations,
            seed=seed * 7919 + iteration,
            fpu_reduction=fpu_reduction,
            search=search,
            workers=workers,
        )
        n_ref = eval_games if ref_eval_games is None else ref_eval_games
        ref_win_rate = (
            evaluate_vs_reference(
                network,
                reference,
                games=n_ref,
                simulations=simulations,
                seed=seed * 104729 + iteration,
                fpu_reduction=fpu_reduction,
                search=search,
                workers=workers,
            )
            if n_ref > 0
            else None
        )

        result = IterationResult(
            iteration=iteration,
            win_rate=win_rate,
            ref_win_rate=ref_win_rate,
            mean_loss=float(np.mean(losses)) if losses else float("nan"),
            examples=len(buffer),
            mean_turns=float(np.mean(turns)),
        )
        results.append(result)
        _write_results(
            out,
            seed,
            iterations,
            games,
            eval_games,
            simulations,
            baseline,
            results,
            value_mix,
        )
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir / f"iter_{iteration:03d}.pt",
                network,
                optimizer,
                iteration=iteration,
                seed=seed,
                action_space_size=action_space_size,
                result=result,
                resume_state={
                    # plain tuples, so other scripts can load the file
                    "buffer": [(e.state, e.policy, e.value) for e in buffer],
                    "results": [r.as_dict() for r in results],
                    "baseline": baseline,
                    "rng": rng_state(),
                },
            )
        print(
            f"iter {iteration:>2}  win_rate {win_rate:.3f}  "
            + (
                f"vs_untrained {ref_win_rate:.3f}  "
                if ref_win_rate is not None
                else ""
            )
            + f"loss {result.mean_loss:.4f}  examples {len(buffer)}  "
            f"mean_turns {result.mean_turns:.1f}",
            flush=True,
        )

    _write_results(
        out,
        seed,
        iterations,
        games,
        eval_games,
        simulations,
        baseline,
        results,
        value_mix,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="save weights after every iteration (iter_NNN.pt)",
    )
    parser.add_argument(
        "--fpu-reduction",
        type=float,
        default=None,
        help="first-play urgency: unvisited moves start at the mean value "
        "of visited siblings minus this; omit for the old flat 0",
    )
    parser.add_argument(
        "--search",
        choices=("puct", "gumbel"),
        default="puct",
        help="root search: AlphaZero PUCT (default) or Gumbel AlphaZero",
    )
    parser.add_argument(
        "--value-mix",
        type=float,
        default=0.0,
        help="soft-Z value target: weight of the root search value against "
        "the game result (0 = result only, the AlphaZero default)",
    )
    parser.add_argument(
        "--ref-eval-games",
        type=int,
        default=None,
        help="games per iteration against the frozen untrained network "
        "with the same search (default: --eval-games; 0 to skip)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for self-play and eval games (one core each); "
        "1 runs inline like before",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest resumable checkpoint in "
        "--checkpoint-dir instead of starting over",
    )
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games=args.games,
        eval_games=args.eval_games,
        simulations=args.simulations,
        seed=args.seed,
        out=args.out,
        checkpoint_dir=args.checkpoint_dir,
        fpu_reduction=args.fpu_reduction,
        resume=args.resume,
        search=args.search,
        value_mix=args.value_mix,
        ref_eval_games=args.ref_eval_games,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
