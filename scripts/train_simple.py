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
import math
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
    # Score against the frozen --anchor checkpoint; None on iterations
    # without an anchor check.
    anchor_score: Optional[float] = None
    # Training diagnostics (#49); None for runs from before they existed.
    lr: Optional[float] = None
    grad_norm: Optional[float] = None
    # Average times each buffer example was sampled this iteration.
    samples_per_example: Optional[float] = None
    # Share of decisive self-play games won by player 0.
    selfplay_p0_rate: Optional[float] = None
    # Network outputs on a fixed probe set of early self-play positions.
    probe_entropy: Optional[float] = None
    probe_value_mean: Optional[float] = None
    probe_value_std: Optional[float] = None
    probe_value_mse: Optional[float] = None
    probe_saturated: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "win_rate": self.win_rate,
            "ref_win_rate": self.ref_win_rate,
            "anchor_score": self.anchor_score,
            "mean_loss": self.mean_loss,
            "examples": self.examples,
            "mean_turns": self.mean_turns,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "samples_per_example": self.samples_per_example,
            "selfplay_p0_rate": self.selfplay_p0_rate,
            "probe_entropy": self.probe_entropy,
            "probe_value_mean": self.probe_value_mean,
            "probe_value_std": self.probe_value_std,
            "probe_value_mse": self.probe_value_mse,
            "probe_saturated": self.probe_saturated,
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


def wilson_interval(
    score: float, games: int, z: float = 1.96
) -> Tuple[float, float]:
    """95% Wilson interval for a score over ``games`` games."""
    if games <= 0:
        return float("nan"), float("nan")
    centre = score + z * z / (2 * games)
    half = z * math.sqrt(
        score * (1 - score) / games + z * z / (4 * games * games)
    )
    denom = 1 + z * z / games
    return (centre - half) / denom, (centre + half) / denom


def plateau_reached(
    scores: List[float], patience: int, tolerance: float = 0.0
) -> bool:
    """True once the last ``patience`` scores all fall clearly below the best.

    ``scores`` are the anchor checks in order. A check within ``tolerance``
    of the best before it (ties included) still counts as keeping pace, so
    one noisy check can't end a run that is holding its level. The run
    stops only after ``patience`` checks in a row land more than
    ``tolerance`` under that best, which is what a collapse looks like. A
    run that holds flat goes on to the --iterations cap.
    """
    if patience <= 0 or len(scores) <= patience:
        return False
    best_before = max(scores[:-patience])
    return all(s < best_before - tolerance for s in scores[-patience:])


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
    grad_norms: Optional[List[float]] = None,
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
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        if grad_norms is not None:
            grad_norms.append(float(norm))
        optimizer.step()
        losses.append(float(total_loss.detach()))

    return losses


def lr_at(
    iteration: int,
    iterations: int,
    lr: float,
    lr_min: float = 1e-4,
    schedule: str = "constant",
) -> float:
    """Learning rate for ``iteration`` (1-based) of ``iterations``.

    ``cosine`` anneals from ``lr`` at iteration 1 to ``lr_min`` at the
    last; it depends only on the iteration, so ``--resume`` picks it up.
    """
    if schedule == "constant" or iterations <= 1:
        return lr
    if schedule != "cosine":
        raise ValueError(f"unknown lr schedule {schedule!r}")
    progress = min(max(iteration - 1, 0), iterations - 1) / (iterations - 1)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


def build_probe(
    examples: List[Example], size: int, seed: int
) -> List[Example]:
    """A fixed sample of positions to track the network's outputs on.

    Uses its own RNG so building it leaves the training streams alone.
    """
    if size <= 0 or not examples:
        return []
    picker = random.Random(seed)
    return picker.sample(examples, min(size, len(examples)))


def probe_stats(
    network: PolicyValueNetwork, probe: List[Example]
) -> Dict[str, float]:
    """Policy entropy and value statistics on the probe positions.

    A collapsing network usually shows it here first: entropy falling
    toward 0 (a policy that stopped exploring) or values pinned near
    +/-1 (an overconfident value head).
    """
    if not probe:
        return {}
    was_training = network.training
    network.eval()
    with torch.no_grad():
        states = torch.stack([network.state_encoder(ex.state) for ex in probe])
        logits, values = network(states)
        log_p = torch.log_softmax(logits, dim=-1)
        entropy = -(log_p.exp() * log_p).sum(dim=-1).mean()
        v = values.view(-1)
        targets = torch.tensor([ex.value for ex in probe], dtype=v.dtype)
        stats = {
            "probe_entropy": float(entropy),
            "probe_value_mean": float(v.mean()),
            "probe_value_std": float(v.std()) if v.numel() > 1 else 0.0,
            "probe_value_mse": float(((v - targets) ** 2).mean()),
            "probe_saturated": float((v.abs() > 0.95).float().mean()),
        }
    network.train(was_training)
    return stats


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
    extra: Optional[Dict[str, object]] = None,
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
    payload.update(extra or {})
    out.write_text(json.dumps(payload, indent=2) + "\n")


def best_anchor(
    results: List[IterationResult],
) -> Optional[Tuple[int, float]]:
    """(iteration, score) of the first-best anchor check, if any."""
    best: Optional[Tuple[int, float]] = None
    for r in results:
        if r.anchor_score is not None and (
            best is None or r.anchor_score > best[1]
        ):
            best = (r.iteration, r.anchor_score)
    return best


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
    anchor: Optional[Path] = None,
    anchor_every: int = 5,
    anchor_games: int = 80,
    plateau: int = 0,
    plateau_tolerance: float = 0.05,
    lr: float = 1e-3,
    lr_schedule: str = "constant",
    lr_min: float = 1e-4,
    train_batches: int = 8,
    batch_size: int = 32,
    buffer_size: int = 20000,
    probe_size: int = 256,
) -> List[IterationResult]:
    if plateau > 0 and anchor is None:
        raise ValueError("--plateau needs --anchor to measure progress")
    seed_everything(seed)
    anchor_net = load_checkpoint(anchor) if anchor is not None else None

    probe = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    action_space_size = len(probe.action_space.action_to_id)

    network = build_simple_network(action_space_size=action_space_size)
    # Taken before any resume load, so a resumed run rebuilds the same
    # untrained weights from the seed.
    reference = copy.deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = PolicyValueLoss(value_weight=1.0, l2_reg=1e-4)

    buffer: List[Example] = []
    results: List[IterationResult] = []
    probe: List[Example] = []
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
        probe = [Example(*item) for item in state.get("probe", [])]
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

    stopped_at: Optional[int] = None

    def extra() -> Dict[str, object]:
        config: Dict[str, object] = {
            "lr": lr,
            "lr_schedule": lr_schedule,
            "lr_min": lr_min,
            "train_batches": train_batches,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "probe_size": probe_size,
        }
        if anchor is None:
            return config
        best = best_anchor(results)
        return {
            **config,
            "anchor": str(anchor),
            "anchor_every": anchor_every,
            "anchor_games": anchor_games,
            "plateau": plateau,
            "plateau_tolerance": plateau_tolerance,
            "best_anchor_iteration": best[0] if best else None,
            "best_anchor_score": best[1] if best else None,
            "stopped_at": stopped_at,
        }

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
        decisive = []
        for winner, length, examples in run_games(
            jobs, {"net": network}, workers
        ):
            buffer.extend(examples)
            turns.append(length)
            if winner is not None:
                decisive.append(winner == 0)
        if not probe:
            probe = build_probe(buffer, probe_size, seed * 31337)

        buffer = buffer[-buffer_size:]
        current_lr = lr_at(iteration, iterations, lr, lr_min, lr_schedule)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        grad_norms: List[float] = []
        losses = train_on_buffer(
            network,
            optimizer,
            loss_fn,
            buffer,
            batch_size=batch_size,
            epochs=train_batches,
            grad_norms=grad_norms,
        )
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
        anchor_score = None
        if anchor_net is not None and (
            iteration % anchor_every == 0 or iteration == iterations
        ):
            anchor_score = evaluate_vs_reference(
                network,
                anchor_net,
                games=anchor_games,
                simulations=simulations,
                seed=seed * 15485863 + iteration,
                fpu_reduction=fpu_reduction,
                search=search,
                workers=workers,
            )

        result = IterationResult(
            iteration=iteration,
            win_rate=win_rate,
            ref_win_rate=ref_win_rate,
            anchor_score=anchor_score,
            mean_loss=float(np.mean(losses)) if losses else float("nan"),
            examples=len(buffer),
            mean_turns=float(np.mean(turns)),
            lr=current_lr,
            grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
            samples_per_example=(
                len(losses) * batch_size / len(buffer) if buffer else None
            ),
            selfplay_p0_rate=(float(np.mean(decisive)) if decisive else None),
            **probe_stats(network, probe),
        )
        results.append(result)
        scores = [
            r.anchor_score for r in results if r.anchor_score is not None
        ]
        if anchor_score is not None and plateau_reached(
            scores, plateau, plateau_tolerance
        ):
            stopped_at = iteration
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
            extra(),
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
                    "probe": [(e.state, e.policy, e.value) for e in probe],
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
        if result.probe_entropy is not None:
            print(
                f"         lr {current_lr:.2e}  grad_norm "
                f"{result.grad_norm or 0.0:.3f}  entropy "
                f"{result.probe_entropy:.3f}  value "
                f"{result.probe_value_mean:+.3f}"
                f"\u00b1{result.probe_value_std:.3f}  "
                f"saturated {result.probe_saturated:.2f}",
                flush=True,
            )
        if anchor_score is not None:
            lo, hi = wilson_interval(anchor_score, anchor_games)
            best = best_anchor(results)
            is_best = best is not None and best[0] == iteration
            print(
                f"         vs_anchor {anchor_score:.3f}  "
                f"(95% CI {lo:.3f}-{hi:.3f}, {anchor_games} games)"
                + ("  new best" if is_best else ""),
                flush=True,
            )
            if is_best and checkpoint_dir is not None:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    network,
                    optimizer,
                    iteration=iteration,
                    seed=seed,
                    action_space_size=action_space_size,
                    result=result,
                )
        if stopped_at is not None:
            print(
                f"stopping: vs_anchor has been more than "
                f"{plateau_tolerance:.2f} below its best for {plateau} checks",
                flush=True,
            )
            break

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
        extra(),
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
        "--anchor",
        type=Path,
        default=None,
        help="frozen checkpoint to play every --anchor-every iterations; "
        "the score against it is the progress signal for --plateau",
    )
    parser.add_argument(
        "--anchor-every",
        type=int,
        default=5,
        help="iterations between anchor checks (the last one always checks)",
    )
    parser.add_argument(
        "--anchor-games",
        type=int,
        default=80,
        help="seat-swapped greedy games per anchor check (80 gives a 95%% "
        "CI of about +/-0.11; 40 was about +/-0.15, too noisy to rank seeds)",
    )
    parser.add_argument(
        "--plateau",
        type=int,
        default=0,
        help="stop once this many anchor checks in a row land more than "
        "--plateau-tol below the best so far (0 = off; --iterations stays "
        "the hard cap; 4 is a sensible value)",
    )
    parser.add_argument(
        "--plateau-tol",
        type=float,
        default=0.05,
        help="how far below the best an anchor check can land and still "
        "count as keeping pace",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate"
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="constant (default) or cosine decay from --lr to --lr-min "
        "over --iterations",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-4,
        help="final learning rate for --lr-schedule cosine",
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=8,
        help="gradient steps per iteration",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="examples per step"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20000,
        help="replay buffer cap (newest examples kept)",
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=256,
        help="fixed iteration-1 positions to log policy entropy and value "
        "stats on each iteration (0 to skip)",
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
        anchor=args.anchor,
        anchor_every=args.anchor_every,
        anchor_games=args.anchor_games,
        plateau=args.plateau,
        plateau_tolerance=args.plateau_tol,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        probe_size=args.probe_size,
    )


if __name__ == "__main__":
    main()
