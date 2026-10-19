"""Train ForgePointerNet on Planar Nexus self-play records (#87).

Planar Nexus plays search-vs-search games with ``scripts/selfplay-forge.ts``
and writes one game per JSONL line: the bridge-shaped ``decisions``, the
search's improved policy ``pi`` on each, and ``returns`` (+1/-1/0 from the
deciding seat). This trains the policy heads toward ``pi`` and the value
head toward the result, over a replay buffer of the newest games, then
writes ``last.pt`` (the ``train_forge`` checkpoint format) and, with
``--onnx``, the ONNX model Planar Nexus loads for the next round.

Example::

    PYTHONPATH=src python -m manamind.training.train_selfplay \\
        --data runs/selfplay/round1 --init runs/selfplay/round0/last.pt \\
        --epochs 2 --out runs/selfplay/round1 --onnx
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from manamind.models.forge_pointer import (
    ForgePointerNet,
    distill_loss,
    load_pointer_net,
)
from manamind.training.audit_labels import audit_decision

BUFFER_GAMES = 20_000
"""Replay buffer cap from #87: the newest this many games."""


def reset_value_head(net: ForgePointerNet) -> None:
    """Re-initialize the value head, keeping the policy trunk.

    Fork-and-flush's flush (arXiv:2610.07447): a probe that trains from
    an inherited checkpoint but starts its value head over can land in a
    different region than the lineage it came from.
    """
    for module in net.value_head.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()


def _open(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with path.open(encoding="utf-8") as fh:
            yield from fh


def selfplay_files(paths: Sequence[Path]) -> List[Path]:
    """``*.jsonl`` / ``*.jsonl.gz`` files under ``paths``, oldest first."""
    found: List[Path] = []
    for p in paths:
        if p.is_dir():
            found += [
                f
                for f in p.rglob("*")
                if f.name.endswith((".jsonl", ".jsonl.gz"))
            ]
        elif p.exists():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(set(found), key=lambda f: (f.stat().st_mtime, str(f)))


def load_selfplay(
    paths: Sequence[Path], buffer_games: int = BUFFER_GAMES
) -> List[Dict[str, Any]]:
    """The newest ``buffer_games`` games from ``paths`` (oldest first)."""
    games: List[Dict[str, Any]] = []
    for f in selfplay_files(paths):
        for line in _open(f):
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            if len(g.get("decisions", [])) != len(g.get("returns", [])):
                raise ValueError(f"{f}: decisions and returns differ")
            games.append(g)
            if len(games) > 2 * buffer_games:
                games = games[-buffer_games:]
    return games[-buffer_games:] if buffer_games > 0 else games


def flatten(
    games: Sequence[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], float]]:
    """``(decision, return)`` pairs from every game."""
    return [
        (d, float(r))
        for g in games
        for d, r in zip(g["decisions"], g["returns"])
    ]


def kl_weights(
    net: ForgePointerNet,
    samples: Sequence[Tuple[Dict[str, Any], float]],
    eps: float = 0.01,
) -> List[float]:
    """Per-sample weights from the audit's target-vs-prior metric (#96).

    ``metric`` is the audit KL for priority/block decisions and the mean
    absolute probability gap for attack (whose target is an independent
    attack probability, not a distribution); weights are
    ``(metric + eps) / (mean metric + eps)``, so the average weight is 1,
    the disagreement tail is oversampled, and near-identity moves keep
    only the ``eps`` floor. Decisions ``audit_decision`` rejects
    (unusable shape) weight exactly 1.
    """
    metrics: List[Optional[float]] = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for d, _ in samples:
            read = audit_decision(net, d)
            if read is None:
                metrics.append(None)
            elif read["kind"] == "attack":
                metrics.append(read["abs_diff"])
            else:
                metrics.append(read["kl"])
    if was_training:
        net.train()
    usable = [m for m in metrics if m is not None]
    if not usable:
        return [1.0] * len(samples)
    denom = sum(usable) / len(usable) + eps
    return [1.0 if m is None else (m + eps) / denom for m in metrics]


def train_epochs(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    samples: Sequence[Tuple[Dict[str, Any], float]],
    epochs: int = 1,
    batch: int = 64,
    value_coef: float = 1.0,
    seed: int = 0,
    log: Optional[List[Dict[str, Any]]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Minibatch passes over ``samples``; returns the last epoch's means.

    With ``weights``, each epoch draws ``len(samples)`` indices with
    replacement weighted by ``weights`` (weighted resampling); otherwise
    it shuffles uniformly.
    """
    if not samples:
        raise ValueError("no self-play decisions to train on")
    if weights is not None and len(weights) != len(samples):
        raise ValueError("weights length must match samples")
    rng = random.Random(seed)
    order = list(range(len(samples)))
    net.train()
    means: Dict[str, float] = {}
    for epoch in range(epochs):
        if weights is not None:
            order = rng.choices(
                range(len(samples)), weights=weights, k=len(samples)
            )
        else:
            rng.shuffle(order)
        totals: Dict[str, float] = {}
        n = 0
        for start in range(0, len(order), batch):
            chunk = [samples[i] for i in order[start : start + batch]]
            steps = [net.distill(d) for d, _ in chunk]
            loss, stats = distill_loss(
                steps, [r for _, r in chunk], value_coef
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        means = {k: v / n for k, v in totals.items()}
        if log is not None:
            log.append({"epoch": epoch, **means})
    return means


def apply_expert_weight(
    weights: Optional[List[float]],
    samples: Sequence[Tuple[Dict[str, Any], float]],
    expert_weight: float,
) -> Optional[List[float]]:
    """Scale expert-source samples' training weights (#96).

    With ``expert_weight == 1.0`` the weights pass through unchanged;
    otherwise each expert-source sample's weight is multiplied by
    ``expert_weight`` (a uniform 1.0 base when ``weights`` is None), so
    Expert AI decisions can be trusted more or less than the net's own.
    """
    if expert_weight == 1.0:
        return weights
    factors = [
        expert_weight if sample_source(d) == "expert" else 1.0
        for d, _ in samples
    ]
    if weights is None:
        return factors
    return [w * f for w, f in zip(weights, factors)]


def sample_source(decision: Dict[str, Any]) -> str:
    """Which brain produced a decision: the net (default) or the Expert."""
    return str(decision.get("source", "net"))


HOLDOUT_FRAC = 0.1
"""Default fraction of decisions carved out as a fixed holdout (#96)."""


def carve_holdout(
    samples: List[Tuple[Dict[str, Any], float]],
    frac: float,
    seed: int,
) -> Tuple[
    List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]
]:
    """Split samples into ``(train, holdout)`` deterministically (#96).

    The holdout is drawn at random with ``seed`` so a run is reproducible,
    and is capped at half the samples so tiny buffers still train.
    """
    n = int(len(samples) * frac)
    n = min(n, len(samples) // 2) if samples else 0
    if n <= 0:
        return list(samples), []
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(samples)), n))
    return (
        [s for i, s in enumerate(samples) if i not in idx],
        [samples[i] for i in sorted(idx)],
    )


def holdout_loss(
    net: ForgePointerNet,
    samples: Sequence[Tuple[Dict[str, Any], float]],
    batch: int = 64,
    value_coef: float = 1.0,
) -> Optional[Dict[str, float]]:
    """Mean loss statistics on the holdout with gradients off (#96).

    Same ``distill_loss`` terms as training; the split between a policy
    drop and a value drop is the no-signal versus noise-overfit read.
    """
    if not samples:
        return None
    net.eval()
    totals: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start : start + batch]
            steps = [net.distill(d) for d, _ in chunk]
            _, stats = distill_loss(steps, [r for _, r in chunk], value_coef)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
    net.train()
    return {k: round(v / n, 5) for k, v in totals.items()}


def save_checkpoint(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    out_dir: Path,
    meta: Dict[str, Any],
) -> Path:
    """Write ``last.pt`` in ``train_forge``'s format (atomic rename)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "last.pt.tmp"
    torch.save(
        {
            "network": net.state_dict(),
            "optimizer": opt.state_dict(),
            "config": {"card_dim": net.card_dim, "state_dim": net.state_dim},
            "meta": meta,
        },
        tmp,
    )
    path = out_dir / "last.pt"
    tmp.replace(path)
    return path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    games = load_selfplay(args.data, args.buffer_games)
    samples = flatten(games)
    train_samples, holdout = carve_holdout(
        samples, args.holdout_frac, args.seed + 1
    )
    if args.init is not None:
        net, prev = load_pointer_net(str(args.init))
        parent_games = int((prev or {}).get("selfplay_games_total", 0))
        if args.flush_value_head:
            reset_value_head(net)
    else:
        net, parent_games = ForgePointerNet(), 0
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    weights: Optional[List[float]] = None
    if args.kl_weighted:
        weights = kl_weights(net, train_samples, eps=args.kl_eps)
    weights = apply_expert_weight(
        weights, train_samples, args.expert_policy_weight
    )
    t0 = time.time()
    history: List[Dict[str, Any]] = []
    stats = train_epochs(
        net,
        opt,
        train_samples,
        epochs=args.epochs,
        batch=args.batch,
        value_coef=args.value_coef,
        seed=args.seed,
        log=history,
        weights=weights,
    )
    meta: Dict[str, Any] = {
        "phase": "selfplay",
        "init": str(args.init) if args.init is not None else None,
        "buffer_games": len(games),
        "decisions": len(samples),
        "selfplay_games_total": parent_games + len(games),
        "epochs": args.epochs,
        "holdout_frac": args.holdout_frac,
        "holdout_examples": len(holdout),
        "kl_weighted": args.kl_weighted,
        "kl_eps": args.kl_eps if args.kl_weighted else None,
        "expert_policy_weight": args.expert_policy_weight,
        "expert_samples": sum(
            1 for d, _ in train_samples if sample_source(d) == "expert"
        ),
        "secs": round(time.time() - t0, 1),
        "history": history,
        **{f"final_{k}": round(v, 5) for k, v in stats.items()},
    }
    expert_holdout = [s for s in holdout if sample_source(s[0]) == "expert"]
    imitation: Optional[Dict[str, Any]] = None
    if args.imitation_epochs > 0:
        expert_train = [
            (d, r) for d, r in train_samples if sample_source(d) == "expert"
        ]
        if not expert_train:
            imitation = {
                "epochs": args.imitation_epochs,
                "skipped": "no expert rows",
            }
        else:
            imitation = {
                "epochs": args.imitation_epochs,
                "samples": len(expert_train),
            }
            if expert_holdout:
                imitation["holdout_expert_pre"] = holdout_loss(
                    net,
                    expert_holdout,
                    batch=args.batch,
                    value_coef=args.value_coef,
                )
            imitation["lr_scale"] = args.imitation_lr_scale
            base_lr = opt.param_groups[0]["lr"]
            if args.imitation_lr_scale != 1.0:
                for group in opt.param_groups:
                    group["lr"] = base_lr * args.imitation_lr_scale
            im_history: List[Dict[str, Any]] = []
            im_stats = train_epochs(
                net,
                opt,
                expert_train,
                epochs=args.imitation_epochs,
                batch=args.batch,
                value_coef=args.value_coef,
                seed=args.seed + 9000,
                log=im_history,
            )
            if args.imitation_lr_scale != 1.0:
                for group in opt.param_groups:
                    group["lr"] = base_lr
            imitation["history"] = im_history
            imitation.update(
                {f"final_{k}": round(v, 5) for k, v in im_stats.items()}
            )
        meta["imitation"] = imitation
    holdout_stats = holdout_loss(
        net, holdout, batch=args.batch, value_coef=args.value_coef
    )
    if holdout_stats is not None:
        meta["holdout"] = holdout_stats
    if expert_holdout:
        meta["holdout_expert"] = holdout_loss(
            net,
            expert_holdout,
            batch=args.batch,
            value_coef=args.value_coef,
        )
    save_checkpoint(net, opt, args.out, meta)
    if args.onnx:
        from manamind.models.forge_pointer_onnx import export

        meta["onnx"] = str(export(net, args.out))
    (args.out / "selfplay_train.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="self-play JSONL(.gz) files or directories",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="start from .pt")
    ap.add_argument("--buffer-games", type=int, default=BUFFER_GAMES)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=HOLDOUT_FRAC,
        help="fraction of decisions held out of training and evaluated "
        "afterward (0 disables; #96 diagnosis)",
    )
    ap.add_argument(
        "--kl-weighted",
        action="store_true",
        help="resample each epoch by the target-vs-prior audit metric so "
        "the disagreement tail drives the policy (#96)",
    )
    ap.add_argument(
        "--kl-eps",
        type=float,
        default=0.01,
        help="eps floor in the KL weighting (with --kl-weighted)",
    )
    ap.add_argument(
        "--expert-policy-weight",
        type=float,
        default=1.0,
        help="policy-loss weight multiplier for source=='expert' decisions "
        "(expert-decision recording, manamind#96); 1.0 disables",
    )
    ap.add_argument(
        "--imitation-epochs",
        type=int,
        default=0,
        help="after the main phase, run N extra epochs on expert-source "
        "rows only (dedicated imitation pass, manamind#96); 0 disables",
    )
    ap.add_argument(
        "--imitation-lr-scale",
        type=float,
        default=1.0,
        help="learning-rate multiplier for the imitation phase "
        "(manamind#96); 1.0 uses the main phase's lr unchanged",
    )
    ap.add_argument(
        "--flush-value-head",
        action="store_true",
        help="re-initialize the value head after loading --init",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--onnx", action="store_true", help="also export forge_pointer.onnx"
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    meta = run(build_parser().parse_args(argv))
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}))


if __name__ == "__main__":
    main()
