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
import copy
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


def matchup_decks_arg(args: argparse.Namespace) -> Optional[Tuple[str, ...]]:
    """Decks for per-matchup value heads (``--matchup-decks``), or None.

    Off by default: the shared value head is unchanged.
    """
    names = tuple(
        t.strip()
        for t in str(getattr(args, "matchup_decks", "")).split(",")
        if t.strip()
    )
    return names or None


def reset_value_head(net: ForgePointerNet) -> None:
    """Re-initialize the value head, keeping the policy trunk.

    Fork-and-flush's flush (arXiv:2610.07447): a probe that trains from
    an inherited checkpoint but starts its value head over can land in a
    different region than the lineage it came from.
    """
    for module in net.value_head.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    for head in getattr(net, "matchup_value_heads", None) or ():
        for module in head.modules():
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


def mix_expert_rows(
    samples: List[Tuple[Dict[str, Any], float]],
    frac: float,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[int], Dict[str, Any]]:
    """Replay expert-source rows up to ``frac`` of the training pool (#96).

    The queued buffer-mixing lever: expert rows' value labels are real
    game outcomes, while the self-play rows' value targets are the noise
    the value head cannot fit at any weight (vc4, #96). Mixing repeats
    the pool's expert rows, cyclically, until they make up ``frac`` of
    the training samples, so the value gradient lands mostly on
    learnable targets. The holdout is carved before mixing and stays
    unique. At ``frac <= 0`` (or with no expert rows) the pool passes
    through unchanged; returns the pool, the source index of each
    appended copy (for weight extension), and meta.
    """
    if frac <= 0:
        return samples, [], {}
    if frac >= 1.0:
        raise ValueError("expert-mix-frac must be below 1.0")
    meta: Dict[str, Any] = {"expert_mix_frac": frac}
    expert = [
        i for i, (d, _) in enumerate(samples) if sample_source(d) == "expert"
    ]
    if not expert:
        meta["expert_mix_skipped"] = "no expert rows"
        return samples, [], meta
    others = len(samples) - len(expert)
    target = int(round(frac / (1.0 - frac) * others))
    extra = target - len(expert)
    meta["expert_rows_base"] = len(expert)
    meta["expert_rows_added"] = max(extra, 0)
    meta["expert_rows_final"] = len(expert) + max(extra, 0)
    if extra <= 0:
        return samples, [], meta
    fill = [expert[i % len(expert)] for i in range(extra)]
    return samples + [samples[i] for i in fill], fill, meta


def blend_value_targets(
    net: ForgePointerNet,
    samples: List[Tuple[Dict[str, Any], float]],
    blend: float,
) -> Optional[Tuple[List[Tuple[Dict[str, Any], float]], Dict[str, Any]]]:
    """Blend each self-play row's value target with a frozen net's value (#96).

    The value-target lever: self-play rows' recorded outcomes are the
    noise the value head cannot fit at any weight (vc4, #96), while
    expert rows' real outcomes fit fine. With ``blend > 0`` each
    self-play row's target becomes ``(1 - blend) * z + blend * V(s)``
    where ``V`` is a frozen copy of the round's starting net (a target
    network, deep-copied before any training step), computed once per
    round before the epochs run. Blending trades label variance for
    bias toward the round's starting value read, the standard target
    network trade. Expert-source rows keep their real outcomes and the
    holdout keeps recorded outcomes everywhere, so holdout metrics stay
    comparable across runs. Returns ``(new_samples, meta)``; ``None``
    when ``blend <= 0`` so the default path stays byte-identical.
    """
    if blend <= 0:
        return None
    if blend >= 1.0:
        raise ValueError("value-target-blend must be below 1.0")
    frozen = copy.deepcopy(net).eval()
    blended: List[Tuple[Dict[str, Any], float]] = []
    rows = 0
    with torch.no_grad():
        for d, z in samples:
            if sample_source(d) == "expert":
                blended.append((d, z))
                continue
            v = float(frozen.distill(d).value)
            blended.append((d, (1.0 - blend) * z + blend * v))
            rows += 1
    meta = {"value_target_blend": blend, "value_target_rows": rows}
    return blended, meta


def search_value_targets(
    samples: List[Tuple[Dict[str, Any], float]],
    enabled: bool,
) -> Optional[Tuple[List[Tuple[Dict[str, Any], float]], Dict[str, Any]]]:
    """Swap self-play rows' value target to the search's root value (#96).

    The search-derived value-target lever: ``ForgeSearchResult.rootValue``
    planar-nexus writes that read as ``rootValue`` (camelCase) in the JSONL
    rows it records; ``root_value`` stays accepted so older local
    constructions keep working.
    (planar-nexus) is the search's own evaluation at each net decision,
    from the deciding seat's point of view -- the same seat convention
    and [-1, 1] scale as the recorded outcome ``z``. With ``enabled``
    each self-play row carrying ``root_value`` trains toward the
    search's read instead of the recorded game outcome (the label the
    value head could not fit under any scheme tried: vc4, deck identity,
    expert mix, the target blend). Expert rows keep real outcomes (the
    Expert AI's moves are not search-scored) and the holdout keeps
    recorded outcomes, so holdout metrics stay comparable across runs.
    Rows without a numeric value under either key fall back to ``z`` and
    are counted. Returns ``(new_samples, meta)``; ``None`` when not
    enabled so the default path stays byte-identical. Runs before
    ``blend_value_targets``, so ``--value-target-blend`` blends toward
    ``V(s)`` on top of the search base: ``(1 - B) * q + B * V(s)``.
    """
    if not enabled:
        return None
    swapped: List[Tuple[Dict[str, Any], float]] = []
    rows = 0
    fallback = 0
    for d, z in samples:
        raw = d.get("root_value")
        if not isinstance(raw, (int, float)):
            raw = d.get("rootValue")
        if sample_source(d) != "expert" and isinstance(raw, (int, float)):
            swapped.append((d, float(raw)))
            rows += 1
        else:
            swapped.append((d, z))
            if sample_source(d) != "expert":
                fallback += 1
    meta = {
        "value_source": "search",
        "value_search_rows": rows,
        "value_source_fallback": fallback,
    }
    return swapped, meta


def nstep_value_targets(
    net: ForgePointerNet,
    samples: List[Tuple[Dict[str, Any], float]],
    traj: Sequence[Tuple[int, int]],
    state_lookup: Dict[Tuple[int, int], Dict[str, Any]],
    nstep: int,
) -> Optional[Tuple[List[Tuple[Dict[str, Any], float]], Dict[str, Any]]]:
    """Bootstrap self-play rows' value targets from a later state (#96).

    The trajectory-level value-target lever: the holdout's value-by-turn
    diagnostic says the value head cannot fit the recorded outcome in
    the opening turns because it cannot tell which deck it holds, while
    the same head reads mid-game states several times better. With
    ``nstep > 0`` each self-play row's target becomes the frozen
    starting net's value at the state ``nstep`` recorded decisions
    deeper in the same game, seat-adjusted: ``V(s_{t+N})`` when both
    rows share a seat, ``-V(s_{t+N})`` when seats differ (``V`` reads
    from the deciding seat). No intermediate rewards exist, so with
    gamma 1 the n-step return is exactly that bootstrapped value. Rows
    within ``N`` of the game's end keep ``z`` (nothing deeper to
    bootstrap from), expert rows keep real outcomes, and self-play rows
    whose seat is unknown keep ``z`` and are counted. Bootstrap states
    come from the full buffer, so a state carved into the holdout can
    still serve as a target source; the holdout itself keeps recorded
    outcomes, so holdout metrics stay comparable across runs. Trades
    the outcome's variance for the frozen net's bias, anchored where
    the head can fit. Runs before ``blend_value_targets``, so
    ``--value-target-blend`` blends toward ``V(s)`` on top of the
    n-step base. Do not combine with ``--flush-value-head``: a reset
    value head has nothing to bootstrap from. Returns
    ``(new_samples, meta)``; ``None`` when ``nstep <= 0`` so the
    default path stays byte-identical.
    """
    if nstep <= 0:
        return None
    if len(traj) != len(samples):
        raise ValueError("traj length must match samples")
    frozen = copy.deepcopy(net).eval()
    bootstrapped: List[Tuple[Dict[str, Any], float]] = []
    rows = 0
    terminal = 0
    unknown = 0
    with torch.no_grad():
        for (d, z), (gi, si) in zip(samples, traj):
            if sample_source(d) == "expert":
                bootstrapped.append((d, z))
                continue
            nxt = state_lookup.get((gi, si + nstep))
            seat = d.get("seat")
            if nxt is None or seat is None or nxt.get("seat") is None:
                bootstrapped.append((d, z))
                if nxt is None:
                    terminal += 1
                else:
                    unknown += 1
                continue
            v = float(frozen.distill(nxt).value)
            bootstrapped.append((d, v if seat == nxt.get("seat") else -v))
            rows += 1
    meta = {
        "value_target_nstep": nstep,
        "value_target_rows": rows,
        "value_target_terminal_rows": terminal,
        "value_target_seat_unknown": unknown,
    }
    return bootstrapped, meta


HOLDOUT_FRAC = 0.1
"""Default fraction of decisions carved out as a fixed holdout (#96)."""


def carve_holdout_indexed(
    samples: List[Tuple[Dict[str, Any], float]],
    frac: float,
    seed: int,
) -> Tuple[
    List[Tuple[Dict[str, Any], float]],
    List[Tuple[Dict[str, Any], float]],
    List[int],
]:
    """``carve_holdout`` that also returns the train pool's flat indices.

    The n-step value-target lever maps each train row back to its
    trajectory position, which only the flat index carries.
    """
    n = int(len(samples) * frac)
    n = min(n, len(samples) // 2) if samples else 0
    if n <= 0:
        return list(samples), [], list(range(len(samples)))
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(samples)), n))
    return (
        [s for i, s in enumerate(samples) if i not in idx],
        [samples[i] for i in sorted(idx)],
        [i for i in range(len(samples)) if i not in idx],
    )


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
    train, holdout, _ = carve_holdout_indexed(samples, frac, seed)
    return train, holdout


VALUE_TURN_BUCKETS = ("1-3", "4-8", "9-16", "17+")


def value_turn_bucket(d: Dict[str, Any]) -> str:
    """Turn bucket for the holdout's value-error-by-turn read (#96).

    Deck identity is only weakly visible in a small opening hand; if the
    mixed-deck matchup is noise the value head cannot explain, value error
    should concentrate in the early buckets. Rows without a usable turn
    land in ``unknown``.
    """
    raw = d.get("turn")
    try:
        turn = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return "unknown"
    if turn <= 0:
        return "unknown"
    if turn <= 3:
        return "1-3"
    if turn <= 8:
        return "4-8"
    if turn <= 16:
        return "9-16"
    return "17+"


def holdout_loss(
    net: ForgePointerNet,
    samples: Sequence[Tuple[Dict[str, Any], float]],
    batch: int = 64,
    value_coef: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Mean loss statistics on the holdout with gradients off (#96).

    Same ``distill_loss`` terms as training; the split between a policy
    drop and a value drop is the no-signal versus noise-overfit read.
    Also reports value error bucketed by turn (``value_by_turn``, with
    row counts in ``value_by_turn_n``): the deck-identity read, since a
    mixed-deck matchup is hardest to infer early.
    """
    if not samples:
        return None
    net.eval()
    totals: Dict[str, float] = {}
    vturn: Dict[str, float] = {}
    nturn: Dict[str, int] = {}
    n = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start : start + batch]
            steps = [net.distill(d) for d, _ in chunk]
            _, stats = distill_loss(steps, [r for _, r in chunk], value_coef)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
            val = torch.stack([s.value for s in steps])
            errs = (val - val.new_tensor([r for _, r in chunk])) ** 2
            for (d, _), err in zip(chunk, errs.tolist()):
                b = value_turn_bucket(d)
                vturn[b] = vturn.get(b, 0.0) + err
                nturn[b] = nturn.get(b, 0) + 1
    net.train()
    out: Dict[str, Any] = {k: round(v / n, 5) for k, v in totals.items()}
    order = (*VALUE_TURN_BUCKETS, "unknown")
    if nturn:
        out["value_by_turn"] = {
            b: round(vturn[b] / nturn[b], 5) for b in order if b in vturn
        }
        out["value_by_turn_n"] = {b: nturn[b] for b in order if b in nturn}
    return out


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
            "config": {
                "card_dim": net.card_dim,
                "state_dim": net.state_dim,
                **(
                    {"matchup_decks": list(net.matchup_decks)}
                    if net.matchup_decks
                    else {}
                ),
            },
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
    traj: List[Tuple[int, int]] = []
    state_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if args.value_target_nstep > 0:
        for gi, game in enumerate(games):
            for si, decision in enumerate(game["decisions"]):
                traj.append((gi, si))
                state_lookup[(gi, si)] = decision
    train_samples, holdout, train_idx = carve_holdout_indexed(
        samples, args.holdout_frac, args.seed + 1
    )
    searched = search_value_targets(
        train_samples, args.value_source == "search"
    )
    value_source_meta: Dict[str, Any] = {}
    if searched is not None:
        train_samples, value_source_meta = searched
    matchup = matchup_decks_arg(args)
    if args.init is not None:
        net, prev = load_pointer_net(str(args.init), matchup_decks=matchup)
        parent_games = int((prev or {}).get("selfplay_games_total", 0))
        if args.flush_value_head:
            reset_value_head(net)
    else:
        net, parent_games = ForgePointerNet(matchup_decks=matchup), 0
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.value_target_nstep > 0 and args.flush_value_head:
        raise ValueError(
            "--flush-value-head resets the value head the n-step targets"
            " bootstrap from; drop one of the two"
        )
    value_nstep_meta: Dict[str, Any] = {}
    if args.value_target_nstep > 0:
        nstepped = nstep_value_targets(
            net,
            train_samples,
            [traj[i] for i in train_idx],
            state_lookup,
            args.value_target_nstep,
        )
        if nstepped is not None:
            train_samples, value_nstep_meta = nstepped
    weights: Optional[List[float]] = None
    if args.kl_weighted:
        weights = kl_weights(net, train_samples, eps=args.kl_eps)
    train_samples, mix_fill, mix_meta = mix_expert_rows(
        train_samples, args.expert_mix_frac
    )
    if mix_fill:
        base = (
            weights
            if weights is not None
            else [1.0] * (len(train_samples) - len(mix_fill))
        )
        weights = base + [base[i] for i in mix_fill]
    weights = apply_expert_weight(
        weights, train_samples, args.expert_policy_weight
    )
    blended = blend_value_targets(net, train_samples, args.value_target_blend)
    value_blend_meta: Dict[str, Any] = {}
    if blended is not None:
        train_samples, value_blend_meta = blended
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
        **mix_meta,
        **value_source_meta,
        **value_nstep_meta,
        **value_blend_meta,
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
    if net.matchup_decks:
        meta["matchup_decks"] = list(net.matchup_decks)
        meta["matchup_value_heads"] = len(net.matchup_value_heads)
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
        "--value-target-blend",
        type=float,
        default=0.0,
        help=(
            "blend self-play value targets with a frozen copy of the "
            "round's starting net (0 = recorded outcomes, the default); "
            "expert rows and the holdout keep recorded outcomes"
        ),
    )
    ap.add_argument(
        "--value-source",
        choices=("outcome", "search"),
        default="outcome",
        help=(
            "self-play value-target source (manamind#96): 'outcome' keeps "
            "the recorded game result, 'search' trains self-play rows "
            "toward the search's recorded root evaluation (root_value, "
            "the deciding seat's point of view, same [-1, 1] scale); rows "
            "without root_value fall back to the outcome and are counted; "
            "expert rows and the holdout keep recorded outcomes"
        ),
    )
    ap.add_argument(
        "--value-target-nstep",
        type=int,
        default=0,
        help=(
            "bootstrap self-play value targets from the frozen starting "
            "net's value N recorded decisions deeper in the same game, "
            "seat-adjusted (0 = recorded outcomes, the default); rows "
            "within N of the end, expert rows, and the holdout keep "
            "recorded outcomes"
        ),
    )
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
        "--expert-mix-frac",
        type=float,
        default=0.0,
        help="replay expert-source rows up to this share of the train "
        "pool (buffer mixing, manamind#96); 0 disables",
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
        "--matchup-decks",
        default="",
        help="comma-separated deck names for per-matchup value heads"
        " (#96 deck identity v2); empty keeps the shared value head",
    )
    ap.add_argument(
        "--onnx", action="store_true", help="also export forge_pointer.onnx"
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    meta = run(build_parser().parse_args(argv))
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}))


if __name__ == "__main__":
    main()
