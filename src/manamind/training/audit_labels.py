"""Read-only label audit of Planar Nexus self-play buffers (#96).

Measures, per round buffer and per decision kind, how far the stored
search target ``pi`` sits from the champion net's own raw prior: the KL
divergence ``KL(pi || prior)`` for priority and block (distributions over
the same simplex) and per-creature agreement for attack (whose target is
an independent attack probability, not a distribution). Also scores the
value head against the game returns: Pearson correlation, MSE, and the
no-skill floor (the MSE of predicting the mean return). Reads the same
``jsonl.gz`` buffers training reads; plays no games and trains nothing.

Example::

    PYTHONPATH=src python -m manamind.training.audit_labels \\
        --run-dir ~/manamind-runs/selfplay_rg2614_holdout \\
        --out audit_labels.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from manamind.models.forge_pointer import ForgePointerNet, load_pointer_net

KINDS = ("priority", "attack", "block")


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``KL(p || q)`` over the last axis; ``p`` may contain zeros."""
    mask = p > 0
    return (p[mask] * (p[mask].log() - q[mask].log().clamp(min=-30))).sum(-1)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation; ``None`` when either side is constant."""
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx, syy, sxy = (
        sum(v * v for v in dx),
        sum(v * v for v in dy),
        sum(a * b for a, b in zip(dx, dy)),
    )
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def audit_decision(
    net: ForgePointerNet, d: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """One decision's prior-vs-target read, or ``None`` when unusable."""
    pi = d.get("pi")
    t = d.get("t")
    if pi is None or t not in KINDS:
        return None
    with torch.no_grad():
        state = net.encode_state(d)
        value = float(net.value(state))
        if t == "priority":
            options = d.get("options", [])
            if len(pi) != len(options) + 1:
                return None
            p = torch.tensor(pi, dtype=torch.float32).clamp(min=0)
            p = (p / p.sum().clamp(min=1e-8)).unsqueeze(0)
            q = F.softmax(net.priority_logits(state, options), -1).unsqueeze(0)
            return {
                "kind": t,
                "kl": float(kl_divergence(p, q)),
                "pi_entropy": float(-(p * p.clamp(min=1e-12).log()).sum()),
                "prior_entropy": float(-(q * q.clamp(min=1e-12).log()).sum()),
                "top1_agree": float(int(p.argmax() == q.argmax())),
                "value": value,
            }
        if t == "attack":
            creatures = d.get("options", [])
            if not creatures or len(pi) != len(creatures):
                return None
            tgt = torch.tensor(pi, dtype=torch.float32).clamp(0.0, 1.0)
            logits = net.attack_logits(state, creatures)
            probs = torch.sigmoid(logits)
            return {
                "kind": t,
                "prior_bce": float(
                    F.binary_cross_entropy_with_logits(
                        logits, tgt, reduction="mean"
                    )
                ),
                "abs_diff": float((tgt - probs).abs().mean()),
                "pi_entropy": float(
                    -(tgt * tgt.clamp(min=1e-12).log()).sum()
                    - ((1 - tgt) * (1 - tgt).clamp(min=1e-12).log()).sum()
                ),
                "top1_agree": float(int(tgt.argmax() == probs.argmax())),
                "value": value,
            }
        blockers = d.get("blockers", [])
        attackers = d.get("attackers", [])
        if not blockers or not attackers or len(pi) != len(blockers):
            return None
        if any(len(row) != len(attackers) + 1 for row in pi):
            return None
        q = F.softmax(net.block_logits(state, blockers, attackers), -1)
        rows: List[float] = []
        weights: List[float] = []
        agrees: List[float] = []
        for b, row in enumerate(pi):
            p = torch.tensor(row, dtype=torch.float32).clamp(min=0)
            mass = float(p.sum())
            if mass <= 0:
                continue
            p = (p / mass).unsqueeze(0)
            rows.append(float(kl_divergence(p, q[b : b + 1])))
            weights.append(mass)
            agrees.append(float(int(p.argmax() == q[b].argmax())))
        if not rows:
            return None
        total = sum(weights)
        return {
            "kind": t,
            "kl": sum(r * w for r, w in zip(rows, weights)) / total,
            "pi_entropy": None,
            "prior_entropy": None,
            "top1_agree": sum(a * w for a, w in zip(agrees, weights)) / total,
            "value": value,
        }


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _percentile(xs: List[float], frac: float) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, int(frac * len(ordered)))
    return ordered[idx]


def audit_buffer(
    net: ForgePointerNet, path: Path, max_games: int
) -> Dict[str, Any]:
    """Audit one ``jsonl.gz`` buffer file."""
    per_kind: Dict[str, List[Dict[str, Any]]] = {k: [] for k in KINDS}
    values: List[float] = []
    returns: List[float] = []
    games = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            if max_games and games >= max_games:
                break
            game = json.loads(line)
            games += 1
            for d, r in zip(game["decisions"], game["returns"]):
                m = audit_decision(net, d)
                if m is None:
                    continue
                per_kind[m["kind"]].append(m)
                values.append(m["value"])
                returns.append(float(r))
    kinds: Dict[str, Dict[str, Any]] = {}
    for kind, rows in per_kind.items():
        kls = [r["kl"] for r in rows] if kind != "attack" else []
        entry: Dict[str, Any] = {
            "n": len(rows),
            "top1_agree": _mean([r["top1_agree"] for r in rows]),
        }
        if kind == "attack":
            entry["prior_bce"] = _mean([r["prior_bce"] for r in rows])
            entry["abs_diff"] = _mean([r["abs_diff"] for r in rows])
            entry["pi_entropy"] = _mean([r["pi_entropy"] for r in rows])
        else:
            entry["kl_mean"] = _mean(kls)
            entry["kl_median"] = _percentile(kls, 0.5)
            entry["kl_p90"] = _percentile(kls, 0.9)
            entry["pi_entropy"] = _mean(
                [r["pi_entropy"] for r in rows if r["pi_entropy"] is not None]
            )
            entry["prior_entropy"] = _mean(
                [
                    r["prior_entropy"]
                    for r in rows
                    if r["prior_entropy"] is not None
                ]
            )
        kinds[kind] = entry
    value_mse = (
        sum((v - r) ** 2 for v, r in zip(values, returns)) / len(values)
        if values
        else None
    )
    ret_mean = _mean(returns)
    floor = (
        sum((r - ret_mean) ** 2 for r in returns) / len(returns)
        if returns and ret_mean is not None
        else None
    )
    return {
        "file": path.name,
        "games": games,
        "kinds": kinds,
        "value": {
            "n": len(values),
            "corr": pearson(values, returns),
            "mse": value_mse,
            "floor_mse": floor,
            "return_mean": ret_mean,
        },
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(0)
    net, _ = load_pointer_net(str(args.model))
    net.eval()
    run_dir = Path(args.run_dir)
    buffers = sorted((run_dir / "buffer").glob("round_*.jsonl.gz"))
    if not buffers:
        raise SystemExit(f"no round_*.jsonl.gz buffers under {run_dir}/buffer")
    reports = [audit_buffer(net, b, args.max_games) for b in buffers]
    out: Dict[str, Any] = {"model": str(args.model), "buffers": reports}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Champion checkpoint (default <run-dir>/rounds/0000/last.pt)",
    )
    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Games per buffer file (0 = all)",
    )
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.model is None:
        args.model = args.run_dir / "rounds" / "0000" / "last.pt"
    out = run(args)
    print(json.dumps({"buffers": out["buffers"]}, indent=2))


if __name__ == "__main__":
    main()
