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

from manamind.models.forge_pointer import (
    ForgePointerNet,
    distill_loss,
    load_pointer_net,
)

BUFFER_GAMES = 20_000
"""Replay buffer cap from #87: the newest this many games."""


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


def train_epochs(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    samples: Sequence[Tuple[Dict[str, Any], float]],
    epochs: int = 1,
    batch: int = 64,
    value_coef: float = 1.0,
    seed: int = 0,
    log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Minibatch passes over ``samples``; returns the last epoch's means."""
    if not samples:
        raise ValueError("no self-play decisions to train on")
    rng = random.Random(seed)
    order = list(range(len(samples)))
    net.train()
    means: Dict[str, float] = {}
    for epoch in range(epochs):
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
    if args.init is not None:
        net, prev = load_pointer_net(str(args.init))
        parent_games = int((prev or {}).get("selfplay_games_total", 0))
    else:
        net, parent_games = ForgePointerNet(), 0
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    history: List[Dict[str, Any]] = []
    stats = train_epochs(
        net,
        opt,
        samples,
        epochs=args.epochs,
        batch=args.batch,
        value_coef=args.value_coef,
        seed=args.seed,
        log=history,
    )
    meta: Dict[str, Any] = {
        "phase": "selfplay",
        "init": str(args.init) if args.init is not None else None,
        "buffer_games": len(games),
        "decisions": len(samples),
        "selfplay_games_total": parent_games + len(games),
        "epochs": args.epochs,
        "secs": round(time.time() - t0, 1),
        "history": history,
        **{f"final_{k}": round(v, 5) for k, v in stats.items()},
    }
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
