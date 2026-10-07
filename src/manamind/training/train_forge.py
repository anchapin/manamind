"""Actor-critic training of ForgePointerNet against Forge AI (#76).

Forge state can't be cloned from Python, so there is no search yet: each
game is played with the sampled policy and every decision is credited with
the final result, with the value head as baseline.

Example (short smoke run)::

    PYTHONPATH=src python -m manamind.training.train_forge \\
        --forge-dir /path/to/forge --java /path/to/java \\
        --deck-a rg.dck --deck-b ub.dck --games 20 --out runs/forge_smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from manamind.forge_interface import ForgeEnv, bridge_command
from manamind.models.forge_pointer import (
    ActOutput,
    ForgePointerNet,
    actor_critic_loss,
)


def play_game(
    env: ForgeEnv, net: ForgePointerNet, greedy: bool = False
) -> Optional[Dict[str, Any]]:
    """Play one game; ``None`` when the bridge has no games left."""
    r = env.reset()
    if env.finished:
        return None
    steps: List[ActOutput] = []
    while not r.done:
        assert r.decision is not None
        out = net.act(r.decision, greedy=greedy)
        steps.append(out)
        r = env.step(out.reply)
    return {"steps": steps, "reward": r.reward, "info": r.info}


def train(
    command: Sequence[str],
    cwd: Optional[Path],
    games: int,
    out_dir: Path,
    lr: float = 3e-4,
    update_every: int = 4,
    seed: int = 0,
    resume: Optional[Path] = None,
    save_every: int = 50,
) -> Dict[str, Any]:
    """Run ``games`` games, updating every ``update_every`` games.

    ``last.pt`` is rewritten every ``save_every`` games and at the end, so
    a long run that dies can be resumed with ``--resume``.
    """
    torch.manual_seed(seed)
    net = ForgePointerNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_game = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["network"])
        opt.load_state_dict(ckpt["optimizer"])
        start_game = int(ckpt["meta"]["games"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    history: List[Dict[str, Any]] = []
    pending: List[torch.Tensor] = []
    t0 = time.time()
    played = 0
    with ForgeEnv(command, cwd=cwd) as env:
        while played < games:
            game = play_game(env, net)
            if game is None:
                break
            played += 1
            rec: Dict[str, Any] = {
                "game": start_game + played,
                "result": game["info"].get("result"),
                "turns": game["info"].get("turns"),
                "decisions": len(game["steps"]),
            }
            if game["steps"]:
                loss, stats = actor_critic_loss(game["steps"], game["reward"])
                pending.append(loss)
                rec.update(stats)
            if pending and (played % update_every == 0 or played == games):
                opt.zero_grad()
                torch.stack(pending).mean().backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                pending = []
                rec["updated"] = True
            rec["secs"] = round(time.time() - t0, 1)
            history.append(rec)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if save_every > 0 and played % save_every == 0:
                _save(net, opt, out_dir, start_game, played, history, t0, env)
    return _save(net, opt, out_dir, start_game, played, history, t0, env)


def _save(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    out_dir: Path,
    start_game: int,
    played: int,
    history: List[Dict[str, Any]],
    t0: float,
    env: ForgeEnv,
) -> Dict[str, Any]:
    meta = {
        "games": start_game + played,
        "wins": sum(h["result"] == "win" for h in history),
        "played_this_run": played,
        "secs": round(time.time() - t0, 1),
        "summary": env.summary,
    }
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
    tmp.replace(out_dir / "last.pt")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forge-dir", type=Path, required=True)
    ap.add_argument(
        "--bridge-out",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "tools"
        / "forge-bridge"
        / "out",
    )
    ap.add_argument("--java", default="java")
    ap.add_argument("--deck-a", type=Path, required=True)
    ap.add_argument("--deck-b", type=Path, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd = bridge_command(
        args.forge_dir,
        args.bridge_out,
        args.games,
        args.deck_a.resolve(),
        args.deck_b.resolve(),
        java=args.java,
    )
    meta = train(
        cmd,
        args.forge_dir,
        args.games,
        args.out,
        lr=args.lr,
        update_every=args.update_every,
        seed=args.seed,
        resume=args.resume,
        save_every=args.save_every,
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
