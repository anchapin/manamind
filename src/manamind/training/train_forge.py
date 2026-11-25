"""Actor-critic training of ForgePointerNet against Forge AI (#76).

Forge state can't be cloned from Python, so there is no search yet: each
game is played with the sampled policy and every decision is credited with
the final result, with the value head as baseline.

``--expert-games N`` warm-starts the policy by imitation (#79): the first
N games are played by Forge AI in the piped seat (the bridge's expert
mode), the network is trained to reproduce its choices, and the value head
to predict those games' results. ``--bc-epochs`` then makes extra passes
over every recorded decision before actor-critic training takes over.

``--deck-pairs`` spreads those imitation games over several deck pairs
("deckA,deckB deckC,deckD ..."), one bridge block per pair, so the BC root
learns from more than one matchup (#96 data axis). Actor-critic games keep
playing ``--deck-a`` vs ``--deck-b``.

``--eval-ckpts a.pt,b.pt`` skips training and plays ``--games`` games per
checkpoint and mode (``--eval-modes greedy,sample``) with no updates, so a
post-imitation ``bc.pt`` can be scored against the final ``last.pt``.
Relative paths resolve against the parent of ``--out``.

Example (short smoke run)::

    PYTHONPATH=src python -m manamind.training.train_forge \\
        --forge-dir /path/to/forge --java /path/to/java \\
        --deck-a rg.dck --deck-b ub.dck --games 20 --out runs/forge_smoke
"""

from __future__ import annotations

import argparse
import array
import ctypes
import gc
import gzip
import json
import random
import resource
import shutil
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence

import torch

from manamind.forge_interface import EXPERT_REPLY, ForgeEnv, bridge_command
from manamind.models.forge_pointer import (
    ActOutput,
    ForgePointerNet,
    actor_critic_loss,
    imitation_loss,
    life_potential,
    shaped_returns,
)

EXPERT_DATA = "expert_data.jsonl.gz"


def _trim_heap(tag: str) -> None:
    """Release freed allocator arenas and log peak RSS (#152).

    A long-lived training process churns millions of tensors; CPython and
    glibc keep fragmented arenas alive long after the tensors are freed,
    so RSS sits well above live data. ``malloc_trim`` returns those pages
    before a fresh multi-GB allocation (the BC corpus load) needs them.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # pragma: no cover - non-glibc or non-Linux hosts
        pass
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # pragma: no cover - platforms without resource
        rss_kb = 0
    print(f"RSS[{tag}]: maxrss={rss_kb // 1024} MB", flush=True)


def play_game(
    env: ForgeEnv, net: ForgePointerNet, greedy: bool = False
) -> Optional[Dict[str, Any]]:
    """Play one game; ``None`` when the bridge has no games left."""
    r = env.reset()
    if env.finished:
        return None
    steps: List[ActOutput] = []
    potentials: List[float] = []
    while not r.done:
        assert r.decision is not None
        out = net.act(r.decision, greedy=greedy)
        steps.append(out)
        potentials.append(life_potential(r.decision))
        r = env.step(out.reply)
    return {
        "steps": steps,
        "potentials": potentials,
        "reward": r.reward,
        "info": r.info,
    }


def play_expert_game(env: ForgeEnv) -> Optional[Dict[str, Any]]:
    """Let Forge AI play the piped seat; record its labelled decisions."""
    r = env.reset()
    if env.finished:
        return None
    decisions: List[Dict[str, Any]] = []
    while not r.done:
        assert r.decision is not None
        decisions.append(r.decision)
        r = env.step(EXPERT_REPLY)
    return {"decisions": decisions, "reward": r.reward, "info": r.info}


def _game_returns(
    decisions: Sequence[Dict[str, Any]],
    reward: float,
    gamma: float,
    shaping_coef: float,
) -> List[float]:
    pots = [life_potential(d) for d in decisions]
    return shaped_returns(reward, pots, gamma, shaping_coef)


def load_expert_data(path: Path) -> List[Dict[str, Any]]:
    """Games written by the imitation phase, oldest first."""
    if not path.exists():
        return []
    games = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


class ExpertCorpus(Sequence[Dict[str, Any]]):
    """Re-iterable, lazy view of the imitation corpus (#96 memory).

    Decompresses the gz corpus once into a plain jsonl sibling (an
    atomic replace, so an interrupted decompress never leaves a
    partial file), indexes each game's byte offset, and parses a game
    only when indexed. Neither the multi-GB parsed-decision form nor
    the multi-GB line cache that earlyoom killed mid-load (#152) sits
    in RAM: the index is 8 bytes per game, and the plain file is read
    by seek on every access.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._plain = path.with_suffix("") if path.suffix == ".gz" else path
        self._offsets: array.array[int] = array.array("q")
        self._fh: Optional[IO[bytes]] = None
        if path.exists() and not self._plain.exists():
            tmp = self._plain.parent / (self._plain.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            with gzip.open(path, "rb") as src, tmp.open("wb") as dst:
                for line in src:
                    if line.strip():
                        self._offsets.append(dst.tell())
                        dst.write(line)
            tmp.replace(self._plain)
        elif self._plain.exists():
            with self._plain.open("rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self._offsets.append(offset)
        if self._offsets:
            self._fh = self._plain.open("rb")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: Any) -> Any:
        if self._fh is None:
            raise IndexError(index)
        self._fh.seek(self._offsets[index])
        return json.loads(self._fh.readline())


def bc_epochs(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    games: Sequence[Dict[str, Any]],
    epochs: int,
    batch: int = 256,
    seed: int = 0,
    log: Optional[Path] = None,
    val_chunk: int = 4096,
    buffer_rows: int = 1 << 17,
) -> List[Dict[str, Any]]:
    """Extra behaviour-cloning passes; every 10th game is held out.

    Rows stream through a bounded shuffle buffer and games are parsed
    one index at a time, so an :class:`ExpertCorpus` never holds the
    corpus as parsed Python objects (the 16k-game corpus killed runs
    44825300109 and 44986357644 as silent exit-1s mid-BC).
    """
    n = len(games)
    train_idx = [i for i in range(n) if i % 10 != 9]
    val_idx = [i for i in range(n) if i % 10 == 9]
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for ep in range(1, epochs + 1):
        print(f"BC: epoch {ep}/{epochs} start", flush=True)
        net.train()
        order = list(train_idx)
        rng.shuffle(order)
        buf: List[tuple] = []
        tr: List[float] = []
        seen = 0

        def drain() -> None:
            nonlocal seen
            if not buf:
                return
            rng.shuffle(buf)
            for j in range(0, len(buf), batch):
                rows = buf[j : j + batch]
                loss, st = imitation_loss(
                    [net.imitate(d) for d, _ in rows],
                    [r for _, r in rows],
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tr.append(st["bc_nll"])
            seen += len(buf)
            buf.clear()
            print(f"BC: epoch {ep} trained {seen} rows", flush=True)

        for i in order:
            g = games[i]
            buf.extend(zip(g["decisions"], g["returns"]))
            if len(buf) >= buffer_rows:
                drain()
        drain()
        rec: Dict[str, Any] = {
            "bc_epoch": ep,
            "train_rows": seen,
            "train_nll": round(sum(tr) / max(len(tr), 1), 4),
        }
        if val_idx:
            n_nll = n_acc = n_value = 0.0
            n_used = n_rows = 0

            def eval_chunk(rows: List[tuple]) -> None:
                nonlocal n_nll, n_acc, n_value, n_used, n_rows
                _, vs = imitation_loss(
                    [net.imitate(d) for d, _ in rows],
                    [r for _, r in rows],
                )
                used = round(vs["labelled_frac"] * len(rows))
                n_rows += len(rows)
                n_used += used
                n_value += vs["value"] * len(rows)
                n_nll += vs["bc_nll"] * used
                n_acc += vs["bc_acc"] * used

            chunk: List[tuple] = []
            with torch.no_grad():
                for i in val_idx:
                    g = games[i]
                    chunk.extend(zip(g["decisions"], g["returns"]))
                    if len(chunk) >= val_chunk:
                        eval_chunk(chunk)
                        chunk = []
                if chunk:
                    eval_chunk(chunk)
            rec.update(
                val_rows=n_rows,
                val_nll=round(n_nll / n_used, 4) if n_used else 0.0,
                val_acc=round(n_acc / n_used, 4) if n_used else 0.0,
                val_value=round(n_value / n_rows, 4),
            )
        out.append(rec)
        if log is not None:
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"BC: epoch {ep} done rec={rec}", flush=True)
    return out


_DEFAULT_NET_DIMS = {"card_dim": 64, "state_dim": 128}


def net_size_kwargs(
    card_dim: Optional[int], state_dim: Optional[int]
) -> Dict[str, int]:
    """Build width overrides for a fresh ``ForgePointerNet``.

    ``None`` keeps the constructor default (card_dim 64, state_dim 128);
    a value below 1 is rejected so a bad flag cannot train a degenerate net.
    """
    kwargs: Dict[str, int] = {}
    for name, value in (("card_dim", card_dim), ("state_dim", state_dim)):
        if value is None:
            continue
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
        kwargs[name] = value
    return kwargs


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
    gamma: float = 1.0,
    shaping_coef: float = 0.0,
    entropy_coef: float = 0.01,
    expert_command: Optional[Sequence[str]] = None,
    expert_commands: Optional[Sequence[Sequence[str]]] = None,
    expert_games: int = 0,
    bc_epoch_count: int = 0,
    bc_batch: int = 256,
    card_dim: Optional[int] = None,
    state_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``games`` games, updating every ``update_every`` games.

    Games numbered below ``expert_games`` (counting resumed ones) are
    imitation games over ``expert_command``, or cycled over
    ``expert_commands`` one bridge block per pair when given; the rest are
    actor-critic.
    ``last.pt`` is rewritten every ``save_every`` games and at the end, so
    a long run that dies can be resumed with ``--resume``.
    """
    torch.manual_seed(seed)
    net_kwargs = net_size_kwargs(card_dim, state_dim)
    net = ForgePointerNet(
        card_dim=int(
            net_kwargs.get("card_dim", _DEFAULT_NET_DIMS["card_dim"])
        ),
        state_dim=int(
            net_kwargs.get("state_dim", _DEFAULT_NET_DIMS["state_dim"])
        ),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_game = 0
    bc_done = False
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        for name in ("card_dim", "state_dim"):
            want = net_kwargs.get(name, _DEFAULT_NET_DIMS[name])
            have = int(cfg.get(name, _DEFAULT_NET_DIMS[name]))
            if want != have:
                raise ValueError(
                    f"--{name.replace('_', '-')} {want} does not match the"
                    f" --resume checkpoint ({have})"
                )
        net.load_state_dict(ckpt["network"])
        opt.load_state_dict(ckpt["optimizer"])
        start_game = int(ckpt["meta"]["games"])
        bc_done = bool(ckpt["meta"].get("bc_done", False))
        print(
            f"RESUME: start_game={start_game} bc_done={bc_done}",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    history: List[Dict[str, Any]] = []
    pending: List[torch.Tensor] = []
    t0 = time.time()
    played = 0
    state = {"bc_done": bc_done}
    data_path = out_dir / EXPERT_DATA
    expert_summary: Optional[Dict[str, Any]] = None
    if expert_games > start_game:
        commands: List[Sequence[str]] = []
        if expert_commands:
            commands = list(expert_commands)
        elif expert_command is not None:
            commands = [expert_command]
        if not commands:
            raise ValueError("expert_games needs expert_command")
        # One bridge block per deck pair; each block gets an even share of
        # the imitation games (the last block absorbs the remainder).
        block = max(1, -(-expert_games // len(commands)))
        env: Optional[ForgeEnv] = None
        active: Optional[int] = None
        try:
            while played < games and start_game + played < expert_games:
                pair = min(played // block, len(commands) - 1)
                if pair != active:
                    if env is not None:
                        if env.summary:
                            expert_summary = env.summary
                        env.close()
                    env = ForgeEnv(commands[pair], cwd=cwd)
                    active = pair
                assert env is not None
                eg = play_expert_game(env)
                if eg is None:
                    break
                played += 1
                rets = _game_returns(
                    eg["decisions"], eg["reward"], gamma, shaping_coef
                )
                with gzip.open(data_path, "at") as f:
                    f.write(
                        json.dumps(
                            {
                                "decisions": eg["decisions"],
                                "returns": rets,
                                "result": eg["info"].get("result"),
                            }
                        )
                        + "\n"
                    )
                rec: Dict[str, Any] = {
                    "game": start_game + played,
                    "phase": "imitate",
                    "pair": pair,
                    "result": eg["info"].get("result"),
                    "turns": eg["info"].get("turns"),
                    "decisions": len(eg["decisions"]),
                }
                if eg["decisions"]:
                    loss, stats = imitation_loss(
                        [net.imitate(d) for d in eg["decisions"]], rets
                    )
                    pending.append(loss)
                    rec.update(stats)
                if pending and (
                    played % update_every == 0
                    or start_game + played == expert_games
                ):
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
                    _save(
                        net,
                        opt,
                        out_dir,
                        start_game,
                        played,
                        history,
                        t0,
                        env,
                        state,
                    )
        finally:
            if env is not None:
                if env.summary:
                    expert_summary = env.summary
                env.close()
    if (
        expert_games > 0
        and start_game + played >= expert_games
        and not state["bc_done"]
    ):
        # Outside the ``expert_games > start_game`` guard so a resume whose
        # checkpoint sits exactly at ``expert_games`` still runs the BC
        # passes (run 44939730233 skipped them and burned the RL budget).
        if bc_epoch_count > 0:
            _trim_heap("pre-bc")
            bc_epochs(
                net,
                opt,
                ExpertCorpus(data_path),
                bc_epoch_count,
                batch=bc_batch,
                seed=seed,
                log=log_path,
            )
        state["bc_done"] = True
        _save(
            net,
            opt,
            out_dir,
            start_game,
            played,
            history,
            t0,
            None,
            state,
            expert_summary,
        )
        shutil.copyfile(out_dir / "last.pt", out_dir / "bc.pt")
    if played >= games:
        return _save(
            net, opt, out_dir, start_game, played, history, t0, None, state
        )
    with ForgeEnv(command, cwd=cwd) as env:
        while played < games:
            game = play_game(env, net)
            if game is None:
                break
            played += 1
            rec = {
                "game": start_game + played,
                "phase": "rl",
                "result": game["info"].get("result"),
                "turns": game["info"].get("turns"),
                "decisions": len(game["steps"]),
            }
            if game["steps"]:
                loss, stats = actor_critic_loss(
                    game["steps"],
                    game["reward"],
                    entropy_coef=entropy_coef,
                    potentials=game["potentials"],
                    gamma=gamma,
                    shaping_coef=shaping_coef,
                )
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
                _save(
                    net,
                    opt,
                    out_dir,
                    start_game,
                    played,
                    history,
                    t0,
                    env,
                    state,
                )
        return _save(
            net, opt, out_dir, start_game, played, history, t0, env, state
        )


def evaluate(
    command: Sequence[str],
    cwd: Optional[Path],
    ckpts: Sequence[Path],
    games: int,
    out_dir: Path,
    modes: Sequence[str] = ("greedy", "sample"),
    seed: int = 0,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score each checkpoint over ``games`` games per mode; no training."""
    for mode in modes:
        if mode not in ("greedy", "sample"):
            raise ValueError(f"unknown eval mode {mode!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    names = list(labels) if labels is not None else [str(c) for c in ckpts]
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for ckpt_path, name in zip(ckpts, names):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        net = ForgePointerNet(
            card_dim=int(cfg.get("card_dim", 64)),
            state_dim=int(cfg.get("state_dim", 128)),
        )
        net.load_state_dict(ckpt["network"])
        net.eval()
        for mode in modes:
            torch.manual_seed(seed)
            wins = played = turns = 0
            with torch.no_grad(), ForgeEnv(command, cwd=cwd) as env:
                while played < games:
                    game = play_game(env, net, greedy=mode == "greedy")
                    if game is None:
                        break
                    played += 1
                    result = game["info"].get("result")
                    wins += result == "win"
                    turns += int(game["info"].get("turns") or 0)
                    rec = {
                        "game": played,
                        "phase": "eval",
                        "ckpt": name,
                        "mode": mode,
                        "result": result,
                        "turns": game["info"].get("turns"),
                        "decisions": len(game["steps"]),
                        "secs": round(time.time() - t0, 1),
                    }
                    with log_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
            results.append(
                {
                    "ckpt": name,
                    "trained_games": ckpt.get("meta", {}).get("games"),
                    "mode": mode,
                    "games": played,
                    "wins": wins,
                    "win_rate": round(wins / played, 4) if played else None,
                    "avg_turns": round(turns / played, 1) if played else None,
                }
            )
    summary = {"eval": results, "secs": round(time.time() - t0, 1)}
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _parse_deck_pairs(spec: str) -> List[tuple[Path, Path]]:
    """Parse "deckA,deckB deckC,deckD" into deck pairs (#96 data axis)."""
    pairs: List[tuple[Path, Path]] = []
    for chunk in spec.split():
        parts = chunk.split(",")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"--deck-pairs entry {chunk!r} is not deckA,deckB"
            )
        pairs.append((Path(parts[0]), Path(parts[1])))
    if not pairs:
        raise ValueError("--deck-pairs is empty")
    return pairs


def _save(
    net: ForgePointerNet,
    opt: torch.optim.Optimizer,
    out_dir: Path,
    start_game: int,
    played: int,
    history: List[Dict[str, Any]],
    t0: float,
    env: Optional[ForgeEnv],
    state: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rl = [h for h in history if h.get("phase") != "imitate"]
    meta = {
        "games": start_game + played,
        "wins": sum(h.get("result") == "win" for h in rl),
        "rl_games_this_run": len(rl),
        "played_this_run": played,
        "secs": round(time.time() - t0, 1),
        "summary": env.summary if env is not None else summary,
        "bc_done": bool((state or {}).get("bc_done", False)),
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
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            f"SAVE: games={meta['games']} maxrss={rss_kb // 1024} MB",
            flush=True,
        )
    except Exception:  # pragma: no cover - platforms without resource
        pass
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
    ap.add_argument(
        "--deck-pairs",
        nargs="+",
        default="",
        help='imitate across deck pairs: space-separated "deckA,deckB"'
        " pairs, one bridge block each (#96 data axis)",
    )
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="per-decision discount (1.0 = plain game result, #79)",
    )
    ap.add_argument(
        "--shaping",
        type=float,
        default=0.0,
        help="weight on life-lead change between decisions (#79)",
    )
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument(
        "--expert-games",
        type=int,
        default=0,
        help="first N games: Forge AI plays our seat and we imitate it (#79)",
    )
    ap.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="extra imitation passes over recorded decisions after them",
    )
    ap.add_argument("--bc-batch", type=int, default=256)
    ap.add_argument(
        "--card-dim",
        type=int,
        default=None,
        help="card encoder width for a fresh net (default 64)",
    )
    ap.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="state encoder width for a fresh net (default 128)",
    )
    ap.add_argument(
        "--eval-ckpts",
        help="comma list of checkpoints to score instead of training; "
        "relative paths resolve against the parent of --out",
    )
    ap.add_argument("--eval-modes", default="greedy,sample")
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
    if args.eval_ckpts:
        names = [c for c in args.eval_ckpts.split(",") if c]
        paths = [
            Path(c) if Path(c).is_absolute() else args.out.parent / c
            for c in names
        ]
        print(
            json.dumps(
                evaluate(
                    cmd,
                    args.forge_dir,
                    paths,
                    args.games,
                    args.out,
                    modes=[m for m in args.eval_modes.split(",") if m],
                    seed=args.seed,
                    labels=names,
                )
            )
        )
        return
    expert_cmd = bridge_command(
        args.forge_dir,
        args.bridge_out,
        args.games,
        args.deck_a.resolve(),
        args.deck_b.resolve(),
        java=args.java,
        expert=True,
    )
    deck_pairs = (
        _parse_deck_pairs(
            " ".join(args.deck_pairs)
            if isinstance(args.deck_pairs, list)
            else args.deck_pairs
        )
        if args.deck_pairs
        else None
    )
    expert_cmds: Optional[List[List[str]]] = None
    if deck_pairs:
        if args.expert_games <= 0:
            raise SystemExit("--deck-pairs needs --expert-games > 0")
        expert_cmds = [
            bridge_command(
                args.forge_dir,
                args.bridge_out,
                args.games,
                a.resolve(),
                b.resolve(),
                java=args.java,
                expert=True,
            )
            for a, b in deck_pairs
        ]
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
        gamma=args.gamma,
        shaping_coef=args.shaping,
        entropy_coef=args.entropy_coef,
        expert_command=expert_cmd,
        expert_commands=expert_cmds,
        expert_games=args.expert_games,
        bc_epoch_count=args.bc_epochs,
        bc_batch=args.bc_batch,
        card_dim=args.card_dim,
        state_dim=args.state_dim,
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
