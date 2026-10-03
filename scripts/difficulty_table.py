"""Strength and latency per difficulty setting (#46).

Exports a checkpoint to ONNX, then plays each setting against two fixed
opponents: a random agent, and the same checkpoint at full strength
(Gumbel search, greedy). Policy-only settings play through the exported
model via ``onnx_host``; search settings use PyTorch ``MCTSAgent``,
since search hasn't been ported out of Python yet. Each setting reports
its score with a 95% CI and milliseconds per real decision (moves with
more than one legal option).

    python scripts/difficulty_table.py CHECKPOINT [--games 40]
        [--ref-simulations 40] [--settings SPEC,...] [--out table.json]

A setting spec is ``sims=N``, ``blunder=P`` and ``temp=T`` joined by ``/``,
e.g. ``sims=1/blunder=0.25`` or ``temp=1.0``; ``policy`` is greedy
policy-only. ``--settings ladder`` runs the sims ladder from LADDER.
"""

import argparse
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_onnx  # noqa: E402
import onnx_host  # noqa: E402
from train_simple import (  # noqa: E402
    MCTSAgent,
    known_deck_lists,
    load_checkpoint,
    play_game,
    wilson_interval,
)

from manamind.core.action import ActionSpace  # noqa: E402
from manamind.core.agent import RandomAgent  # noqa: E402


@dataclass(frozen=True)
class Setting:
    name: str
    temperature: float = 0.0
    blunder_rate: float = 0.0
    simulations: int = 0  # 0 = policy only, through ONNX

    @property
    def kind(self) -> str:
        return "policy (onnx)" if self.simulations == 0 else "search (torch)"


DEFAULT_SETTINGS: List[Setting] = [
    Setting("policy, blunder 25%", blunder_rate=0.25),
    Setting("policy, blunder 10%", blunder_rate=0.10),
    Setting("policy, sampled (T=1)", temperature=1.0),
    Setting("policy, greedy"),
    Setting("search, 10 sims", simulations=10),
    Setting("search, 40 sims", simulations=40),
]


LADDER: List[Setting] = [
    Setting("search, 1 sim, blunder 25%", blunder_rate=0.25, simulations=1),
    Setting("search, 1 sim", simulations=1),
    Setting("search, 2 sims", simulations=2),
    Setting("search, 5 sims", simulations=5),
    Setting("search, 10 sims", simulations=10),
    Setting("search, 20 sims", simulations=20),
]


def parse_setting(spec: str) -> Setting:
    """Parse ``sims=5/blunder=0.1/temp=1.0`` (any subset) or ``policy``."""
    values: Dict[str, float] = {"sims": 0, "blunder": 0.0, "temp": 0.0}
    if spec.strip() != "policy":
        for part in spec.split("/"):
            key, sep, raw = part.partition("=")
            key = key.strip()
            if not sep or key not in values:
                raise ValueError(f"bad setting part {part!r} in {spec!r}")
            values[key] = float(raw)
    sims = int(values["sims"])
    if sims < 0 or not 0.0 <= values["blunder"] <= 1.0 or values["temp"] < 0:
        raise ValueError(f"out-of-range setting {spec!r}")
    kind = (
        f"search, {sims} sim{'s' if sims != 1 else ''}" if sims else "policy"
    )
    extras = []
    if values["blunder"]:
        extras.append(f"blunder {values['blunder']:.0%}")
    if values["temp"]:
        extras.append(f"T={values['temp']:g}")
    name = (
        ", ".join([kind] + extras)
        if extras
        else (kind if sims else "policy, greedy")
    )
    return Setting(name, values["temp"], values["blunder"], sims)


def parse_settings(arg: Optional[str]) -> List[Setting]:
    if not arg:
        return DEFAULT_SETTINGS
    if arg.strip() == "ladder":
        return LADDER
    return [parse_setting(s) for s in arg.split(",") if s.strip()]


class Timed:
    """Times an agent's real decisions (more than one legal move)."""

    def __init__(self, agent, player_id: int):
        self.agent = agent
        self.player_id = player_id
        self._space = ActionSpace()
        self.decisions = 0
        self.seconds = 0.0

    def select_action(self, state):
        if len(self._space.get_legal_actions(state)) <= 1:
            return self.agent.select_action(state)
        start = time.perf_counter()
        action = self.agent.select_action(state)
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return action


def make_agent(setting: Setting, pid: int, net, model_dir: Path, seed: int):
    if setting.simulations == 0:
        return onnx_host.OnnxPolicyAgent(
            model_dir,
            player_id=pid,
            temperature=setting.temperature,
            blunder_rate=setting.blunder_rate,
            seed=seed,
        )
    agent = MCTSAgent(
        player_id=pid,
        policy_network=net,
        value_network=net,
        simulations=setting.simulations,
        simulation_time=30.0,
        deck_lists=known_deck_lists(),
        search="gumbel",
    )
    if setting.blunder_rate > 0:
        return Blundering(agent, setting.blunder_rate, seed)
    return agent


class Blundering:
    """Replaces a search agent's move with a random legal one at a rate."""

    def __init__(self, agent, rate: float, seed: int):
        self.agent = agent
        self.rate = rate
        self._rng = random.Random(seed)
        self._space = ActionSpace()

    def select_action(self, state):
        legal = self._space.get_legal_actions(state)
        if len(legal) > 1 and self._rng.random() < self.rate:
            return self._rng.choice(legal)
        return self.agent.select_action(state)


def score_setting(
    setting: Setting,
    opponent: Callable[[int, int], object],
    net,
    model_dir: Path,
    games: int,
    seed: int,
) -> Dict[str, float]:
    points = 0.0
    decisions = 0
    seconds = 0.0
    for g in range(games):
        seat = g % 2
        timed = Timed(
            make_agent(setting, seat, net, model_dir, seed + g), seat
        )
        agents = {seat: timed, 1 - seat: opponent(1 - seat, seed + g)}
        # Each deal is played twice, once from each seat.
        winner, _, _ = play_game(agents, seed=seed * 1000 + g // 2)
        points += 0.5 if winner is None else float(winner == seat)
        decisions += timed.decisions
        seconds += timed.seconds
    score = points / games
    lo, hi = wilson_interval(score, games)
    return {
        "score": score,
        "ci_low": lo,
        "ci_high": hi,
        "ms_per_decision": 1000 * seconds / decisions if decisions else 0.0,
    }


def build_table(
    checkpoint: Path,
    games: int = 40,
    ref_simulations: int = 40,
    settings: Optional[List[Setting]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    net = load_checkpoint(checkpoint)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        export_onnx.export(net, model_dir)

        def random_opp(pid: int, s: int):
            return RandomAgent(pid, seed=s)

        def full_strength(pid: int, s: int):
            return make_agent(
                Setting("ref", simulations=ref_simulations),
                pid,
                net,
                model_dir,
                s,
            )

        for setting in settings or DEFAULT_SETTINGS:
            vs_random = score_setting(
                setting, random_opp, net, model_dir, games, seed
            )
            vs_full = score_setting(
                setting, full_strength, net, model_dir, games, seed + 7
            )
            row = {
                "setting": setting.name,
                "kind": setting.kind,
                "temperature": setting.temperature,
                "blunder_rate": setting.blunder_rate,
                "simulations": setting.simulations,
                "vs_random": vs_random,
                "vs_full_strength": vs_full,
            }
            rows.append(row)
            print(format_row(row), flush=True)
    return {
        "checkpoint": str(checkpoint),
        "games_per_cell": games,
        "full_strength": f"gumbel search, {ref_simulations} sims, greedy",
        "rows": rows,
    }


def format_row(row: Dict) -> str:
    r, f = row["vs_random"], row["vs_full_strength"]
    ms = max(r["ms_per_decision"], f["ms_per_decision"])
    return (
        f"{row['setting']:<24} vs random {r['score']:.3f} "
        f"({r['ci_low']:.2f}-{r['ci_high']:.2f})  vs full "
        f"{f['score']:.3f} ({f['ci_low']:.2f}-{f['ci_high']:.2f})  "
        f"{ms:.2f} ms/decision"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--ref-simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settings",
        default=None,
        help="comma-separated specs, or 'ladder' (default: built-in set)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.games % 2:
        parser.error("--games must be even (each deal is played twice)")
    try:
        settings = parse_settings(args.settings)
    except ValueError as exc:
        parser.error(str(exc))
    table = build_table(
        args.checkpoint,
        args.games,
        args.ref_simulations,
        settings=settings,
        seed=args.seed,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
