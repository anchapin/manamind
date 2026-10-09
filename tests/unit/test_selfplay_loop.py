"""Self-play round loop: promote-or-keep, Elo ladder, kill switch (#87)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from manamind.training.selfplay_loop import (
    build_parser,
    elo_diff,
    run,
    table,
)


class Fakes:
    """Stand-ins for the Node scripts and the trainer."""

    def __init__(self, scores: List[float]) -> None:
        self.scores = list(scores)
        self.cmds: List[List[str]] = []
        self.trains: List[List[str]] = []

    def runner(self, cmd: List[str], cwd: Path) -> None:
        self.cmds.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        if "scripts/gate-forge.ts" in cmd:
            games = int(cmd[cmd.index("--games") + 1])
            score = self.scores.pop(0)
            wins = round(score * games)
            out.write_text(
                json.dumps(
                    {
                        "games": games,
                        "wins": wins,
                        "losses": games - wins,
                        "draws": 0,
                        "score": score,
                        "ci95": [max(score - 0.15, 0), min(score + 0.15, 1)],
                        "threshold": 0.55,
                        "promote": score >= 0.55,
                    }
                )
            )
        else:
            out.write_bytes(b"")

    def trainer(self, argv: List[str]) -> Dict[str, Any]:
        self.trains.append(argv)
        return {}


def args(tmp_path: Path, *extra: str) -> Any:
    return build_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--pn-dir", str(tmp_path)]
        + ["--games", "10", "--gate-games", "20", *extra]
    )


def test_elo_diff_is_symmetric_and_clamped() -> None:
    assert elo_diff(0.5, 40) == pytest.approx(0.0)
    assert elo_diff(0.75, 40) == pytest.approx(-elo_diff(0.25, 40))
    assert elo_diff(0.75, 40) == pytest.approx(190.85, abs=0.01)
    assert elo_diff(1.0, 20) == pytest.approx(elo_diff(0.975, 20))
    assert elo_diff(0.6, 0) == 0.0


def test_promote_moves_champion_and_raises_elo(tmp_path: Path) -> None:
    fakes = Fakes([0.75, 0.4])
    state = run(args(tmp_path, "--rounds", "2"), fakes.runner, fakes.trainer)
    run_dir = tmp_path / "run"
    assert (run_dir / "rounds/0000/forge_pointer.onnx").exists()
    assert (run_dir / "rounds/0000/last.pt").exists()
    assert state["round"] == 2 and state["games_total"] == 20
    assert state["champion"] == "rounds/0001"
    assert state["rises"] == 1 and state["flat"] == 1
    assert state["elo"] == pytest.approx(elo_diff(0.75, 20))
    # Round 2 self-plays and trains from the promoted champion.
    sp2 = fakes.cmds[2]
    assert sp2[sp2.index("--model") + 1].endswith(
        "rounds/0001/forge_pointer.onnx"
    )
    assert fakes.trains[1][fakes.trains[1].index("--init") + 1].endswith(
        "rounds/0001/last.pt"
    )
    # Fresh seeds every round: self-play 1..10, 11..20; gate blocks differ.
    seeds = [c[c.index("--seed") + 1] for c in fakes.cmds]
    assert seeds == ["1", "1000001", "11", "1000021"]
    assert "| 1 | 10 | 0.750" in (run_dir / "loop.md").read_text()


def test_kill_switch_after_three_flat_rounds_and_resume(
    tmp_path: Path,
) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.5, 0.4, 0.3])
    state = run(args(tmp_path, "--rounds", "10"), fakes.runner, fakes.trainer)
    assert state["round"] == 6
    assert state["stopped"] == "Elo flat for 3 rounds"
    assert [h["promote"] for h in state["history"]] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    # A new dispatch resumes from loop.json and stays stopped...
    again = run(args(tmp_path, "--rounds", "2"), fakes.runner, fakes.trainer)
    assert again["round"] == 6 and len(fakes.cmds) == 12
    # ...unless forced.
    fakes.scores = [0.7]
    forced = run(
        args(tmp_path, "--rounds", "1", "--force"),
        fakes.runner,
        fakes.trainer,
    )
    assert forced["round"] == 7 and forced["stopped"] is None
    assert "Elo rose in 2 round(s)" in table(forced)


def test_buffer_cap_and_gate_settings_reach_the_scripts(
    tmp_path: Path,
) -> None:
    fakes = Fakes([0.5])
    run(
        args(
            tmp_path,
            "--buffer-games",
            "123",
            "--threshold",
            "0.6",
            "--sims",
            "32",
        ),
        fakes.runner,
        fakes.trainer,
    )
    train = fakes.trains[0]
    assert train[train.index("--buffer-games") + 1] == "123"
    assert "--onnx" in train
    gate = fakes.cmds[1]
    assert gate[gate.index("--threshold") + 1] == "0.6"
    assert gate[gate.index("--sims") + 1] == "32"
