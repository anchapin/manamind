"""Self-play round loop: promote-or-keep, Elo ladder, kill switch (#87)."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from manamind.training.selfplay_loop import (
    build_parser,
    elo_diff,
    run,
    table,
)


class Fakes:
    """Stand-ins for the Node scripts and the trainer."""

    def __init__(
        self, scores: List[float], experts: Optional[List[float]] = None
    ) -> None:
        self.scores = list(scores)
        self.experts = list(experts or [])
        self.fail_with_results = False
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
        elif "scripts/yardstick-pn.ts" in cmd:
            games = int(cmd[cmd.index("--games") + 1])
            score = self.experts.pop(0)
            win = round(score * games)
            out.write_text(
                json.dumps(
                    {
                        "games": games,
                        "win": win,
                        "loss": games - win,
                        "draw": 0,
                        "score": score,
                        "ci95": [0.0, 1.0],
                    }
                )
            )
            if self.fail_with_results:
                raise subprocess.CalledProcessError(1, cmd)
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


def test_anchor_is_off_by_default(tmp_path: Path) -> None:
    fakes = Fakes([0.75])
    state = run(args(tmp_path), fakes.runner, fakes.trainer)
    assert not any("scripts/yardstick-pn.ts" in c for c in fakes.cmds)
    assert "expert" not in state["history"][0]
    assert "Expert" not in table(state)


def test_anchor_vetoes_a_promotion_that_loses_to_the_expert(
    tmp_path: Path,
) -> None:
    # Gate promotes rounds 1-3 and keeps round 4. Expert: champion 0.30,
    # round 1 0.34 (new best), round 2 0.30 (within 0.05 of 0.34), round
    # 3 0.25 (below 0.29, vetoed); round 4 never reaches the anchor.
    fakes = Fakes([0.75, 0.6, 0.7, 0.4], [0.30, 0.34, 0.30, 0.25])
    state = run(
        args(tmp_path, "--rounds", "4", "--anchor-games", "20"),
        fakes.runner,
        fakes.trainer,
    )
    assert [h["promote"] for h in state["history"]] == [
        True,
        True,
        False,
        False,
    ]
    assert [h["vetoed"] for h in state["history"]] == [
        False,
        False,
        True,
        False,
    ]
    assert [h["expert"] for h in state["history"]] == [
        0.34,
        0.30,
        0.25,
        None,
    ]
    assert state["champion"] == "rounds/0002"
    assert state["anchor_best"] == pytest.approx(0.34)
    assert state["rises"] == 2 and state["flat"] == 2
    assert state["elo"] == pytest.approx(
        elo_diff(0.75, 20) + elo_diff(0.6, 20)
    )
    anchors = [c for c in fakes.cmds if "scripts/yardstick-pn.ts" in c]
    # Champion measured once, then one run per gated candidate.
    models = [c[c.index("--model") + 1] for c in anchors]
    assert [Path(m).parent.name for m in models] == [
        "0000",
        "0001",
        "0002",
        "0003",
    ]
    # Same deals every time, greedy at the loop's search budget.
    assert {c[c.index("--seed") + 1] for c in anchors} == {"2000001"}
    assert all(c[c.index("--games") + 1] == "20" for c in anchors)
    assert all("--sample" not in c for c in anchors)
    md = table(state)
    assert "| Expert |" in md.splitlines()[0] or " Expert |" in md
    assert "kept champion (Expert regression)" in md
    assert "| 4 | 40 |" in md and md.splitlines()[5].endswith(" - |")
    assert "Best champion Expert score 0.340." in md


def test_anchor_keeps_results_when_some_games_error(tmp_path: Path) -> None:
    fakes = Fakes([0.75], [0.30, 0.40])
    fakes.fail_with_results = True
    state = run(
        args(tmp_path, "--anchor-games", "20"), fakes.runner, fakes.trainer
    )
    assert state["history"][0]["promote"] is True
    assert state["anchor_best"] == pytest.approx(0.40)


def test_rerun_round_drops_a_stale_candidate_anchor(tmp_path: Path) -> None:
    stale = tmp_path / "run" / "rounds" / "0001" / "anchor.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"games": 20, "score": 0.99}))
    fakes = Fakes([0.75], [0.30, 0.10])
    state = run(
        args(tmp_path, "--anchor-games", "20"), fakes.runner, fakes.trainer
    )
    assert state["history"][0]["expert"] == pytest.approx(0.10)
    assert state["history"][0]["vetoed"] is True


def test_expert_games_and_decks_reach_the_scripts(tmp_path: Path) -> None:
    a = args(
        tmp_path,
        "--expert-games",
        "100",
        "--deck-a",
        "red",
        "--deck-b",
        "green",
    )
    fakes = Fakes([0.5])
    run(a, fakes.runner, fakes.trainer)
    plays = [c for c in fakes.cmds if "scripts/selfplay-forge.ts" in c]
    assert len(plays) == 2
    expert = plays[1]
    assert expert[expert.index("--opponent") + 1] == "expert"
    assert expert[expert.index("--games") + 1] == "100"
    assert expert[expert.index("--seed") + 1] == "3000001"
    assert expert[expert.index("--out") + 1].endswith(
        "round_0001_expert.jsonl.gz"
    )
    for cmd in fakes.cmds:
        assert cmd[cmd.index("--deck-a") + 1] == "red"
        assert cmd[cmd.index("--deck-b") + 1] == "green"


def test_expert_games_and_decks_are_off_by_default(tmp_path: Path) -> None:
    fakes = Fakes([0.5])
    run(args(tmp_path), fakes.runner, fakes.trainer)
    plays = [c for c in fakes.cmds if "scripts/selfplay-forge.ts" in c]
    assert len(plays) == 1
    assert all("--deck-a" not in c and "--opponent" not in c for c in plays)


def test_gumbel_pick_reaches_every_script(tmp_path: Path) -> None:
    fakes = Fakes([0.5])
    run(
        args(tmp_path, "--expert-games", "4", "--pick", "gumbel"),
        fakes.runner,
        fakes.trainer,
    )
    scripts = [
        c
        for c in fakes.cmds
        if "scripts/selfplay-forge.ts" in c or "scripts/gate-forge.ts" in c
    ]
    assert len(scripts) == 3
    for cmd in scripts:
        assert cmd[cmd.index("--pick") + 1] == "gumbel"
    fakes = Fakes([0.5])
    run(args(tmp_path / "plain"), fakes.runner, fakes.trainer)
    assert all("--pick" not in c for c in fakes.cmds)
