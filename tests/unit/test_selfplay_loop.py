"""Self-play round loop: promote-or-keep, Elo ladder, kill switch (#87)."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from manamind.training.selfplay_loop import (
    build_parser,
    elo_diff,
    main,
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


def test_c_scale_reaches_selfplay_but_not_gate(tmp_path: Path) -> None:
    fakes = Fakes([0.5])
    run(
        args(tmp_path, "--expert-games", "4", "--c-scale", "0.1"),
        fakes.runner,
        fakes.trainer,
    )
    plays = [c for c in fakes.cmds if "scripts/selfplay-forge.ts" in c]
    assert len(plays) == 2
    for cmd in plays:
        assert cmd[cmd.index("--c-scale") + 1] == "0.1"
    gates = [c for c in fakes.cmds if "scripts/gate-forge.ts" in c]
    assert gates and all("--c-scale" not in c for c in gates)
    fakes = Fakes([0.5])
    run(args(tmp_path / "plain"), fakes.runner, fakes.trainer)
    assert all("--c-scale" not in c for c in fakes.cmds)


def test_fork_on_plateau_fans_out_and_promotes_best(tmp_path: Path) -> None:
    # Rounds 1-2 flat, so the kill switch fans out two probes at
    # c-scales 0.25 and 0.5; the second scores higher and becomes champion.
    fakes = Fakes([0.5, 0.45, 0.6, 0.7, 0.4])
    state = run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--fork-on-plateau",
            "--fork-c-scales",
            "0.25,0.5",
        ),
        fakes.runner,
        fakes.trainer,
    )
    # Three loop iterations: two flat rounds, then the fork (two probes),
    # then one serial round from the new champion.
    assert state["round"] == 5
    assert state["champion"] == "rounds/0004"
    assert state["stopped"] is None and state["flat"] == 1
    assert state["rises"] == 1
    assert [h.get("fork") for h in state["history"]] == [
        None,
        None,
        1,
        2,
        None,
    ]
    assert state["elo"] == pytest.approx(elo_diff(0.7, 20))
    plays = [c for c in fakes.cmds if "scripts/selfplay-forge.ts" in c]
    # Both probes play from the flatlined champion at their own c-scale;
    # the later serial round plays from the promoted one.
    assert all(
        c[c.index("--model") + 1].endswith("rounds/0000/forge_pointer.onnx")
        for c in plays[2:4]
    )
    assert plays[4][plays[4].index("--model") + 1].endswith(
        "rounds/0004/forge_pointer.onnx"
    )
    assert [c[c.index("--c-scale") + 1] for c in plays[2:4]] == ["0.25", "0.5"]
    # Probe seeds live in their own blocks, clear of the serial rounds'.
    assert plays[2][plays[2].index("--seed") + 1] == "5000005"
    assert plays[3][plays[3].index("--seed") + 1] == "5000008"
    gate_seeds = [
        c[c.index("--seed") + 1]
        for c in fakes.cmds
        if "scripts/gate-forge.ts" in c
    ]
    assert gate_seeds == [
        "1000001",
        "1000021",
        "6000005",
        "6000008",
        "1000081",
    ]
    # Probes train from the champion checkpoint (fresh optimizer each);
    # the later serial round trains from the promoted one.
    assert all(
        t2[t2.index("--init") + 1].endswith("rounds/0000/last.pt")
        for t2 in fakes.trains[2:4]
    )
    assert fakes.trains[4][fakes.trains[4].index("--init") + 1].endswith(
        "rounds/0004/last.pt"
    )
    assert "promoted (fork)" in (tmp_path / "run" / "loop.md").read_text()


def test_fork_round_keeps_champion_and_stops_without_a_pass(
    tmp_path: Path,
) -> None:
    fakes = Fakes([0.5, 0.45, 0.5, 0.4])
    state = run(
        args(
            tmp_path,
            "--rounds",
            "10",
            "--flat-rounds",
            "2",
            "--fork-on-plateau",
            "--fork-c-scales",
            "0.25,0.5",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert state["round"] == 4
    assert state["champion"] == "rounds/0000"
    assert state["rises"] == 0 and state["elo"] == 0.0
    assert state["stopped"] is not None
    assert "fork round: no probe promoted" in state["stopped"]
    assert "Stopped: fork round" in table(state)


def test_kill_switch_stops_when_forking_is_off(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6])
    state = run(
        args(tmp_path, "--rounds", "10", "--flat-rounds", "2"),
        fakes.runner,
        fakes.trainer,
    )
    assert state["round"] == 2
    assert state["stopped"] == "Elo flat for 2 rounds"
    assert not any(h.get("fork") for h in state["history"])


def test_fork_flush_value_head_reaches_the_trainer(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--fork-on-plateau",
            "--fork-c-scales",
            "0.5",
            "--fork-flush-value-head",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--flush-value-head" in fakes.trains[2]
    assert "--flush-value-head" not in fakes.trains[0]


def test_fork_round_inherits_kl_weighting(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--fork-on-plateau",
            "--fork-c-scales",
            "0.5",
            "--kl-weighted",
            "--kl-eps",
            "0.01",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--kl-weighted" in fakes.trains[2]
    assert fakes.trains[2][fakes.trains[2].index("--kl-eps") + 1] == "0.01"
    # The plain main-round path carries the same flags.
    assert "--kl-weighted" in fakes.trains[0]


def test_fork_round_without_kl_weighting_stays_plain(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--fork-on-plateau",
            "--fork-c-scales",
            "0.5",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--kl-weighted" not in fakes.trains[2]


@pytest.mark.parametrize("value", ["", "0", "0.5,-1", "abc"])
def test_fork_c_scales_must_be_positive(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-dir",
                str(tmp_path),
                "--pn-dir",
                str(tmp_path),
                "--fork-c-scales",
                value,
            ]
        )
    assert "c-scale" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_c_scale_must_be_positive(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit, match="--c-scale must be positive"):
        main(
            [
                "--run-dir",
                str(tmp_path),
                "--pn-dir",
                str(tmp_path),
                "--c-scale",
                value,
            ]
        )


def test_expert_decisions_flag_records_both_sides(tmp_path: Path) -> None:
    a = args(tmp_path, "--expert-games", "4", "--expert-decisions")
    fakes = Fakes([0.5])
    run(a, fakes.runner, fakes.trainer)
    plays = [c for c in fakes.cmds if "scripts/selfplay-forge.ts" in c]
    experts = [c for c in plays if "--opponent" in c]
    assert experts
    assert all(c[c.index("--expert-record") + 1] == "both" for c in experts)
    assert all(
        "--expert-record" not in c for c in plays if "--opponent" not in c
    )


def test_expert_policy_weight_forwards_to_train(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--expert-policy-weight",
            "2.0",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--expert-policy-weight" in fakes.trains[0]
    assert (
        fakes.trains[0][fakes.trains[0].index("--expert-policy-weight") + 1]
        == "2.0"
    )


def test_expert_policy_weight_default_is_omitted(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(tmp_path, "--rounds", "3", "--flat-rounds", "2"),
        fakes.runner,
        fakes.trainer,
    )
    assert "--expert-policy-weight" not in fakes.trains[0]


def test_imitation_epochs_forwards_to_train(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--imitation-epochs",
            "2",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--imitation-epochs" in fakes.trains[0]
    assert (
        fakes.trains[0][fakes.trains[0].index("--imitation-epochs") + 1] == "2"
    )


def test_imitation_epochs_default_is_omitted(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(tmp_path, "--rounds", "3", "--flat-rounds", "2"),
        fakes.runner,
        fakes.trainer,
    )
    assert "--imitation-epochs" not in fakes.trains[0]


def test_imitation_lr_scale_forwards_to_train(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(
            tmp_path,
            "--rounds",
            "3",
            "--flat-rounds",
            "2",
            "--imitation-lr-scale",
            "0.3",
        ),
        fakes.runner,
        fakes.trainer,
    )
    assert "--imitation-lr-scale" in fakes.trains[0]
    assert (
        fakes.trains[0][fakes.trains[0].index("--imitation-lr-scale") + 1]
        == "0.3"
    )


def test_imitation_lr_scale_default_is_omitted(tmp_path: Path) -> None:
    fakes = Fakes([0.5, 0.45, 0.6, 0.4])
    run(
        args(tmp_path, "--rounds", "3", "--flat-rounds", "2"),
        fakes.runner,
        fakes.trainer,
    )
    assert "--imitation-lr-scale" not in fakes.trains[0]
