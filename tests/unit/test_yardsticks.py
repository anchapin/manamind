import json
import math
import subprocess
from pathlib import Path

import pytest

from manamind.evaluation.yardsticks import (
    build_parser,
    fit_elo,
    forge_rows,
    forge_table,
    main,
    resolve_ckpts,
    run_elo,
    run_pn,
    versus_baseline,
    wilson_interval,
)

SUMMARY = {
    "eval": [
        {
            "ckpt": "forge_bc/bc.pt",
            "trained_games": 1000,
            "mode": "greedy",
            "games": 200,
            "wins": 36,
            "win_rate": 0.18,
            "avg_turns": 11.2,
        },
        {
            "ckpt": "forge_rl/last.pt",
            "trained_games": 3000,
            "mode": "sample",
            "games": 200,
            "wins": 21,
            "win_rate": 0.105,
            "avg_turns": 10.4,
        },
        {
            "ckpt": "forge_rl/last.pt",
            "trained_games": None,
            "mode": "greedy",
            "games": 200,
            "wins": 70,
            "win_rate": 0.35,
            "avg_turns": 12.0,
        },
    ],
    "secs": 12.5,
}


def test_wilson_interval_matches_known_values() -> None:
    lo, hi = wilson_interval(36, 200)
    assert lo == pytest.approx(0.1329, abs=1e-4)
    assert hi == pytest.approx(0.2391, abs=1e-4)
    lo, hi = wilson_interval(0, 200)
    assert lo == 0.0 and 0 < hi < 0.02
    lo, hi = wilson_interval(200, 200)
    assert hi == 1.0 and 0.98 < lo < 1.0
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_versus_baseline() -> None:
    assert versus_baseline(0.2, 0.3, 0.18) == "above"
    assert versus_baseline(0.05, 0.15, 0.18) == "below"
    assert versus_baseline(0.13, 0.24, 0.18) == "within CI"
    assert versus_baseline(float("nan"), float("nan"), 0.18) == "no games"


def test_forge_rows_and_table() -> None:
    rows = forge_rows(SUMMARY)
    assert [r["vs_baseline"] for r in rows] == [
        "within CI",
        "below",
        "above",
    ]
    assert rows[0]["ci_low"] == pytest.approx(0.1329, abs=1e-4)
    table = forge_table(rows)
    lines = table.strip().splitlines()
    assert lines[0].startswith("**vs Forge AI** (baseline 18.0%")
    assert len(lines) == 4 + len(rows)
    assert "| `forge_bc/bc.pt` | 1000 | greedy | 36 / 200 | 18.0% |" in table
    assert "| ? | greedy |" in table


def test_resolve_ckpts(tmp_path: Path) -> None:
    abs_ckpt = tmp_path / "x.pt"
    got = resolve_ckpts(["run/bc.pt", str(abs_ckpt)], tmp_path / "runs")
    assert got == [tmp_path / "runs" / "run" / "bc.pt", abs_ckpt]
    with pytest.raises(ValueError):
        resolve_ckpts(["../etc/x.pt"], tmp_path)


def test_cli_renders_existing_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "eval_summary.json"
    src.write_text(json.dumps(SUMMARY))
    out = tmp_path / "out"
    main(["forge", "--summary", str(src), "--out", str(out)])
    printed = capsys.readouterr().out
    assert printed == (out / "forge.md").read_text()
    data = json.loads((out / "forge.json").read_text())
    assert data["baseline"] == 0.18 and len(data["rows"]) == 3


def test_cli_needs_forge_and_ckpts_to_play(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["forge", "--out", str(tmp_path)])


def test_fit_elo_matches_pairwise_scores() -> None:
    assert fit_elo(2, [(0, 1, 40, 0.5)]) == pytest.approx([0.0, 0.0])
    # 75% over many games is about +191 Elo; the virtual draw barely moves it.
    elo = fit_elo(2, [(0, 1, 4000, 0.75)])
    assert elo[0] == 0.0 and elo[1] == pytest.approx(190.85, abs=0.5)
    # A clean sweep stays finite.
    assert math.isfinite(fit_elo(2, [(0, 1, 10, 1.0)])[1])
    # Transitive ladder: 2 > 1 > 0.
    elo3 = fit_elo(3, [(0, 1, 40, 0.7), (1, 2, 40, 0.7), (0, 2, 40, 0.85)])
    assert elo3[0] == 0.0 < elo3[1] < elo3[2]


def test_elo_leg_plays_every_pair(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for name in ("a", "b", "c"):
        (runs / name).mkdir(parents=True)
        (runs / name / "forge_pointer.onnx").write_bytes(b"")
    strength = {"a": 0, "b": 1, "c": 2}
    calls = []

    def runner(cmd: list, cwd: Path) -> None:
        calls.append((cmd, cwd))
        cand = Path(cmd[cmd.index("--candidate") + 1]).parent.name
        base = Path(cmd[cmd.index("--baseline") + 1]).parent.name
        score = 0.5 + 0.2 * (strength[cand] - strength[base])
        games = int(cmd[cmd.index("--games") + 1])
        Path(cmd[cmd.index("--out") + 1]).write_text(
            json.dumps(
                {
                    "games": games,
                    "wins": round(score * games),
                    "losses": games - round(score * games),
                    "draws": 0,
                    "score": score,
                    "ci95": [0.0, 1.0],
                }
            )
        )

    args = build_parser().parse_args(
        ["elo", "--pn-dir", str(tmp_path), "--runs-dir", str(runs)]
        + ["--ckpts", "a,b,c", "--games", "20", "--out", str(tmp_path / "o")]
    )
    table = run_elo(args, runner)
    assert len(calls) == 3 and all(cwd == tmp_path for _, cwd in calls)
    assert [c[c.index("--seed") + 1] for c, _ in calls] == ["1", "21", "41"]
    assert all(c[c.index("--threshold") + 1] == "0.5" for c, _ in calls)
    rows = json.loads((tmp_path / "o" / "elo.json").read_text())["rows"]
    elo = {r["ckpt"]: r["elo"] for r in rows}
    assert elo["a"] == 0.0 < elo["b"] < elo["c"]
    assert all(r["games"] == 40 for r in rows)
    assert table.splitlines()[2].startswith("| `c` |")


def test_elo_leg_needs_two_existing_ckpts(tmp_path: Path) -> None:
    base = ["elo", "--pn-dir", str(tmp_path), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="at least two"):
        run_elo(build_parser().parse_args(base + ["--ckpts", "x.onnx"]))
    with pytest.raises(SystemExit, match="missing checkpoints"):
        run_elo(
            build_parser().parse_args(
                base + ["--runs-dir", str(tmp_path), "--ckpts", "x,y"]
            )
        )
    with pytest.raises(ValueError, match="leaves --runs-dir"):
        run_elo(build_parser().parse_args(base + ["--ckpts", "../x,y"]))


def _pn_result(win: int, loss: int, draw: int, errors: int = 0) -> dict:
    n = win + loss + draw
    return {
        "games": n,
        "win": win,
        "loss": loss,
        "draw": draw,
        "score": round((win + draw / 2) / n, 3),
        "ci95": [0.1, 0.4],
        "byDeck": {
            "red": {"win": win, "loss": 0, "draw": 0},
            "green": {"win": 0, "loss": loss, "draw": draw},
        },
        "errorCount": errors,
    }


def test_pn_leg_runs_each_ckpt_and_mode(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "forge_pointer.onnx").write_bytes(b"")
    (runs / "b.onnx").write_bytes(b"")
    calls = []

    def runner(cmd: list, cwd: Path) -> None:
        calls.append((cmd, cwd))
        out = Path(cmd[cmd.index("--out") + 1])
        errors = 2 if "--sample" in cmd and "b.onnx" in cmd[6] else 0
        out.write_text(json.dumps(_pn_result(3, 4, 1, errors)))
        if errors:  # yardstick-pn.ts exits 1 but keeps its results
            raise subprocess.CalledProcessError(1, cmd)

    args = build_parser().parse_args(
        ["pn", "--pn-dir", str(tmp_path), "--runs-dir", str(runs)]
        + ["--ckpts", "r1,b.onnx", "--games", "8", "--out", str(tmp_path)]
    )
    table = run_pn(args, runner)
    assert len(calls) == 4 and all(cwd == tmp_path for _, cwd in calls)
    assert all(
        c[:5] == ["npx", "tsx", "scripts/yardstick-pn.ts", "--agent", "model"]
        for c, _ in calls
    )
    assert ["--sample" in c for c, _ in calls] == [False, True] * 2
    assert all(c[c.index("--games") + 1] == "8" for c, _ in calls)
    rows = json.loads((tmp_path / "pn.json").read_text())["rows"]
    assert [(r["ckpt"], r["mode"]) for r in rows] == [
        ("r1", "greedy"),
        ("r1", "sample"),
        ("b.onnx", "greedy"),
        ("b.onnx", "sample"),
    ]
    assert rows[0]["score"] == 0.438 and rows[0]["red"] == 1.0
    assert rows[0]["green"] == 0.1 and rows[3]["errors"] == 2
    assert "| `r1` | greedy | 8 | 3-4-1 | 43.8% | 10.0%-40.0% |" in table


def test_pn_leg_checks_its_inputs(tmp_path: Path) -> None:
    base = ["pn", "--pn-dir", str(tmp_path), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="--ckpts"):
        run_pn(build_parser().parse_args(base))
    with pytest.raises(SystemExit, match="even"):
        run_pn(
            build_parser().parse_args(base + ["--ckpts", "x", "--games", "7"])
        )
    with pytest.raises(SystemExit, match="--modes"):
        run_pn(
            build_parser().parse_args(
                base + ["--ckpts", "x", "--modes", "best"]
            )
        )
    with pytest.raises(SystemExit, match="missing checkpoints"):
        run_pn(
            build_parser().parse_args(
                base + ["--runs-dir", str(tmp_path), "--ckpts", "x"]
            )
        )

    def runner(cmd: list, cwd: Path) -> None:
        raise subprocess.CalledProcessError(1, cmd)

    (tmp_path / "x.onnx").write_bytes(b"")
    args = build_parser().parse_args(
        base + ["--runs-dir", str(tmp_path), "--ckpts", "x.onnx"]
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_pn(args, runner)
