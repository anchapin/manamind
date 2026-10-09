import json
import math
from pathlib import Path

import pytest

from manamind.evaluation.yardsticks import (
    forge_rows,
    forge_table,
    main,
    resolve_ckpts,
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
