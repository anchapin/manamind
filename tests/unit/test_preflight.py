"""Training preflight: bounds and wall-time projection (#68)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"
_spec = importlib.util.spec_from_file_location("preflight", _SCRIPT)
assert _spec is not None and _spec.loader is not None
preflight = importlib.util.module_from_spec(_spec)
sys.modules["preflight"] = preflight
_spec.loader.exec_module(preflight)

ts = preflight._load_train_simple()


def _args(*argv: str):
    return ts.build_parser().parse_args(list(argv))


def test_defaults_pass_hard_bounds() -> None:
    errors, _ = preflight.check_bounds(_args())
    assert errors == []


def test_out_of_bounds_rejected() -> None:
    errors, _ = preflight.check_bounds(_args("--simulations", "40000"))
    assert any("--simulations=40000" in e for e in errors)


def test_buffer_cap_is_hard() -> None:
    errors, _ = preflight.check_bounds(_args("--buffer-size", "50000"))
    assert any("--buffer-size" in e for e in errors)


def test_recommended_band_warns() -> None:
    errors, warnings = preflight.check_bounds(_args("--simulations", "500"))
    assert errors == []
    assert any("--simulations=500" in w for w in warnings)


def test_odd_games_warns() -> None:
    _, warnings = preflight.check_bounds(_args("--games", "25"))
    assert any("odd" in w for w in warnings)


def test_plateau_needs_anchor() -> None:
    errors, _ = preflight.check_bounds(_args("--plateau", "3"))
    assert any("--plateau needs --anchor" in e for e in errors)


def test_projection_arithmetic() -> None:
    args = _args(
        "--iterations",
        "10",
        "--games",
        "4",
        "--eval-games",
        "2",
        "--train-batches",
        "5",
        "--workers",
        "2",
    )
    t = preflight.SliceTimes(
        selfplay_game=1.0, random_game=0.5, train_step=0.1
    )
    # per iter: (4*1 + 2*0.5 + 2*1)/2 + 5*0.1 = 4.0
    assert preflight.project_seconds(args, t) == pytest.approx(40.0)


def test_projection_counts_anchor_rounds() -> None:
    args = _args(
        "--iterations",
        "10",
        "--games",
        "2",
        "--eval-games",
        "2",
        "--train-batches",
        "1",
        "--anchor",
        "x.pt",
        "--anchor-every",
        "5",
        "--anchor-games",
        "4",
    )
    t = preflight.SliceTimes(1.0, 1.0, 0.0)
    # 10 * (2 + 2 + 2) + 2 rounds * 4 games
    assert preflight.project_seconds(args, t) == pytest.approx(68.0)


def test_main_rejects_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "pf.json"
    code = preflight.main(["--json-out", str(out), "--", "--simulations", "0"])
    assert code == 2
    report = json.loads(out.read_text())
    assert report["errors"]


def test_main_times_a_slice(tmp_path: Path) -> None:
    out = tmp_path / "pf.json"
    code = preflight.main(
        [
            "--json-out",
            str(out),
            "--",
            "--simulations",
            "2",
            "--games",
            "2",
            "--eval-games",
            "2",
            "--iterations",
            "1",
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["projected_minutes"] >= 0
    assert set(report["slice_seconds"]) == {
        "selfplay_game",
        "random_game",
        "train_step",
    }


def test_main_flags_projection_over_timeout(tmp_path: Path) -> None:
    code = preflight.main(
        [
            "--timeout-minutes",
            "0",
            "--",
            "--simulations",
            "2",
            "--games",
            "2",
            "--eval-games",
            "2",
            "--iterations",
            "1",
        ]
    )
    assert code == 3
