"""Replay and search fixtures for host ports (planar-nexus#2378, #2573)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import export_host_fixtures  # noqa: E402

MODEL_DIR = ROOT / "models" / "simple-v1" / "seed0_settle_iter045"


def test_search_records_index_legal_moves() -> None:
    model, _, net = export_host_fixtures._onnx_model(MODEL_DIR)
    game = export_host_fixtures.play_fixture_game(
        0, model, 0, net, [2, 4], search_decisions=3, search_every=2
    )
    searched = [s for s in game["steps"] if "search" in s]
    assert len(searched) == 3
    for step in searched:
        assert len(step["legal"]) > 1
        assert [r["sims"] for r in step["search"]] == [2, 4]
        for record in step["search"]:
            assert 0 <= record["chosen"] < len(step["legal"])
            assert len(record["policy"]) == len(step["legal"])
            assert sum(record["policy"]) == pytest.approx(1.0, abs=1e-4)


def test_search_needs_a_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        export_host_fixtures.main(
            [str(tmp_path / "out.json"), "--search-sims", "10"]
        )
