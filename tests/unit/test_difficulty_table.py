"""Difficulty strength/latency table (#46)."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import difficulty_table  # noqa: E402
import export_onnx  # noqa: E402
import train_simple  # noqa: E402


def test_table_covers_policy_and_search_settings(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = train_simple.build_simple_network(
        action_space_size=len(export_onnx.action_names())
    )
    ckpt = tmp_path / "iter_001.pt"
    torch.save(
        {
            "network": net.state_dict(),
            "action_space_size": net.action_space_size,
        },
        ckpt,
    )
    settings = [
        difficulty_table.Setting("policy, greedy"),
        difficulty_table.Setting("policy, blunder 50%", blunder_rate=0.5),
        difficulty_table.Setting("search, 2 sims", simulations=2),
    ]
    table = difficulty_table.build_table(
        ckpt, games=2, ref_simulations=2, settings=settings
    )
    assert [r["setting"] for r in table["rows"]] == [s.name for s in settings]
    for row in table["rows"]:
        for cell in ("vs_random", "vs_full_strength"):
            c = row[cell]
            assert 0.0 <= c["ci_low"] <= c["score"] <= c["ci_high"] <= 1.0
            assert c["ms_per_decision"] >= 0.0
    assert table["rows"][0]["kind"] == "policy (onnx)"
    assert table["rows"][2]["kind"] == "search (torch)"


def test_parse_settings_specs_and_ladder() -> None:
    parse = difficulty_table.parse_setting
    assert parse("policy") == difficulty_table.Setting("policy, greedy")
    s = parse("sims=1/blunder=0.25")
    assert (s.simulations, s.blunder_rate, s.temperature) == (1, 0.25, 0.0)
    assert s.name == "search, 1 sim, blunder 25%"
    assert parse("temp=1.0").name == "policy, T=1"
    assert parse("sims=20").name == "search, 20 sims"
    assert difficulty_table.parse_settings("ladder") == difficulty_table.LADDER
    assert (
        difficulty_table.parse_settings(None)
        == difficulty_table.DEFAULT_SETTINGS
    )
    for bad in ("sims=5/foo=1", "blunder=2", "sims", "sims=-1"):
        with pytest.raises(ValueError):
            parse(bad)


def test_blundering_search_setting_runs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = train_simple.build_simple_network(
        action_space_size=len(export_onnx.action_names())
    )
    ckpt = tmp_path / "iter_001.pt"
    torch.save(
        {
            "network": net.state_dict(),
            "action_space_size": net.action_space_size,
        },
        ckpt,
    )
    table = difficulty_table.build_table(
        ckpt,
        games=2,
        ref_simulations=2,
        settings=[difficulty_table.parse_setting("sims=1/blunder=0.5")],
    )
    assert table["rows"][0]["simulations"] == 1
    assert table["rows"][0]["blunder_rate"] == 0.5
