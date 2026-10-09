"""ONNX export of ForgePointerNet for host engines (#86)."""

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_forge_pointer import (  # noqa: E402
    ATTACK,
    BEAR,
    BLOCK,
    FOREST,
    PRIORITY,
    VIEW,
)

from manamind.models import forge_pointer_onnx as fo  # noqa: E402
from manamind.models.forge_pointer import ForgePointerNet  # noqa: E402

TOL = 1e-5


def _net() -> ForgePointerNet:
    torch.manual_seed(0)
    return ForgePointerNet().eval()


def test_export_matches_pytorch_on_every_head(tmp_path: Path) -> None:
    net = _net()
    model = fo.export(net, tmp_path)
    assert fo.max_abs_diff(net, model, [PRIORITY, ATTACK, BLOCK]) < TOL


def test_empty_zones_and_lists_match(tmp_path: Path) -> None:
    net = _net()
    model = fo.export(net, tmp_path)
    bare = {"t": "priority", "turn": 1, "phase": "UPKEEP", "options": []}
    no_attackers = dict(VIEW, t="attack", options=[])
    no_blockers = dict(VIEW, t="block", blockers=[], attackers=[BEAR])
    decisions = [bare, no_attackers, no_blockers]
    assert fo.max_abs_diff(net, model, decisions) < TOL


def test_sizes_other_than_the_trace_sample(tmp_path: Path) -> None:
    net = _net()
    model = fo.export(net, tmp_path)
    many = {"text": "x", "land": False, "spell": True, "card": BEAR}
    wide = dict(VIEW, hand=[BEAR] * 7, battlefield=[FOREST] * 9)
    decisions = [
        dict(wide, t="priority", options=[many] * 6),
        dict(wide, t="attack", options=[BEAR] * 5),
        dict(wide, t="block", blockers=[BEAR] * 4, attackers=[BEAR] * 3),
    ]
    assert fo.max_abs_diff(net, model, decisions) < TOL


def test_schema_lists_inputs_and_version(tmp_path: Path) -> None:
    net = _net()
    fo.export(net, tmp_path)
    schema = json.loads((tmp_path / "forge_pointer.schema.json").read_text())
    assert schema["schema_version"] == fo.SCHEMA_VERSION
    assert schema["inputs"] == list(fo.INPUT_NAMES)
    assert schema["outputs"] == list(fo.OUTPUT_NAMES)
    assert schema["card_dim"] == net.card_dim


def test_cli_exports_a_checkpoint_and_checks_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    net = _net()
    ckpt = tmp_path / "last.pt"
    torch.save({"config": {}, "network": net.state_dict()}, ckpt)
    check = tmp_path / "decisions.jsonl"
    check.write_text(
        "\n".join(json.dumps(d) for d in (PRIORITY, ATTACK, BLOCK)) + "\n"
    )
    out = tmp_path / "out"
    assert fo.main([str(ckpt), str(out), "--check", str(check)]) == 0
    assert (out / "forge_pointer.onnx").exists()
    printed = capsys.readouterr().out
    diff = float(printed.rsplit("=", 1)[1])
    assert diff < TOL
