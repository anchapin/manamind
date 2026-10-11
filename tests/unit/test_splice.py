"""Tests for the value-head splice (#96)."""

from pathlib import Path

import pytest
import torch

from manamind.models.forge_pointer import ForgePointerNet, load_pointer_net
from manamind.models.splice import main, splice_value_head


def ckpt(seed: int, tag: str) -> dict:
    torch.manual_seed(seed)
    net = ForgePointerNet()
    return {
        "network": net.state_dict(),
        "config": {"card_dim": net.card_dim, "state_dim": net.state_dim},
        "meta": {"tag": tag},
    }


def test_only_the_value_head_moves() -> None:
    a, b = ckpt(1, "a"), ckpt(2, "b")
    out = splice_value_head(a, b)
    for k, v in out["network"].items():
        src = b if k.startswith("value_head.") else a
        assert torch.equal(v, src["network"][k]), k
    assert any(k.startswith("value_head.") for k in out["network"])
    assert out["meta"] == {"tag": "a", "value_head_from": {"tag": "b"}}
    assert a["meta"] == {"tag": "a"}


def test_rejects_mismatched_or_missing_heads() -> None:
    a, b = ckpt(1, "a"), ckpt(2, "b")
    with pytest.raises(ValueError, match="different configs"):
        splice_value_head(a, {**b, "config": {"card_dim": 1}})
    no_value = {
        **b,
        "network": {
            k: v
            for k, v in b["network"].items()
            if not k.startswith("value_head.")
        },
    }
    with pytest.raises(ValueError, match="no value_head"):
        splice_value_head(a, no_value)
    with pytest.raises(ValueError, match="lacks"):
        splice_value_head(no_value, b)


def test_main_writes_checkpoint_and_onnx(tmp_path: Path) -> None:
    a, b = ckpt(1, "a"), ckpt(2, "b")
    torch.save(a, tmp_path / "a.pt")
    torch.save(b, tmp_path / "b.pt")
    out = tmp_path / "spliced"
    path = main(
        [
            "--policy",
            str(tmp_path / "a.pt"),
            "--value",
            str(tmp_path / "b.pt"),
            "--out",
            str(out),
        ]
    )
    assert path == out / "forge_pointer.onnx" and path.is_file()
    net, meta = load_pointer_net(str(out / "last.pt"))
    assert meta is not None and meta["splice"]["value"].endswith("b.pt")
    want = b["network"]["value_head.0.weight"]
    assert torch.equal(net.state_dict()["value_head.0.weight"], want)
