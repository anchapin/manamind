"""ONNX export of simple-mode checkpoints and its schema (#46)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import export_onnx  # noqa: E402
import train_simple  # noqa: E402

from manamind.rules.simple import (  # noqa: E402
    SimpleStateEncoder,
    create_simple_game_start,
)


def _network():
    size = len(export_onnx.action_names())
    torch.manual_seed(0)
    return train_simple.build_simple_network(action_space_size=size).eval()


def test_export_matches_pytorch(tmp_path: Path) -> None:
    net = _network()
    onnx_path, _ = export_onnx.export(net, tmp_path)
    obs = export_onnx.probe_observations(16)
    diff = export_onnx.check_parity(net, onnx_path, obs)
    assert diff["logits"] < 1e-4 and diff["value"] < 1e-4


def test_observation_matches_the_training_encoder() -> None:
    encoder = SimpleStateEncoder()
    state = create_simple_game_start(seed=3)
    obs = export_onnx.probe_observations(1, seed=3)[0]
    np.testing.assert_allclose(obs, encoder.features(state).numpy())


def test_batch_dimension_is_dynamic(tmp_path: Path) -> None:
    onnx_path, _ = export_onnx.export(_network(), tmp_path)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    for batch in (1, 5):
        obs = export_onnx.probe_observations(batch)
        logits, value = session.run(None, {"observation": obs})
        assert logits.shape == (batch, len(export_onnx.action_names()))
        assert value.shape == (batch, 1)
        assert np.all(np.abs(value) <= 1.0)


def test_schema_describes_inputs_and_actions(tmp_path: Path) -> None:
    _, schema_path = export_onnx.export(_network(), tmp_path)
    schema = json.loads(schema_path.read_text())
    assert schema["schema_version"] == export_onnx.SCHEMA_VERSION
    dim = SimpleStateEncoder.FEATURE_DIM
    assert schema["observation_dim"] == dim == len(schema["observation"])
    names = [f["name"] for f in schema["observation"]]
    assert names[0] == "self.life" and names[10] == "opponent.life"
    assert len(set(names)) == len(names)
    space = train_simple.MCTSAgent(player_id=0, simulations=1).action_space
    for i, action in enumerate(schema["actions"]):
        assert space.action_to_id[action] == i


def test_checkpoint_round_trip_through_the_cli(tmp_path: Path) -> None:
    net = _network()
    ckpt = tmp_path / "iter_001.pt"
    torch.save(
        {
            "network": net.state_dict(),
            "action_space_size": net.action_space_size,
        },
        ckpt,
    )
    loaded = train_simple.load_checkpoint(ckpt)
    onnx_path, _ = export_onnx.export(loaded, tmp_path / "out")
    export_onnx.check_parity(net, onnx_path, export_onnx.probe_observations(8))
