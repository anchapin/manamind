"""Export a simple-mode checkpoint to ONNX with a versioned schema (#46).

The exported graph takes the raw observation vector (the features
``SimpleStateEncoder.features`` builds, before its learned projection)
and returns ``policy_logits`` over the full action space plus ``value``
in [-1, 1] for the player to act. A host engine fills the observation
from its own game state, masks the logits to its legal actions, and
never needs PyTorch.

Alongside ``model.onnx`` it writes ``model.schema.json``: the schema
version, every observation field in order with its scaling, and the
action names in id order. Hosts should refuse a file whose
``schema_version`` they don't know, so old weights fail loudly instead
of silently misreading state.

    python scripts/export_onnx.py CHECKPOINT OUT_DIR [--check N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_simple import load_checkpoint  # noqa: E402

from manamind.core.agent import MCTSAgent  # noqa: E402
from manamind.models.policy_value_network import (  # noqa: E402
    PolicyValueNetwork,
)
from manamind.rules.simple import (  # noqa: E402
    SIMPLE_PHASES,
    SimpleStateEncoder,
    create_simple_game_start,
)

SCHEMA_VERSION = "simple-v1"

_PLAYER_FIELDS: List[Tuple[str, str]] = [
    ("life", "life / 20"),
    ("hand_size", "cards in hand / 7"),
    ("library_size", "cards in library / 40"),
    ("graveyard_size", "cards in graveyard / 10"),
    ("lands", "lands on battlefield / 10"),
    ("untapped_lands", "untapped lands / 10"),
    ("creatures", "creatures on battlefield / 10"),
    ("total_power", "sum of creature power / 20"),
    ("total_toughness", "sum of creature toughness / 20"),
    ("untapped_creatures", "untapped creatures / 10"),
]


def observation_fields() -> List[Dict[str, str]]:
    """Every observation entry, in order, with how the host computes it."""
    fields: List[Dict[str, str]] = []
    for who in ("self", "opponent"):
        for name, scale in _PLAYER_FIELDS:
            fields.append({"name": f"{who}.{name}", "scale": scale})
    for phase in SIMPLE_PHASES:
        fields.append(
            {"name": f"phase.{phase}", "scale": "1 if current phase else 0"}
        )
    fields.append(
        {"name": "self_is_active", "scale": "1 if it's our turn else 0"}
    )
    fields.append({"name": "turn", "scale": "min(turn number, 60) / 60"})
    assert len(fields) == SimpleStateEncoder.FEATURE_DIM
    return fields


class _ObservationModel(nn.Module):
    """Raw observation -> (policy_logits, value), the encoder included."""

    def __init__(self, network: PolicyValueNetwork):
        super().__init__()
        encoder = network.state_encoder
        if not isinstance(encoder, SimpleStateEncoder):
            raise TypeError(
                "only simple-mode checkpoints export for now; got "
                f"{type(encoder).__name__}"
            )
        self.projection = encoder.projection
        self.network = network

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.network(self.projection(obs))
        return logits, value


def action_names() -> List[str]:
    space = MCTSAgent(player_id=0, simulations=1).action_space
    ids = space.action_to_id
    return [name for name, _ in sorted(ids.items(), key=lambda kv: kv[1])]


def probe_observations(count: int, seed: int = 0) -> np.ndarray:
    """Observations from real positions: opening states across seeds."""
    encoder = SimpleStateEncoder()
    rows = [
        encoder.features(create_simple_game_start(seed=seed + i)).numpy()
        for i in range(count)
    ]
    return np.stack(rows).astype(np.float32)


def export(network: PolicyValueNetwork, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _ObservationModel(network).eval()
    onnx_path = out_dir / "model.onnx"
    dummy = torch.zeros(1, SimpleStateEncoder.FEATURE_DIM)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["observation"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    names = action_names()
    if len(names) != network.action_space_size:
        raise ValueError(
            f"{len(names)} action names for a {network.action_space_size}"
            "-wide policy head"
        )
    schema = {
        "schema_version": SCHEMA_VERSION,
        "observation_dim": SimpleStateEncoder.FEATURE_DIM,
        "observation": observation_fields(),
        "perspective": "player to act (holding priority)",
        "actions": names,
        "outputs": {
            "policy_logits": "unnormalised, full action space; mask to "
            "legal actions before softmax",
            "value": "expected result for the player to act, -1 to 1",
        },
    }
    schema_path = out_dir / "model.schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    return onnx_path, schema_path


def check_parity(
    network: PolicyValueNetwork,
    onnx_path: Path,
    obs: np.ndarray,
    atol: float = 1e-4,
) -> Dict[str, float]:
    """Max abs difference between PyTorch and onnxruntime outputs."""
    import onnxruntime as ort

    model = _ObservationModel(network).eval()
    with torch.no_grad():
        ref_logits, ref_value = model(torch.from_numpy(obs))
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    logits, value = session.run(None, {"observation": obs})
    diff = {
        "logits": float(np.abs(logits - ref_logits.numpy()).max()),
        "value": float(np.abs(value - ref_value.numpy()).max()),
    }
    if max(diff.values()) > atol:
        raise AssertionError(f"ONNX output drifted from PyTorch: {diff}")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--check",
        type=int,
        default=64,
        help="positions to compare against PyTorch (0 to skip)",
    )
    args = parser.parse_args()

    network = load_checkpoint(args.checkpoint)
    onnx_path, schema_path = export(network, args.out_dir)
    print(f"wrote {onnx_path} and {schema_path}")
    if args.check > 0:
        diff = check_parity(network, onnx_path, probe_observations(args.check))
        print(
            f"parity on {args.check} positions: max |logit diff| "
            f"{diff['logits']:.2e}, max |value diff| {diff['value']:.2e}"
        )


if __name__ == "__main__":
    main()
