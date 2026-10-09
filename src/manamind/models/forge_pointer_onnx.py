"""ONNX export of ``ForgePointerNet`` for host engines (#86).

Planar Nexus runs Gumbel search in Node next to its engine, so the network
has to run without PyTorch. ``ForgePointerNet`` reads Python dicts; the
exported graph instead takes padded tensors that a host builds from the
same Forge-bridge view (Planar Nexus ``playerView``):

* per card zone (``hand``, ``bf``, ``obf``): features ``[n, CARD_FEATURES]``
  (``card_features``), name buckets ``[n]`` (``name_bucket``: CRC-32 of the
  lower-cased name mod ``NAME_BUCKETS``), and a 0/1 mask ``[n]``;
* per graveyard (``gy``, ``ogy``): name buckets and a mask;
* ``glob``: ``global_features`` ``[GLOBAL_FEATURES]``;
* option lists for the three heads: priority cards plus ``[land, spell]``
  flags, attackers, blockers, and the attacking creatures to block.

Every list needs at least one row: pad an empty one with a zero row (mask 0
for zones; for option lists the extra logits are simply ignored). Masked
pooling is exact because the card encoder ends in a ReLU, so the zone max
over ``emb * mask`` equals the max over the real cards, and an empty zone
pools to zeros just as in PyTorch.

Outputs: ``value`` ``[1]``, ``priority`` ``[P + 1]`` (last entry is pass),
``attack`` ``[A]``, and ``block`` ``[B, K + 1]`` (last column is no block).
``unpack`` slices the padding back off.

    python -m manamind.models.forge_pointer_onnx CKPT OUT_DIR [--check N]

writes ``forge_pointer.onnx`` and ``forge_pointer.schema.json``. Hosts must
refuse a schema version they don't know.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from manamind.models.forge_pointer import (
    CARD_FEATURES,
    DECISION_TYPES,
    GLOBAL_FEATURES,
    NAME_BUCKETS,
    NAME_DIM,
    PHASES,
    PIPS,
    TYPE_WORDS,
    Decision,
    ForgePointerNet,
    card_features,
    global_features,
    load_pointer_net,
    name_bucket,
)

SCHEMA_VERSION = 1
ZONES = ("hand", "bf", "obf")
GRAVES = ("gy", "ogy")
INPUT_NAMES: Tuple[str, ...] = (
    *(f"{z}_{k}" for z in ZONES for k in ("x", "id", "m")),
    *(f"{g}_{k}" for g in GRAVES for k in ("id", "m")),
    "glob",
    "pri_x",
    "pri_id",
    "pri_flags",
    "att_x",
    "att_id",
    "blk_x",
    "blk_id",
    "batt_x",
    "batt_id",
)
OUTPUT_NAMES = ("value", "priority", "attack", "block")
Inputs = Dict[str, np.ndarray]


class PointerExport(nn.Module):
    """Tensor-in, tensor-out wrapper around a ``ForgePointerNet``."""

    def __init__(self, net: ForgePointerNet) -> None:
        super().__init__()
        self.net = net

    def _cards(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net.card_mlp(
            torch.cat([x, self.net.name_emb(ids)], dim=-1)
        )
        return out

    def _pool(
        self, x: torch.Tensor, ids: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        emb = self._cards(x, ids) * m.unsqueeze(-1)
        count = torch.clamp(m.sum(), min=1.0)
        return torch.cat([emb.sum(0) / count, emb.max(0).values])

    def _names(self, ids: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        emb = self.net.name_emb(ids) * m.unsqueeze(-1)
        mean: torch.Tensor = emb.sum(0) / torch.clamp(m.sum(), min=1.0)
        return mean

    def forward(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        a = dict(zip(INPUT_NAMES, args))
        net = self.net
        state = net.state_mlp(
            torch.cat(
                [
                    *(
                        self._pool(a[f"{z}_x"], a[f"{z}_id"], a[f"{z}_m"])
                        for z in ZONES
                    ),
                    *(self._names(a[f"{g}_id"], a[f"{g}_m"]) for g in GRAVES),
                    a["glob"],
                ]
            )
        )
        value = torch.tanh(net.value_head(state))

        pri = net.option_proj(
            torch.cat(
                [self._cards(a["pri_x"], a["pri_id"]), a["pri_flags"]], -1
            )
        )
        opts = torch.cat([pri, net.pass_token.unsqueeze(0)], 0)
        priority = net.priority_head(
            torch.cat([state.unsqueeze(0).expand(opts.shape[0], -1), opts], -1)
        ).squeeze(-1)

        att = self._cards(a["att_x"], a["att_id"])
        attack = net.attack_head(
            torch.cat([state.unsqueeze(0).expand(att.shape[0], -1), att], -1)
        ).squeeze(-1)

        b = self._cards(a["blk_x"], a["blk_id"])
        k = self._cards(a["batt_x"], a["batt_id"])
        nb, nk = b.shape[0], k.shape[0]
        pair = torch.cat(
            [
                state.view(1, 1, -1).expand(nb, nk, -1),
                b.unsqueeze(1).expand(nb, nk, -1),
                k.unsqueeze(0).expand(nb, nk, -1),
            ],
            -1,
        )
        hit = net.block_head(pair).squeeze(-1)
        none = net.no_block_head(
            torch.cat([state.unsqueeze(0).expand(nb, -1), b], -1)
        )
        block = torch.cat([hit, none], dim=1)
        return value, priority, attack, block


def _cards(cards: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, ...]:
    """Features, name buckets, and mask; one zero row when empty."""
    if not cards:
        return (
            np.zeros((1, CARD_FEATURES), np.float32),
            np.zeros(1, np.int64),
            np.zeros(1, np.float32),
        )
    return (
        np.asarray([card_features(c) for c in cards], np.float32),
        np.asarray(
            [name_bucket(str(c.get("name", ""))) for c in cards], np.int64
        ),
        np.ones(len(cards), np.float32),
    )


def _names(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not names:
        return np.zeros(1, np.int64), np.zeros(1, np.float32)
    ids = np.asarray([name_bucket(str(n)) for n in names], np.int64)
    return ids, np.ones(len(names), np.float32)


def pack(d: Decision) -> Inputs:
    """Graph inputs for one decision (the reference a host mirrors)."""
    out: Inputs = {}
    for zone, key in zip(ZONES, ("hand", "battlefield", "opp_battlefield")):
        x, ids, m = _cards(d.get(key, []))
        out.update({f"{zone}_x": x, f"{zone}_id": ids, f"{zone}_m": m})
    for grave, key in zip(GRAVES, ("graveyard", "opp_graveyard")):
        ids, m = _names(d.get(key, []))
        out.update({f"{grave}_id": ids, f"{grave}_m": m})
    out["glob"] = np.asarray(global_features(d), np.float32)

    t = d.get("t")
    options = d.get("options", []) if t == "priority" else []
    pri_cards = [o.get("card", {}) for o in options]
    out["pri_x"], out["pri_id"], _ = _cards(pri_cards)
    flags = [
        [float(bool(o.get("land"))), float(bool(o.get("spell")))]
        for o in options
    ]
    out["pri_flags"] = np.asarray(flags or [[0.0, 0.0]], np.float32)
    attackers = d.get("options", []) if t == "attack" else []
    out["att_x"], out["att_id"], _ = _cards(attackers)
    block = t == "block"
    out["blk_x"], out["blk_id"], _ = _cards(
        d.get("blockers", []) if block else []
    )
    out["batt_x"], out["batt_id"], _ = _cards(
        d.get("attackers", []) if block else []
    )
    return out


def unpack(d: Decision, outputs: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Slice padding off the graph outputs for decision ``d``."""
    value, priority, attack, block = outputs
    t = d.get("t")
    n_pri = len(d.get("options", [])) if t == "priority" else 0
    n_att = len(d.get("options", [])) if t == "attack" else 0
    n_blk = len(d.get("blockers", [])) if t == "block" else 0
    n_batt = len(d.get("attackers", [])) if t == "block" else 0
    cols = list(range(n_batt)) + [block.shape[1] - 1]
    # Like ``ForgePointerNet.act``: no blockers or no attackers, no choice.
    blocks = block[:n_blk][:, cols] if n_blk and n_batt else block[:0, :1]
    return {
        "value": float(value.reshape(-1)[0]),
        "priority": np.concatenate([priority[:n_pri], priority[-1:]]),
        "attack": attack[:n_att],
        "block": blocks,
    }


def schema(net: ForgePointerNet) -> Dict[str, Any]:
    """What a host needs to build the inputs and read the outputs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": "ForgePointerNet",
        "card_dim": net.card_dim,
        "state_dim": net.state_dim,
        "inputs": list(INPUT_NAMES),
        "outputs": list(OUTPUT_NAMES),
        "card_features": CARD_FEATURES,
        "global_features": GLOBAL_FEATURES,
        "name_buckets": NAME_BUCKETS,
        "name_dim": NAME_DIM,
        "name_hash": "crc32(name.lower() utf-8) % name_buckets",
        "type_words": list(TYPE_WORDS),
        "pips": list(PIPS),
        "phases": list(PHASES),
        "decision_types": list(DECISION_TYPES),
        "padding": "every list needs >= 1 row; pad with zeros (mask 0)",
    }


def export(net: ForgePointerNet, out_dir: Path) -> Path:
    """Write ``forge_pointer.onnx`` and its schema; return the model path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "forge_pointer.onnx"
    sample = pack(
        {
            "t": "block",
            "hand": [{}, {}],
            "battlefield": [{}],
            "blockers": [{}, {}],
            "attackers": [{}],
        }
    )
    args = tuple(torch.from_numpy(sample[n]) for n in INPUT_NAMES)
    axes: Dict[str, Dict[int, str]] = {}
    for name in INPUT_NAMES:
        if name != "glob":
            axes[name] = {0: f"n_{name.split('_')[0]}"}
    axes["priority"] = {0: "n_pri_plus_pass"}
    axes["attack"] = {0: "n_att"}
    axes["block"] = {0: "n_blk", 1: "n_batt_plus_none"}
    torch.onnx.export(
        PointerExport(net.eval()),
        args,
        str(path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    (out_dir / "forge_pointer.schema.json").write_text(
        json.dumps(schema(net), indent=2) + "\n"
    )
    return path


def torch_reference(net: ForgePointerNet, d: Decision) -> Dict[str, Any]:
    """The same quantities straight from ``ForgePointerNet``."""
    with torch.no_grad():
        state = net.encode_state(d)
        t = d.get("t")
        empty = np.zeros(0, np.float32)
        out: Dict[str, Any] = {
            "value": float(net.value(state)),
            "priority": net.priority_logits(state, []).numpy(),
            "attack": empty,
            "block": np.zeros((0, 1), np.float32),
        }
        if t == "priority":
            out["priority"] = net.priority_logits(
                state, d.get("options", [])
            ).numpy()
        if t == "attack" and d.get("options"):
            out["attack"] = net.attack_logits(state, d["options"]).numpy()
        if t == "block" and d.get("blockers") and d.get("attackers"):
            out["block"] = net.block_logits(
                state, d["blockers"], d["attackers"]
            ).numpy()
        return out


def max_abs_diff(
    net: ForgePointerNet, model: Path, decisions: List[Decision]
) -> float:
    """Largest |ONNX - PyTorch| over every output of every decision."""
    import onnxruntime as ort  # type: ignore[import-untyped]

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    worst = 0.0
    for d in decisions:
        got = unpack(d, sess.run(list(OUTPUT_NAMES), pack(d)))
        ref = torch_reference(net, d)
        for key in OUTPUT_NAMES:
            a = np.asarray(got[key], np.float64)
            b = np.asarray(ref[key], np.float64)
            if a.shape != b.shape:
                raise AssertionError(f"{key}: {a.shape} != {b.shape}")
            if a.size:
                worst = max(worst, float(np.abs(a - b).max()))
    return worst


def main(argv: Sequence[str] = ()) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ckpt")
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--check",
        type=Path,
        help="JSONL of bridge decisions to check ONNX/PyTorch parity on",
    )
    args = p.parse_args(list(argv) or None)
    net, _ = load_pointer_net(args.ckpt)
    path = export(net.eval(), args.out_dir)
    print(f"wrote {path}")
    if args.check:
        lines = args.check.read_text().splitlines()
        decisions = [json.loads(x) for x in lines if x.strip()]
        print(f"max |onnx - torch| = {max_abs_diff(net, path, decisions):.2e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
