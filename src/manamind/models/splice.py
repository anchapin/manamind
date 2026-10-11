"""Splice one checkpoint's value head onto another's network (#96).

Diagnostic for the self-play loop. A candidate that matches bc.pt with no
search but loses to it at 16 sims points at the value head, the only part
the search adds. Splicing the candidate's network with bc.pt's value head
(or bc.pt with the candidate's) and gating the result tells the two apart.

Only ``value_head.*`` moves; the trunk stays the policy checkpoint's, so the
spliced value head reads features from a slightly different encoder. That is
fine when the encoders barely differ (a raw-vs-raw gate near 0.5).

    python -m manamind.models.splice --policy cand.pt --value bc.pt \\
        --out runs/splice_dir

writes ``<out>/last.pt`` and ``<out>/forge_pointer.onnx``.
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from manamind.models.forge_pointer import load_pointer_net

VALUE_PREFIX = "value_head."


def splice_value_head(
    policy: Dict[str, Any], value: Dict[str, Any]
) -> Dict[str, Any]:
    """Return ``policy``'s checkpoint with ``value``'s value-head weights."""
    if policy.get("config", {}) != value.get("config", {}):
        raise ValueError(
            "checkpoints have different configs: "
            f"{policy.get('config')} vs {value.get('config')}"
        )
    keys = [k for k in value["network"] if k.startswith(VALUE_PREFIX)]
    if not keys:
        raise ValueError("value checkpoint has no value_head weights")
    missing = [k for k in keys if k not in policy["network"]]
    if missing:
        raise ValueError(f"policy checkpoint lacks {missing}")
    network = dict(policy["network"])
    for k in keys:
        network[k] = value["network"][k].clone()
    meta = copy.deepcopy(policy.get("meta") or {})
    meta["value_head_from"] = value.get("meta")
    return {
        "network": network,
        "config": dict(policy.get("config", {})),
        "meta": meta,
    }


def _load(path: Path) -> Dict[str, Any]:
    ckpt: Dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False
    )
    return ckpt


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--value", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from manamind.models.forge_pointer_onnx import export

    spliced = splice_value_head(_load(args.policy), _load(args.value))
    spliced["meta"]["splice"] = {
        "policy": str(args.policy),
        "value": str(args.value),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pt = args.out / "last.pt"
    torch.save(spliced, pt)
    net, _ = load_pointer_net(str(pt))
    path = export(net, args.out)
    print(path)
    return path


if __name__ == "__main__":  # pragma: no cover
    main()
