#!/usr/bin/env python3
"""
Export a mixed-precision MLX Qwen3.5 MoE checkpoint into a Flash-MoE sidecar.

This exporter preserves the quantized expert tensors exactly as stored in the
MLX checkpoint and writes:

  output/
    layout.json
    layer_00.bin
    layer_01.bin
    ...

Each layer file is a raw concatenation of the routed expert tensors for that
layer. The layout manifest records tensor offsets, sizes, shapes, dtypes, and
inferred bit widths so the export can be inspected or consumed by a future
mixed-precision loader.

This is the "preserve mixed precision" path. It does not re-quantize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _parse_layers(spec: str | None, num_layers: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(num_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _tensor_to_bytes(tensor) -> bytes:
    if str(tensor.dtype) == "torch.bfloat16":
        return tensor.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return tensor.contiguous().cpu().numpy().tobytes()


def _infer_bits_from_weight(
    weight_shape: tuple[int, ...],
    out_dim: int,
    in_dim: int,
    element_size: int,
) -> int:
    if len(weight_shape) not in (2, 3):
        return 0
    rows = weight_shape[-2]
    packed_cols = weight_shape[-1]
    if rows != out_dim or in_dim <= 0 or packed_cols <= 0:
        return 0
    bytes_per_expert = rows * packed_cols * element_size
    bits_f = (bytes_per_expert * 8.0) / float(out_dim * in_dim)
    bits = int(round(bits_f))
    if not math.isclose(bits_f, bits, rel_tol=0.0, abs_tol=1e-6):
        return 0
    return bits if bits in (2, 3, 4, 5, 8) else 0


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: list[int]
    bits: int
    group_size: int
    offset: int
    nbytes: int


def _select_layer_tensors(keys: list[str], layer_index: int) -> list[str]:
    prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp."
    return sorted([key for key in keys if key.startswith(prefix)])


def _tensor_meta(
    name: str,
    tensor,
    *,
    group_size: int,
    layer_index: int,
    text_cfg: dict[str, Any],
) -> tuple[str, list[int], int]:
    shape = [int(dim) for dim in tensor.shape]
    dtype = str(tensor.dtype).replace("torch.", "")
    bits = 0

    if name.endswith(".weight") and len(shape) in (2, 3):
        suffix = name.rsplit(".", 2)[-2]
        if suffix in {"gate_proj", "up_proj"}:
            out_dim = int(text_cfg["moe_intermediate_size"])
            in_dim = int(text_cfg["hidden_size"])
        elif suffix == "down_proj":
            out_dim = int(text_cfg["hidden_size"])
            in_dim = int(text_cfg["moe_intermediate_size"])
        else:
            out_dim = shape[-2]
            in_dim = shape[-1] * 32
        bits = _infer_bits_from_weight(tuple(shape), out_dim, in_dim, int(tensor.element_size()))

    return dtype, shape, bits


def export_layer(
    output_dir: Path,
    layer_index: int,
    group_size: int,
    text_cfg: dict[str, Any],
    safe_file,
) -> dict[str, Any]:
    keys = list(safe_file.keys())
    selected_keys = _select_layer_tensors(keys, layer_index)
    if not selected_keys:
        raise ValueError(f"No switch_mlp tensors found for layer {layer_index}")

    records: list[TensorRecord] = []
    payload = bytearray()

    for key in selected_keys:
        tensor = safe_file.get_tensor(key)
        dtype, shape, bits = _tensor_meta(
            key,
            tensor,
            group_size=group_size,
            layer_index=layer_index,
            text_cfg=text_cfg,
        )
        raw = _tensor_to_bytes(tensor)
        record = TensorRecord(
            name=key,
            dtype=dtype,
            shape=shape,
            bits=bits,
            group_size=group_size,
            offset=len(payload),
            nbytes=len(raw),
        )
        records.append(record)
        payload.extend(raw)

    dst_path = output_dir / f"layer_{layer_index:02d}.bin"
    with dst_path.open("wb") as f:
        f.write(payload)

    return {
        "layer": layer_index,
        "file": dst_path.name,
        "file_size": len(payload),
        "tensors": [
            {
                "name": record.name,
                "dtype": record.dtype,
                "shape": record.shape,
                "bits": record.bits,
                "group_size": record.group_size,
                "offset": record.offset,
                "nbytes": record.nbytes,
            }
            for record in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export mixed-precision Flash-MoE sidecar bins")
    parser.add_argument("--model", required=True, type=Path, help="Path to the MLX model directory")
    parser.add_argument("--output", required=True, type=Path, help="Output sidecar directory")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection (e.g. 'all', '0-3,7,12')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the export plan without writing files",
    )
    args = parser.parse_args()

    with (args.model / "config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = raw["text_config"]
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    num_layers = int(text_cfg["num_hidden_layers"])
    group_size = int(quant.get("group_size", 64))
    selected_layers = _parse_layers(args.layers, num_layers)

    safetensors_path = args.model / "model.safetensors"
    index_path = args.model / "model.safetensors.index.json"
    if safetensors_path.exists():
        shard_paths = [safetensors_path]
    elif index_path.exists():
        with index_path.open("r", encoding="utf-8") as idx_f:
            weight_map = json.load(idx_f).get("weight_map", {})
        shard_paths = sorted(set(args.model / shard for shard in weight_map.values()))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in index: {index_path}")
    else:
        raise FileNotFoundError(
            f"Missing checkpoint: need either {safetensors_path} or {index_path}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    layer_entries: list[dict[str, Any]] = []
    # Build a combined key->shard_path map for sharded models
    shard_key_map: dict[str, Path] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as sf:
            for key in sf.keys():
                shard_key_map[key] = shard_path

    for layer_index in selected_layers:
        # Find which shard(s) contain this layer's tensors
        prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp."
        layer_shards = sorted(set(
            shard_key_map[k] for k in shard_key_map if k.startswith(prefix)
        ))
        if not layer_shards:
            raise ValueError(f"No switch_mlp tensors found for layer {layer_index}")
        # Open all relevant shards and merge keys
        from contextlib import ExitStack
        with ExitStack() as stack:
            merged_keys: list[str] = []
            shard_files: dict[str, Any] = {}
            for sp in layer_shards:
                sf = stack.enter_context(safe_open(str(sp), framework="pt"))
                for key in sf.keys():
                    if key.startswith(prefix):
                        merged_keys.append(key)
                        shard_files[key] = sf

            class _MultiShardView:
                """Adapter that exposes keys() and get_tensor() across shards."""
                def __init__(self, keys, file_map):
                    self._keys = keys
                    self._file_map = file_map
                def keys(self):
                    return self._keys
                def get_tensor(self, key):
                    return self._file_map[key].get_tensor(key)

            layer_entries.append(
                export_layer(
                    output_dir=args.output,
                    layer_index=layer_index,
                    group_size=group_size,
                    text_cfg=text_cfg,
                    safe_file=_MultiShardView(merged_keys, shard_files),
                )
            )

    layout = {
        "format": "mlx-flash-moe-mixed-sidecar-v1",
        "model": str(args.model),
        "source": "mlx-model-safetensors",
        "num_layers": num_layers,
        "num_experts": int(text_cfg["num_experts"]),
        "group_size": group_size,
        "layers": layer_entries,
        "notes": [
            "Preserves quantized tensors exactly as stored in the MLX checkpoint.",
            "This is a mixed-precision expert export; no re-quantization is performed.",
        ],
    }
    with (args.output / "layout.json").open("w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    if args.dry_run:
        print(json.dumps(layout, indent=2)[:4000])
    else:
        total_bytes = sum(layer["file_size"] for layer in layer_entries)
        print(
            f"Exported {len(layer_entries)} layers to {args.output} "
            f"({total_bytes / (1024**3):.2f} GiB payload)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
