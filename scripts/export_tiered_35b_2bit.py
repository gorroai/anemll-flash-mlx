#!/usr/bin/env python3
"""
export_tiered_35b_2bit.py

Convert the mixed 35B tiered expert pack into a uniform 2-bit expert directory
that matches the current MLX-MoE loader/runtime layout.

The input tiered pack may contain a mix of 2-bit and 4-bit experts per layer.
The output directory contains uniform 2-bit experts only:

  layer_XX.bin = num_experts * 2-bit expert size

Usage:
  .env/bin/python mlx/export_tiered_35b_2bit.py \
    --model /Users/anemll/Models/Qwen3.5/mlx-Qwen3.5-35B-A3B-4bit \
    --source /Users/anemll/Library/Containers/flashmoe.anemll.com/Data/Documents/qwen3.5-35b-a3b-tiered/packed_experts_tiered \
    --output /Users/anemll/Library/Containers/flashmoe.anemll.com/Data/Documents/qwen3.5-35b-a3b-mlx-2bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flash_moe_mlx.expert_io import ExpertGeometry


GROUP_SIZE = 64


@dataclass(frozen=True)
class TieredManifest:
    num_layers: int
    num_experts: int
    expert_size_4bit: int
    expert_size_2bit: int
    threshold: float
    layers: dict[str, dict]


def parse_layers(spec: str | None, num_layers: int) -> list[int]:
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


def bf16_to_f32(bf16_u16: np.ndarray) -> np.ndarray:
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def unpack_4bit(packed: np.ndarray) -> np.ndarray:
    shape = packed.shape
    flat = packed.ravel()
    out = np.empty(flat.size * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))


def pack_2bit(vals: np.ndarray) -> np.ndarray:
    shape = vals.shape
    if shape[-1] % 16 != 0:
        raise ValueError(f"Last dim {shape[-1]} is not divisible by 16")
    n_packed = shape[-1] // 16
    flat = vals.reshape(-1, shape[-1])
    out = np.zeros((flat.shape[0], n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(shape[:-1] + (n_packed,))


def requantize_projection_4bit_to_2bit(
    packed_4bit: np.ndarray,
    scales_bf16: np.ndarray,
    biases_bf16: np.ndarray,
    out_dim: int,
    in_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_groups = in_dim // GROUP_SIZE
    vals_4bit = unpack_4bit(packed_4bit)
    if vals_4bit.shape != (out_dim, in_dim):
        raise ValueError(f"Unexpected unpacked weight shape: {vals_4bit.shape}, expected {(out_dim, in_dim)}")

    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)
    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    dequant = vals_grouped * scales_f32[:, :, None] + biases_f32[:, :, None]

    f_min = dequant.min(axis=2, keepdims=True)
    f_max = dequant.max(axis=2, keepdims=True)
    s2 = (f_max - f_min) / 3.0
    b2 = f_min
    s2_safe = np.where(s2 == 0.0, 1.0, s2)
    vals_2bit = np.clip(np.round((dequant - b2) / s2_safe), 0, 3).astype(np.uint8)

    packed_2bit = pack_2bit(vals_2bit.reshape(out_dim, in_dim))
    new_scales_bf16 = f32_to_bf16(s2.squeeze(axis=2).astype(np.float32))
    new_biases_bf16 = f32_to_bf16(b2.squeeze(axis=2).astype(np.float32))
    return packed_2bit, new_scales_bf16, new_biases_bf16


def convert_4bit_expert_blob_to_2bit(expert_blob: bytes, src_geom: ExpertGeometry, dst_geom: ExpertGeometry) -> bytes:
    if len(expert_blob) != src_geom.expert_size:
        raise ValueError(f"Expected {src_geom.expert_size} bytes, got {len(expert_blob)}")

    output = bytearray(dst_geom.expert_size)
    projections = [
        (
            "gate",
            src_geom.moe_intermediate_size,
            src_geom.hidden_size,
            src_geom.gate_weight_offset,
            src_geom.gate_scale_offset,
            src_geom.gate_bias_offset,
            dst_geom.gate_weight_offset,
            dst_geom.gate_scale_offset,
            dst_geom.gate_bias_offset,
        ),
        (
            "up",
            src_geom.moe_intermediate_size,
            src_geom.hidden_size,
            src_geom.up_weight_offset,
            src_geom.up_scale_offset,
            src_geom.up_bias_offset,
            dst_geom.up_weight_offset,
            dst_geom.up_scale_offset,
            dst_geom.up_bias_offset,
        ),
        (
            "down",
            src_geom.hidden_size,
            src_geom.moe_intermediate_size,
            src_geom.down_weight_offset,
            src_geom.down_scale_offset,
            src_geom.down_bias_offset,
            dst_geom.down_weight_offset,
            dst_geom.down_scale_offset,
            dst_geom.down_bias_offset,
        ),
    ]

    for _name, out_dim, in_dim, w_off_4, s_off_4, b_off_4, w_off_2, s_off_2, b_off_2 in projections:
        src_weight_cols = in_dim // src_geom.values_per_uint32
        src_groups = in_dim // src_geom.group_size
        w_end = w_off_4 + out_dim * src_weight_cols * 4
        s_end = s_off_4 + out_dim * src_groups * 2
        b_end = b_off_4 + out_dim * src_groups * 2

        packed_4bit = np.frombuffer(expert_blob[w_off_4:w_end], dtype=np.uint32).reshape(out_dim, src_weight_cols)
        scales_bf16 = np.frombuffer(expert_blob[s_off_4:s_end], dtype=np.uint16).reshape(out_dim, src_groups)
        biases_bf16 = np.frombuffer(expert_blob[b_off_4:b_end], dtype=np.uint16).reshape(out_dim, src_groups)

        packed_2bit, new_scales, new_biases = requantize_projection_4bit_to_2bit(
            packed_4bit=packed_4bit,
            scales_bf16=scales_bf16,
            biases_bf16=biases_bf16,
            out_dim=out_dim,
            in_dim=in_dim,
        )

        output[w_off_2 : w_off_2 + packed_2bit.nbytes] = packed_2bit.tobytes()
        output[s_off_2 : s_off_2 + new_scales.nbytes] = new_scales.tobytes()
        output[b_off_2 : b_off_2 + new_biases.nbytes] = new_biases.tobytes()

    return bytes(output)


def load_model_geometry(model_dir: Path) -> ExpertGeometry:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = raw["text_config"]
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    return ExpertGeometry(
        hidden_size=int(text_cfg["hidden_size"]),
        moe_intermediate_size=int(text_cfg["moe_intermediate_size"]),
        num_experts=int(text_cfg["num_experts"]),
        group_size=int(quant.get("group_size", 64)),
        bits=2,
        mode=str(quant.get("mode", "affine")),
    )


def load_model_layer_count(model_dir: Path) -> int:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = raw["text_config"]
    return int(text_cfg["num_hidden_layers"])


def load_tiered_manifest(source_dir: Path) -> TieredManifest:
    path = source_dir / "tiered_manifest.json"
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return TieredManifest(
        num_layers=int(raw["num_layers"]),
        num_experts=int(raw["num_experts"]),
        expert_size_4bit=int(raw["expert_size_4bit"]),
        expert_size_2bit=int(raw["expert_size_2bit"]),
        threshold=float(raw.get("threshold", 0.0)),
        layers=dict(raw["layers"]),
    )


def export_layer(
    layer_idx: int,
    source_dir: Path,
    manifest: TieredManifest,
    src_geom: ExpertGeometry,
    dst_geom: ExpertGeometry,
    output_dir: Path,
) -> tuple[int, int, int, float]:
    layer_key = str(layer_idx)
    layer_info = manifest.layers[layer_key]
    expert_entries = layer_info["experts"]
    if len(expert_entries) != dst_geom.num_experts:
        raise ValueError(
            f"Layer {layer_idx} has {len(expert_entries)} experts in manifest, expected {dst_geom.num_experts}"
        )

    src_path = source_dir / f"layer_{layer_idx:02d}.bin"
    dst_path = output_dir / f"layer_{layer_idx:02d}.bin"
    expected_src_size = int(layer_info["file_size"])
    actual_src_size = src_path.stat().st_size
    if actual_src_size != expected_src_size:
        raise ValueError(
            f"Layer {layer_idx:02d} size mismatch: manifest says {expected_src_size}, file is {actual_src_size}"
        )

    output_size = dst_geom.layer_file_size
    fd_out = os.open(dst_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, output_size)

    bit2_count = 0
    bit4_count = 0
    t0 = time.time()
    with open(src_path, "rb") as fin:
        for expert_idx, expert_info in enumerate(expert_entries):
            src_offset = int(expert_info["offset"])
            src_size = int(expert_info["size"])
            bits = int(expert_info["bits"])
            expert_blob = fin.read(0)  # keeps file object live for type checkers
            del expert_blob
            chunk = os.pread(fin.fileno(), src_size, src_offset)
            if len(chunk) != src_size:
                raise IOError(
                    f"Short read on layer {layer_idx} expert {expert_idx}: "
                    f"expected {src_size} bytes, got {len(chunk)}"
                )

            dst_offset = expert_idx * dst_geom.expert_size
            if bits == 2:
                if src_size != dst_geom.expert_size:
                    raise ValueError(
                        f"Layer {layer_idx} expert {expert_idx}: 2-bit source size {src_size} does not match "
                        f"MLX 2-bit expert size {dst_geom.expert_size}"
                    )
                os.pwrite(fd_out, chunk, dst_offset)
                bit2_count += 1
            elif bits == 4:
                if src_size != src_geom.expert_size:
                    raise ValueError(
                        f"Layer {layer_idx} expert {expert_idx}: 4-bit source size {src_size} does not match "
                        f"expected 4-bit expert size {src_geom.expert_size}"
                    )
                converted = convert_4bit_expert_blob_to_2bit(chunk, src_geom, dst_geom)
                if len(converted) != dst_geom.expert_size:
                    raise AssertionError(
                        f"Converted expert has size {len(converted)}; expected {dst_geom.expert_size}"
                    )
                os.pwrite(fd_out, converted, dst_offset)
                bit4_count += 1
            else:
                raise ValueError(f"Unsupported expert bits={bits} at layer {layer_idx} expert {expert_idx}")

    os.close(fd_out)
    elapsed = time.time() - t0
    return bit2_count, bit4_count, output_size, elapsed


def write_layout(output_dir: Path, src_manifest: TieredManifest, dst_geom: ExpertGeometry, source_dir: Path, model_dir: Path) -> None:
    layout = {
        "source_dir": str(source_dir),
        "model_dir": str(model_dir),
        "source_format": "tiered-mixed-4bit-2bit",
        "output_format": "uniform-2bit-affine",
        "num_layers": src_manifest.num_layers,
        "num_experts": src_manifest.num_experts,
        "expert_size": dst_geom.expert_size,
        "layer_file_size": dst_geom.layer_file_size,
        "hidden_size": dst_geom.hidden_size,
        "moe_intermediate_size": dst_geom.moe_intermediate_size,
        "group_size": dst_geom.group_size,
        "bits": dst_geom.bits,
        "mode": dst_geom.mode,
        "threshold": src_manifest.threshold,
    }
    with (output_dir / "layout.json").open("w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export tiered 35B expert tensors into a uniform 2-bit layout compatible with MLX-MoE"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="MLX model directory containing config.json for the target geometry",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Tiered expert directory containing tiered_manifest.json and layer_XX.bin files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for uniform 2-bit layers (default: SOURCE/packed_experts_2bit_mlx)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layer spec like "all", "0-4", or "0,5,10" (default: all)',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source/target geometry and report what would be written without creating output files",
    )
    parser.add_argument(
        "--smoke-layer",
        type=int,
        default=None,
        help="Export only this layer (shortcut for a smoke test)",
    )
    args = parser.parse_args()

    model_dir = args.model
    source_dir = args.source
    output_dir = args.output if args.output is not None else source_dir / "packed_experts_2bit_mlx"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_manifest = load_tiered_manifest(source_dir)
    dst_geom = load_model_geometry(model_dir)
    model_num_layers = load_model_layer_count(model_dir)
    src_geom = ExpertGeometry(
        hidden_size=dst_geom.hidden_size,
        moe_intermediate_size=dst_geom.moe_intermediate_size,
        num_experts=dst_geom.num_experts,
        group_size=dst_geom.group_size,
        bits=4,
        mode=dst_geom.mode,
    )
    src_geom.validate()
    dst_geom.validate()

    if src_manifest.num_layers != model_num_layers or src_manifest.num_experts != dst_geom.num_experts:
        raise ValueError(
            f"Tiered source dims {src_manifest.num_layers}x{src_manifest.num_experts} do not match "
            f"MLX target dims {model_num_layers}x{dst_geom.num_experts}"
        )

    layers = [args.smoke_layer] if args.smoke_layer is not None else parse_layers(args.layers, model_num_layers)
    if not layers:
        raise ValueError("No layers selected")

    print(f"Model:       {model_dir}")
    print(f"Source:      {source_dir}")
    print(f"Output:      {output_dir}")
    print(f"Layers:      {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Source bits:  mixed 4/2-bit tiered pack")
    print(f"Target bits:  2-bit MLX-compatible affine")
    print(f"Expert size:  {dst_geom.expert_size:,} bytes")
    print(f"Layer size:   {dst_geom.layer_file_size:,} bytes")

    total_bytes = len(layers) * dst_geom.layer_file_size
    print(f"Total output: {total_bytes / (1024 ** 3):.2f} GiB")

    if args.dry_run:
        for layer_idx in layers:
            layer_key = str(layer_idx)
            expert_entries = src_manifest.layers[layer_key]["experts"]
            bit2_count = sum(1 for e in expert_entries if int(e["bits"]) == 2)
            bit4_count = sum(1 for e in expert_entries if int(e["bits"]) == 4)
            print(
                f"  Layer {layer_idx:02d}: {len(expert_entries)} experts "
                f"({bit2_count} x 2-bit, {bit4_count} x 4-bit)"
            )
        return 0

    write_layout(output_dir, src_manifest, dst_geom, source_dir, model_dir)

    total_written = 0
    t_start = time.time()
    for layer_idx in layers:
        print(f"=== Layer {layer_idx:02d} ===")
        bit2_count, bit4_count, output_size, elapsed = export_layer(
            layer_idx=layer_idx,
            source_dir=source_dir,
            manifest=src_manifest,
            src_geom=src_geom,
            dst_geom=dst_geom,
            output_dir=output_dir,
        )
        total_written += output_size
        throughput = output_size / elapsed / (1024 ** 3) if elapsed > 0 else float("inf")
        print(
            f"  exported {output_size / (1024 ** 3):.2f} GiB in {elapsed:.1f}s "
            f"({throughput:.2f} GiB/s) | 2-bit experts={bit2_count} 4-bit experts={bit4_count}"
        )

    total_elapsed = time.time() - t_start
    print("\nDONE")
    print(f"  output:      {output_dir}")
    print(f"  layers:      {len(layers)}")
    print(f"  bytes:       {total_written:,}")
    print(f"  elapsed:     {total_elapsed:.1f}s")
    print(f"  throughput:  {total_written / total_elapsed / (1024 ** 3):.2f} GiB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
