#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flash_moe_mlx.expert_io import ExpertGeometry, NativeExpertLoader, ResidentFlashExpertLoader, unpack_expert_slot
from flash_moe_mlx.model import ModelArgs
from flash_moe_mlx.upstream_switch_layers import QuantizedSwitchGLUExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbench the slot-bank miss commit path with stage-fenced timings",
    )
    parser.add_argument("--mlx", required=True, help="Path to the raw MLX model directory")
    parser.add_argument("--experts", required=True, help="Path to the packed_experts directory")
    parser.add_argument("--layer", type=int, default=0, help="Sparse layer index to benchmark")
    parser.add_argument("--misses", type=int, default=4, help="How many expert misses to commit per iteration")
    parser.add_argument("--slot-bank", type=int, default=64, help="Stable slot-bank size")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cache-io-split", type=int, default=1, help="Native pread split fanout")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["native", "flash"],
        choices=["native", "flash"],
        help="Commit sources to benchmark in order",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="MLX device used for slot-bank commit",
    )
    parser.add_argument(
        "--2-bit",
        "--2bit",
        dest="expert_bits_2",
        action="store_true",
        help="Interpret external packed experts as 2-bit affine experts",
    )
    return parser.parse_args()


def _device(name: str):
    return mx.gpu if name == "gpu" else mx.cpu


def _prepare_batches(model_args: ModelArgs, misses: int, slot_bank_size: int, total: int, seed: int):
    rng = np.random.default_rng(seed)
    expert_batches = [
        rng.choice(model_args.num_experts, size=misses, replace=False).astype(np.int32)
        for _ in range(total)
    ]
    slot_batches = [
        rng.choice(slot_bank_size, size=misses, replace=False).astype(np.int32)
        for _ in range(total)
    ]
    return expert_batches, slot_batches


def _geometry_for_run(model_args: ModelArgs, expert_bits: int) -> ExpertGeometry:
    geometry = ExpertGeometry(
        hidden_size=model_args.hidden_size,
        moe_intermediate_size=model_args.moe_intermediate_size,
        num_experts=model_args.num_experts,
        group_size=model_args.quant_group_size,
        bits=expert_bits,
        mode=model_args.quant_mode,
    )
    geometry.validate()
    return geometry


def _make_loader(
    source: str,
    experts_dir: Path,
    model_args: ModelArgs,
    geometry: ExpertGeometry,
    misses: int,
    cache_io_split: int,
):
    if source == "native":
        return NativeExpertLoader(
            experts_dir=experts_dir,
            num_layers=model_args.num_hidden_layers,
            geometry=geometry,
            max_k=misses,
            cache_io_split=cache_io_split,
        )
    if source == "flash":
        return ResidentFlashExpertLoader(
            experts_dir=experts_dir,
            num_layers=model_args.num_hidden_layers,
            geometry=geometry,
            max_k=misses,
        )
    raise ValueError(f"Unsupported source: {source}")


def _bench_source_loop(
    source: str,
    device,
    experts_dir: Path,
    model_args: ModelArgs,
    geometry: ExpertGeometry,
    misses: int,
    slot_bank_size: int,
    cache_io_split: int,
    expert_batches,
    slot_batches,
    warmup: int,
) -> dict[str, float]:
    loader = _make_loader(source, experts_dir, model_args, geometry, misses, cache_io_split)
    load_s = 0.0
    unpack_s = 0.0
    commit_s = 0.0
    forward_s = 0.0
    total_s = 0.0
    measured = 0
    try:
        with mx.stream(device):
            executor = QuantizedSwitchGLUExecutor(
                model_args.hidden_size,
                model_args.moe_intermediate_size,
                slot_bank_size,
                group_size=model_args.quant_group_size,
                bits=geometry.bits,
                mode=geometry.mode,
            )
            mx.eval(*executor.resident_buffers())
            mx.synchronize()
            rng = np.random.default_rng(cli.seed)
            x = mx.array(
                rng.standard_normal((1, 1, model_args.hidden_size), dtype=np.float32),
                dtype=mx.float32,
            )
            uniform_scores = mx.full((1, 1, misses), 1.0 / float(misses), dtype=mx.float32)
            mx.eval(x, uniform_scores)
            mx.synchronize()

            for index, (expert_ids, slot_ids) in enumerate(zip(expert_batches, slot_batches)):
                t0 = time.perf_counter()
                slot_buffers = loader.load_layer(cli.layer, expert_ids.tolist())
                t1 = time.perf_counter()
                experts = [unpack_expert_slot(slot_buffer, geometry) for slot_buffer in slot_buffers]
                t2 = time.perf_counter()
                executor.load_quantized_views_into_slots(slot_ids.tolist(), experts)
                mx.eval(*executor.resident_buffers())
                mx.synchronize()
                t3 = time.perf_counter()
                local_indices = mx.array(slot_ids.reshape(1, 1, -1), dtype=mx.int32)
                routed_y = executor(x, local_indices)
                combined = (routed_y * uniform_scores[..., None]).sum(axis=-2)
                mx.eval(combined)
                mx.synchronize()
                t4 = time.perf_counter()

                if index >= warmup:
                    measured += 1
                    load_s += t1 - t0
                    unpack_s += t2 - t1
                    commit_s += t3 - t2
                    forward_s += t4 - t3
                    total_s += t4 - t0
    finally:
        loader.close()

    if measured == 0:
        raise ValueError("warmup must be smaller than total iterations")

    bytes_per_iter = geometry.expert_size * misses
    mib_per_iter = bytes_per_iter / float(1024 * 1024)
    load_ms = (load_s / measured) * 1000.0
    unpack_ms = (unpack_s / measured) * 1000.0
    commit_ms = (commit_s / measured) * 1000.0
    forward_ms = (forward_s / measured) * 1000.0
    total_ms = (total_s / measured) * 1000.0
    load_gib_s = 0.0 if load_s <= 0 else (bytes_per_iter * measured) / load_s / float(1024**3)
    total_gib_s = 0.0 if total_s <= 0 else (bytes_per_iter * measured) / total_s / float(1024**3)
    return {
        "load_ms": load_ms,
        "unpack_ms": unpack_ms,
        "commit_ms": commit_ms,
        "forward_ms": forward_ms,
        "total_ms": total_ms,
        "mib_per_iter": mib_per_iter,
        "load_gib_s": load_gib_s,
        "total_gib_s": total_gib_s,
    }


def main() -> int:
    global cli
    cli = parse_args()
    model_dir = Path(cli.mlx)
    experts_dir = Path(cli.experts)
    model_args = ModelArgs.from_model_dir(model_dir)
    expert_bits = 2 if cli.expert_bits_2 else model_args.quant_bits
    geometry = _geometry_for_run(model_args, expert_bits)
    if cli.slot_bank < cli.misses:
        raise ValueError(f"--slot-bank must be >= --misses ({cli.misses})")

    total = cli.warmup + cli.iters
    expert_batches, slot_batches = _prepare_batches(
        model_args,
        cli.misses,
        cli.slot_bank,
        total,
        cli.seed,
    )
    device = _device(cli.device)

    print(
        f"[commit-bench] layer={cli.layer} misses={cli.misses} slot_bank={cli.slot_bank} "
        f"warmup={cli.warmup} iters={cli.iters} device={cli.device} "
        f"expert_bits={expert_bits} expert_size={geometry.expert_size}",
        flush=True,
    )

    for source in cli.sources:
        stats = _bench_source_loop(
            source=source,
            device=device,
            experts_dir=experts_dir,
            model_args=model_args,
            geometry=geometry,
            misses=cli.misses,
            slot_bank_size=cli.slot_bank,
            cache_io_split=cli.cache_io_split,
            expert_batches=expert_batches,
            slot_batches=slot_batches,
            warmup=cli.warmup,
        )
        print(
            f"[commit-bench:{source}] load_ms={stats['load_ms']:.3f} "
            f"unpack_ms={stats['unpack_ms']:.3f} commit_ms={stats['commit_ms']:.3f} "
            f"forward_ms={stats['forward_ms']:.3f} "
            f"total_ms={stats['total_ms']:.3f} "
            f"mib_per_iter={stats['mib_per_iter']:.2f} "
            f"load_gib_s={stats['load_gib_s']:.2f} total_gib_s={stats['total_gib_s']:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
