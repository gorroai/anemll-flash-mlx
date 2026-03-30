#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flash_moe_mlx import (
    collect_slot_bank_stats,
    decode_incremental,
    generate_with_stats,
    load_model_bundle,
    set_routing_sample_callback,
    set_sparse_moe_tail_compile,
)


def _green(text: str) -> str:
    if not sys.stderr.isatty():
        return text
    return f"\033[32m{text}\033[0m"


def _orange(text: str) -> str:
    if not sys.stderr.isatty():
        return text
    return f"\033[38;5;208m{text}\033[0m"


def _prefix() -> str:
    return f"[{_orange('flash-moe-mlx')}]"


class RoutingSampleWriter:
    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._f = self.path.open("ab" if append else "wb")
        self.samples = 0

    def write(self, layer_index: int, hidden: object, expert_indices: object) -> None:
        hidden_mv = memoryview(hidden)
        expert_mv = memoryview(expert_indices)
        k = int(expert_mv.nbytes // 4)
        self._f.write(struct.pack("<ii", int(layer_index), k))
        self._f.write(hidden_mv.cast("B"))
        self._f.write(expert_mv.cast("B"))
        self.samples += 1

    def close(self) -> None:
        self._f.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="35B MLX dense path with native streamed experts",
    )
    parser.add_argument("--mlx", required=True, help="Path to the raw MLX model directory")
    parser.add_argument("--experts", help="Path to the packed_experts directory")
    parser.add_argument(
        "--2-bit",
        "--2bit",
        dest="expert_bits_2",
        action="store_true",
        help="Interpret external packed experts as 2-bit affine experts",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Load routed experts from the MLX checkpoint instead of packed_experts",
    )
    parser.add_argument(
        "--resident-flash",
        action="store_true",
        help="Preload packed_experts into RAM and serve routed experts from memory instead of SSD",
    )
    parser.add_argument(
        "--resident-pread-mlx",
        "--resident-pread",
        "--resident-bank-index",
        dest="resident_pread_mlx",
        action="store_true",
        help="pread packed_experts once at startup into a persistent full MLX expert bank and index it directly by expert id",
    )
    parser.add_argument(
        "--resident-rebind",
        action="store_true",
        help="pread packed_experts once at startup, then rebind the selected K expert tensors each token",
    )
    parser.add_argument(
        "--resident-copy-k",
        action="store_true",
        help="pread packed_experts once at startup, then copy the selected K experts into stable executor buffers each token",
    )
    parser.add_argument(
        "--bypass-routed-mlp",
        action="store_true",
        help="Bypass routed expert compute but keep router and shared expert active",
    )
    parser.add_argument(
        "--slot-bank",
        type=int,
        default=0,
        help="Keep a stable per-layer expert slot bank of this size and reload only misses; can also be combined with --resident-pread-mlx to source misses from the resident packed bank",
    )
    parser.add_argument(
        "--slot-bank-native",
        action="store_true",
        help="Use native C slot ownership/victim selection for --slot-bank",
    )
    parser.add_argument(
        "--prefetch-temporal",
        action="store_true",
        help="After each token, prefetch the same routed experts into the next slot-bank step as a real temporal one-step predictor",
    )
    parser.add_argument(
        "--compiled-tail",
        action="store_true",
        help="Compile the final sparse-block shared-expert tail: routed_y + sigmoid(gate) * shared_y",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated text incrementally as tokens are produced (this is also the current default behavior)",
    )
    parser.add_argument(
        "--collect-routing",
        type=Path,
        help="Write binary routing samples for predictor training: int32 layer_idx, int32 K, float32[hidden], int32[K]",
    )
    parser.add_argument(
        "--append-routing",
        action="store_true",
        help="Append to --collect-routing instead of overwriting it",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum new tokens to decode")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override routed experts per token (1..model config num_experts_per_tok)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; use 0 for greedy decode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument(
        "--cache-io-split",
        type=int,
        default=1,
        help="Split each routed expert pread into N page-aligned chunks",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    resident_modes = sum(
        (
            args.resident,
            args.resident_flash,
            args.resident_pread_mlx,
            args.resident_rebind,
            args.resident_copy_k,
        )
    )
    if resident_modes > 1:
        parser.error(
            "--resident, --resident-flash, --resident-pread-mlx, --resident-rebind, and --resident-copy-k are mutually exclusive"
        )
    if args.prefetch_temporal and not args.slot_bank:
        parser.error("--prefetch-temporal currently requires --slot-bank")
    if not args.resident and not args.experts:
        parser.error("--experts is required unless --resident is set")
    bundle = load_model_bundle(
        mlx_model_dir=args.mlx,
        experts_dir=args.experts,
        routed_top_k=args.k,
        cache_io_split=args.cache_io_split,
        expert_bits=2 if args.expert_bits_2 else None,
        resident_experts=args.resident,
        resident_flash=args.resident_flash,
        resident_pread_mlx=args.resident_pread_mlx,
        resident_rebind=args.resident_rebind,
        resident_copy_k=args.resident_copy_k,
        slot_bank_size=args.slot_bank,
        slot_bank_native=args.slot_bank_native,
        bypass_routed_mlp=args.bypass_routed_mlp,
    )
    if args.compiled_tail:
        set_sparse_moe_tail_compile(bundle.model, True)
    routing_writer: RoutingSampleWriter | None = None
    if args.collect_routing is not None:
        routing_writer = RoutingSampleWriter(args.collect_routing, append=args.append_routing)
        set_routing_sample_callback(bundle.model, routing_writer.write)

    effective_k = args.k or bundle.config.num_experts_per_tok
    if args.resident:
        expert_source = "resident-mlx"
    elif args.resident_copy_k:
        expert_source = f"resident-copy-k:{args.experts}"
    elif args.resident_rebind:
        expert_source = f"resident-rebind:{args.experts}"
    elif args.resident_pread_mlx:
        expert_source = f"resident-pread-mlx:{args.experts}"
    elif args.resident_flash:
        expert_source = f"resident-flash:{args.experts}"
    else:
        expert_source = args.experts
    if args.slot_bank:
        expert_source = f"{expert_source}|slot-bank:{args.slot_bank}"
        if args.slot_bank_native:
            expert_source = f"{expert_source}|native"
        if args.prefetch_temporal:
            expert_source = f"{expert_source}|prefetch-temporal"
    if args.expert_bits_2:
        expert_source = f"{expert_source}|2bit"
    if args.bypass_routed_mlp:
        expert_source = f"{expert_source}|bypass-routed-mlp"
    if args.compiled_tail:
        expert_source = f"{expert_source}|compiled-tail"
    print(
        f"{_prefix()} loaded model={args.mlx} experts={expert_source} "
        f"layers={bundle.config.num_hidden_layers} k={effective_k} "
        f"expert_size={bundle.expert_geometry.expert_size} bytes",
        file=sys.stderr,
    )

    generated_ids: list[int] = []
    rendered_ids: list[int] = []
    emitted = ""

    def handle_token(token_id: int) -> None:
        nonlocal emitted
        generated_ids.append(token_id)
        rendered_ids.append(token_id)
        decoded = decode_incremental(bundle.tokenizer, rendered_ids)
        if args.stream:
            delta = decoded[len(emitted) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        emitted = decoded

    try:
        stats = generate_with_stats(
            bundle,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            on_token=handle_token,
            slot_bank_temporal_prefetch=args.prefetch_temporal,
        )
    finally:
        if routing_writer is not None:
            set_routing_sample_callback(bundle.model, None)
            routing_writer.close()

    if emitted:
        if not args.stream:
            sys.stdout.write(emitted)
        sys.stdout.write("\n")
    print(f"{_prefix()} decode_tps={_green(f'{stats.decode_tokens_per_second:.2f}')}", file=sys.stderr)
    print(
        f"{_prefix()} prefill_tokens={stats.prompt_tokens} "
        f"prefill_s={stats.prefill_seconds:.3f} "
        f"prefill_tps={stats.prefill_tokens_per_second:.2f}",
        file=sys.stderr,
    )
    print(
        f"{_prefix()} decode_tokens={stats.generated_tokens} "
        f"decode_s={stats.decode_seconds:.3f}",
        file=sys.stderr,
    )
    if args.slot_bank:
        slot_stats = collect_slot_bank_stats(bundle.model)
        print(
            f"{_prefix()} slot_bank={args.slot_bank} "
            f"layers={int(slot_stats['layers'])} "
            f"calls={int(slot_stats['calls'])} "
            f"requests={int(slot_stats['requests'])} "
            f"misses={int(slot_stats['misses'])}",
            file=sys.stderr,
        )
        print(
            f"{_prefix()} hit_rate={slot_stats['hit_rate']:.4f} "
            f"full_hit_rate={slot_stats['full_hit_rate']:.4f}",
            file=sys.stderr,
        )
    if routing_writer is not None:
        print(
            f"{_prefix()} routing_samples={routing_writer.samples} "
            f"routing_file={routing_writer.path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
