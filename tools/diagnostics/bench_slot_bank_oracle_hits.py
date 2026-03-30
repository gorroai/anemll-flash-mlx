#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flash_moe_mlx import (
    collect_slot_bank_stage_stats,
    collect_slot_bank_stats,
    decode_incremental,
    eval_slot_bank_buffers,
    enable_slot_bank_stage_timing,
    list_sparse_moe_layer_indices,
    load_model_bundle,
    prefetch_slot_banks,
    prime_slot_banks,
    reset_slot_bank_state,
    reset_slot_bank_stats,
    set_slot_bank_direct_contiguous_hit,
    set_slot_bank_device_hit_lookup,
)
from flash_moe_mlx.model import _sample_next_token, _set_routing_trace_callback


class DecodeTraceCollector:
    def __init__(self) -> None:
        self.phase = "prefill"
        self.records: dict[str, dict[int, list[list[int]]]] = {
            "prefill": collections.defaultdict(list),
            "decode": collections.defaultdict(list),
        }

    def callback(self, layer_index: int, expert_indices: np.ndarray, expert_scores: np.ndarray) -> None:
        del expert_scores
        self.records[self.phase][layer_index].append(
            expert_indices.reshape(-1).astype(np.int32, copy=False).tolist()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the real slot-bank path with oracle all-hit priming outside the timed decode loop",
    )
    parser.add_argument("--mlx", required=True, help="Path to the raw MLX model directory")
    parser.add_argument("--experts", required=True, help="Path to the packed_experts directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=40, help="Maximum new tokens to decode")
    parser.add_argument("--k", type=int, default=4, help="Experts per token to route")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--slot-bank", type=int, default=64, help="Slot-bank size for the replay path")
    parser.add_argument(
        "--trace-mode",
        choices=("resident-pread", "slot-bank", "streamed"),
        default="resident-pread",
        help="Reference mode used to trace and replay the forced token sequence",
    )
    parser.add_argument(
        "--2-bit",
        "--2bit",
        dest="expert_bits_2",
        action="store_true",
        help="Interpret external packed experts as 2-bit affine experts",
    )
    parser.add_argument(
        "--stage-timing",
        action="store_true",
        help="Enable fenced stage timing for the oracle all-hit slot-bank replay",
    )
    parser.add_argument(
        "--device-hit-lookup",
        action="store_true",
        help="Assume the oracle replay is all-hit and keep expert->slot lookup on device without Python selector extraction",
    )
    parser.add_argument(
        "--direct-contiguous-hit",
        action="store_true",
        help="Assume oracle priming repacks each layer contiguously into slots [0..k-1] and skip lookup/index materialization",
    )
    parser.add_argument(
        "--one-step-prefetch",
        action="store_true",
        help="Replay the real slot-bank path while oracle-prefetching the next token's experts outside the timed window",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the traced generated text in real time and print replay phase progress to stderr",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional trace, timing, and locality details",
    )
    return parser.parse_args()


def _bundle_kwargs(args: argparse.Namespace, mode: str) -> dict:
    kwargs = dict(
        mlx_model_dir=args.mlx,
        experts_dir=args.experts,
        routed_top_k=args.k,
        expert_bits=2 if args.expert_bits_2 else None,
    )
    if mode == "resident-pread":
        kwargs["resident_pread_mlx"] = True
    elif mode == "slot-bank":
        kwargs["slot_bank_size"] = args.slot_bank
    elif mode == "streamed":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return kwargs


def _stream_token(tokenizer, rendered_ids: list[int], emitted: str) -> str:
    decoded = decode_incremental(tokenizer, rendered_ids)
    delta = decoded[len(emitted) :]
    if delta:
        sys.stdout.write(delta)
        sys.stdout.flush()
    return decoded


def _trace_decode_schedule(args: argparse.Namespace):
    bundle = load_model_bundle(**_bundle_kwargs(args, args.trace_mode))
    collector = DecodeTraceCollector()
    _set_routing_trace_callback(bundle.model, collector.callback)

    encoded = bundle.tokenizer.encode(args.prompt)
    prompt_ids = list(encoded.ids)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence")

    rng = np.random.default_rng(args.seed)
    cache = bundle.model.make_cache()
    logits = None
    for token_id in prompt_ids:
        collector.phase = "prefill"
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)

    generated_ids: list[int] = []
    rendered_ids: list[int] = []
    emitted = ""
    eos_token_ids = set(bundle.config.eos_token_ids)
    if args.stream:
        print("[oracle-slot-bank] phase=trace-collect", file=sys.stderr, flush=True)
    decode_start = time.perf_counter()
    if logits is not None:
        for _ in range(args.max_tokens):
            next_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
            if next_token in eos_token_ids:
                break
            generated_ids.append(next_token)
            rendered_ids.append(next_token)
            if args.stream:
                emitted = _stream_token(bundle.tokenizer, rendered_ids, emitted)
            collector.phase = "decode"
            logits = bundle.model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
    decode_seconds = time.perf_counter() - decode_start
    if args.stream and emitted:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _set_routing_trace_callback(bundle.model, None)

    decode_records = collector.records["decode"]
    sparse_layers = list_sparse_moe_layer_indices(bundle.model)
    for layer_index in sparse_layers:
        calls = decode_records.get(layer_index, [])
        if len(calls) != len(generated_ids):
            raise RuntimeError(
                f"Layer {layer_index} has {len(calls)} decode trace records for {len(generated_ids)} generated tokens"
            )

    return prompt_ids, generated_ids, decode_records, decode_seconds


def _replay_oracle_all_hit(
    args: argparse.Namespace,
    prompt_ids: list[int],
    generated_ids: list[int],
    decode_records: dict[int, list[list[int]]],
):
    bundle = load_model_bundle(**_bundle_kwargs(args, "slot-bank"))
    if args.stage_timing:
        enable_slot_bank_stage_timing(bundle.model, True)
    if args.direct_contiguous_hit:
        set_slot_bank_direct_contiguous_hit(bundle.model, True)
    if args.device_hit_lookup:
        set_slot_bank_device_hit_lookup(bundle.model, True)
    sparse_layers = list_sparse_moe_layer_indices(bundle.model)

    cache = bundle.model.make_cache()
    logits = None
    for token_id in prompt_ids:
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)

    reset_slot_bank_state(bundle.model)
    reset_slot_bank_stats(bundle.model)

    rng = np.random.default_rng(args.seed)
    parity_ok = True
    predicted_ids: list[int] = []
    if logits is not None and generated_ids:
        first_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
        predicted_ids.append(first_token)
        parity_ok = first_token == generated_ids[0]

    decode_seconds = 0.0
    for step, token_id in enumerate(generated_ids):
        experts_by_layer = {layer_index: decode_records[layer_index][step] for layer_index in sparse_layers}
        prime_slot_banks(bundle.model, experts_by_layer)
        eval_slot_bank_buffers(bundle.model)

        step_start = time.perf_counter()
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        decode_seconds += time.perf_counter() - step_start

        if step + 1 < len(generated_ids):
            next_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
            predicted_ids.append(next_token)
            if next_token != generated_ids[step + 1]:
                parity_ok = False

    slot_stats = collect_slot_bank_stats(bundle.model)
    stage_stats = collect_slot_bank_stage_stats(bundle.model) if args.stage_timing else None
    return decode_seconds, parity_ok, predicted_ids, slot_stats, stage_stats


def _replay_oracle_one_step_prefetch(
    args: argparse.Namespace,
    prompt_ids: list[int],
    generated_ids: list[int],
    decode_records: dict[int, list[list[int]]],
):
    bundle = load_model_bundle(**_bundle_kwargs(args, "slot-bank"))
    sparse_layers = list_sparse_moe_layer_indices(bundle.model)

    cache = bundle.model.make_cache()
    logits = None
    for token_id in prompt_ids:
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)

    reset_slot_bank_state(bundle.model)
    reset_slot_bank_stats(bundle.model)

    if generated_ids:
        first_prefetch = {layer_index: decode_records[layer_index][0] for layer_index in sparse_layers}
        prefetch_slot_banks(bundle.model, first_prefetch)
        eval_slot_bank_buffers(bundle.model)

    rng = np.random.default_rng(args.seed)
    parity_ok = True
    predicted_ids: list[int] = []
    if logits is not None and generated_ids:
        first_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
        predicted_ids.append(first_token)
        parity_ok = first_token == generated_ids[0]

    decode_seconds = 0.0
    for step, token_id in enumerate(generated_ids):
        step_start = time.perf_counter()
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        decode_seconds += time.perf_counter() - step_start

        if step + 1 < len(generated_ids):
            next_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
            predicted_ids.append(next_token)
            if next_token != generated_ids[step + 1]:
                parity_ok = False

            next_prefetch = {layer_index: decode_records[layer_index][step + 1] for layer_index in sparse_layers}
            prefetch_slot_banks(bundle.model, next_prefetch)
            eval_slot_bank_buffers(bundle.model)

    slot_stats = collect_slot_bank_stats(bundle.model)
    return decode_seconds, parity_ok, predicted_ids, slot_stats


def _replay_reference_inputs(
    args: argparse.Namespace,
    prompt_ids: list[int],
    generated_ids: list[int],
):
    bundle = load_model_bundle(**_bundle_kwargs(args, args.trace_mode))
    cache = bundle.model.make_cache()
    logits = None
    for token_id in prompt_ids:
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)

    rng = np.random.default_rng(args.seed)
    parity_ok = True
    predicted_ids: list[int] = []
    if logits is not None and generated_ids:
        first_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
        predicted_ids.append(first_token)
        parity_ok = first_token == generated_ids[0]

    decode_seconds = 0.0
    for step, token_id in enumerate(generated_ids):
        step_start = time.perf_counter()
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        decode_seconds += time.perf_counter() - step_start

        if step + 1 < len(generated_ids):
            next_token = _sample_next_token(logits[:, -1, :], args.temperature, rng)
            predicted_ids.append(next_token)
            if next_token != generated_ids[step + 1]:
                parity_ok = False

    return decode_seconds, parity_ok, predicted_ids


def main() -> int:
    args = parse_args()
    prompt_ids, generated_ids, decode_records, trace_decode_seconds = _trace_decode_schedule(args)
    reference_decode_seconds, reference_parity_ok, reference_predicted_ids = _replay_reference_inputs(
        args,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
    )
    oracle_decode_seconds, parity_ok, predicted_ids, slot_stats, stage_stats = _replay_oracle_all_hit(
        args,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        decode_records=decode_records,
    )
    prefetch_decode_seconds = 0.0
    prefetch_parity_ok = False
    prefetch_predicted_ids: list[int] = []
    prefetch_slot_stats = None
    if args.one_step_prefetch:
        (
            prefetch_decode_seconds,
            prefetch_parity_ok,
            prefetch_predicted_ids,
            prefetch_slot_stats,
        ) = _replay_oracle_one_step_prefetch(
            args,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            decode_records=decode_records,
        )

    trace_decode_tps = 0.0 if trace_decode_seconds <= 0.0 else len(generated_ids) / trace_decode_seconds
    reference_decode_tps = 0.0 if reference_decode_seconds <= 0.0 else len(generated_ids) / reference_decode_seconds
    prefetch_decode_tps = 0.0 if prefetch_decode_seconds <= 0.0 else len(generated_ids) / prefetch_decode_seconds
    oracle_decode_tps = 0.0 if oracle_decode_seconds <= 0.0 else len(generated_ids) / oracle_decode_seconds
    max_unique_per_layer = 0
    unique_per_layer: dict[int, int] = {}
    for layer_calls in decode_records.values():
        unique_experts = {expert_id for call in layer_calls for expert_id in call}
        unique_count = len(unique_experts)
        max_unique_per_layer = max(max_unique_per_layer, unique_count)
    for layer_index, layer_calls in decode_records.items():
        unique_per_layer[layer_index] = len({expert_id for call in layer_calls for expert_id in call})

    print(
        f"[oracle-slot-bank] prompt={args.prompt!r} generated_tokens={len(generated_ids)} "
        f"k={args.k} slot_bank={args.slot_bank} trace_mode={args.trace_mode} "
        f"expert_bits={(2 if args.expert_bits_2 else 4)} max_unique_per_layer={max_unique_per_layer}",
        flush=True,
    )
    print(
        f"[oracle-slot-bank] trace_collect_decode_tps={trace_decode_tps:.2f} "
        f"reference_replay_decode_tps={reference_decode_tps:.2f} "
        f"one_step_prefetch_decode_tps={prefetch_decode_tps:.2f} "
        f"oracle_all_hit_decode_tps={oracle_decode_tps:.2f} "
        f"delta_ms_per_token_vs_reference={(1000.0 / oracle_decode_tps - 1000.0 / reference_decode_tps) if reference_decode_tps > 0.0 and oracle_decode_tps > 0.0 else 0.0:.3f}",
        flush=True,
    )
    print(
        f"[oracle-slot-bank] reference_parity_ok={reference_parity_ok} "
        f"one_step_prefetch_parity_ok={prefetch_parity_ok} "
        f"oracle_parity_ok={parity_ok} "
        f"slot_calls={int(slot_stats['calls'])} "
        f"slot_misses={int(slot_stats['misses'])} "
        f"slot_hit_rate={slot_stats['hit_rate']:.4f} "
        f"slot_full_hit_rate={slot_stats['full_hit_rate']:.4f}",
        flush=True,
    )
    if prefetch_slot_stats is not None:
        print(
            f"[oracle-slot-bank] one_step_prefetch_slot_calls={int(prefetch_slot_stats['calls'])} "
            f"one_step_prefetch_slot_misses={int(prefetch_slot_stats['misses'])} "
            f"one_step_prefetch_slot_hit_rate={prefetch_slot_stats['hit_rate']:.4f} "
            f"one_step_prefetch_slot_full_hit_rate={prefetch_slot_stats['full_hit_rate']:.4f}",
            flush=True,
        )
    print(
        f"[oracle-slot-bank] generated_ids_preview={generated_ids[:8]} "
        f"reference_predicted_ids_preview={reference_predicted_ids[:8]} "
        f"one_step_prefetch_predicted_ids_preview={prefetch_predicted_ids[:8]} "
        f"oracle_predicted_ids_preview={predicted_ids[:8]}",
        flush=True,
    )
    if stage_stats is not None and stage_stats["calls"] > 0 and generated_ids:
        generated = float(len(generated_ids))
        print(
            f"[oracle-slot-bank:stages] calls={int(stage_stats['calls'])} "
            f"hit_calls={int(stage_stats['hit_calls'])} "
            f"miss_calls={int(stage_stats['miss_calls'])} "
            f"selector_ms_per_token={(stage_stats['selector_seconds'] * 1000.0) / generated:.3f} "
            f"lookup_ms_per_token={(stage_stats['lookup_seconds'] * 1000.0) / generated:.3f} "
            f"install_ms_per_token={(stage_stats['install_seconds'] * 1000.0) / generated:.3f} "
            f"index_ms_per_token={(stage_stats['index_seconds'] * 1000.0) / generated:.3f} "
            f"forward_ms_per_token={(stage_stats['forward_seconds'] * 1000.0) / generated:.3f} "
            f"combine_ms_per_token={(stage_stats['combine_seconds'] * 1000.0) / generated:.3f} "
            f"total_ms_per_token={(stage_stats['total_seconds'] * 1000.0) / generated:.3f}",
            flush=True,
        )
    if args.verbose:
        prompt_token_count = len(prompt_ids)
        generated = max(1, len(generated_ids))
        print(
            f"[oracle-slot-bank:verbose] prompt_tokens={prompt_token_count} "
            f"generated_tokens={len(generated_ids)} "
            f"trace_decode_s={trace_decode_seconds:.3f} "
            f"reference_decode_s={reference_decode_seconds:.3f} "
            f"one_step_prefetch_decode_s={prefetch_decode_seconds:.3f} "
            f"oracle_decode_s={oracle_decode_seconds:.3f}",
            flush=True,
        )
        print(
            f"[oracle-slot-bank:verbose] trace_ms_per_token={(trace_decode_seconds * 1000.0) / generated:.3f} "
            f"reference_ms_per_token={(reference_decode_seconds * 1000.0) / generated:.3f} "
            f"one_step_prefetch_ms_per_token={(prefetch_decode_seconds * 1000.0) / generated if prefetch_decode_seconds > 0.0 else 0.0:.3f} "
            f"oracle_ms_per_token={(oracle_decode_seconds * 1000.0) / generated:.3f}",
            flush=True,
        )
        hottest_layers = sorted(unique_per_layer.items(), key=lambda item: (-item[1], item[0]))[:8]
        print(
            "[oracle-slot-bank:verbose] top_unique_layers="
            + ",".join(f"L{layer}:{count}" for layer, count in hottest_layers),
            flush=True,
        )
        sample_layers = sorted(decode_records.keys())[:4]
        sample_bits: list[str] = []
        for layer_index in sample_layers:
            calls = decode_records[layer_index]
            if calls:
                sample_bits.append(f"L{layer_index}:{calls[0]}")
        if sample_bits:
            print(
                "[oracle-slot-bank:verbose] first_decode_experts="
                + " ".join(sample_bits),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
