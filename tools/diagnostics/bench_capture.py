#!/usr/bin/env python3
"""
bench_capture.py — Repeatable benchmark capture for Flash-MoE slot-bank inference.

What this captures
------------------
For each (variant, prompt, mode) combination, this script runs a full
generate pass through the slot-bank runtime and records:

  - **Prefill speed** (tok/s) — how fast the prompt is ingested.
  - **Decode speed** (tok/s) — steady-state token generation throughput.
  - **Slot-bank hit rate** — fraction of expert requests served from the
    resident slot bank without an SSD miss.
  - **Full-hit rate** — fraction of tokens where *all* K routed experts
    were already resident (zero misses for that token).
  - **Total misses** — number of expert loads from SSD during the run.
  - **Routing trace** (optional) — binary dump of (layer, hidden, expert_ids)
    per token, for offline locality analysis or predictor training.

The results are printed as a markdown summary table and optionally written
to a JSONL file for programmatic comparison across runs, machines, or configs.

Usage examples
--------------
  # Run the built-in prompt suite against a 4-bit model
  python3 tools/diagnostics/bench_capture.py \\
    --mlx ~/Models/mlx-Qwen3.5-35B-A3B-4bit \\
    --experts ~/Models/packed_experts \\
    --slot-bank 64

  # Custom prompts, compare baseline vs temporal prefetch
  python3 tools/diagnostics/bench_capture.py \\
    --mlx ~/Models/mlx-Qwen3.5-35B-A3B-4bit \\
    --experts ~/Models/packed_experts \\
    --prompt "What is quantum computing?" \\
    --prompt "Write a Python sort function" \\
    --compare-prefetch --slot-bank 64

  # Two variants (4-bit vs 2-bit), save results to JSONL
  python3 tools/diagnostics/bench_capture.py \\
    --mlx ~/Models/mlx-Qwen3.5-35B-A3B-4bit \\
    --experts ~/Models/packed_experts \\
    --mlx-alt ~/Models/mlx-Qwen3.5-35B-A3B-4bit \\
    --experts-alt ~/Models/packed_experts_2bit --alt-2bit \\
    --jsonl results/capture.jsonl
"""

from __future__ import annotations

import argparse
import json
import struct
import statistics
import sys
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flash_moe_mlx import (
    collect_slot_bank_stats,
    decode_incremental,
    generate_with_stats,
    load_model_bundle,
    set_routing_sample_callback,
)


DEFAULT_PROMPTS = [
    "What is Apple Neural Engine",
    "What is Apple Neural Engine? How it compares to GPU?",
    "Explain quantum computing in simple terms",
    "Write a Python function to reverse a linked list",
    "Why do airplanes generate lift",
    "Give a short history of the Roman Empire",
    "Explain the chain rule with an example",
    "Summarize the pros and cons of electric cars in JSON",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatable benchmark capture for Flash-MoE slot-bank inference: "
            "records prefill/decode throughput, slot-bank hit/miss statistics, "
            "and optionally binary routing traces for offline locality analysis "
            "or predictor training."
        ),
    )
    parser.add_argument("--label", default="bench", help="Label for this capture run")

    # Primary variant
    parser.add_argument("--mlx", required=True, help="MLX model directory")
    parser.add_argument("--experts", required=True, help="Packed experts directory")
    parser.add_argument(
        "--2-bit", "--2bit", dest="expert_bits_2", action="store_true",
        help="Interpret primary experts as 2-bit",
    )

    # Optional second variant for comparison
    parser.add_argument("--mlx-alt", help="MLX model directory for comparison variant")
    parser.add_argument("--experts-alt", help="Packed experts directory for comparison variant")
    parser.add_argument(
        "--alt-2bit", dest="alt_expert_bits_2", action="store_true",
        help="Interpret comparison experts as 2-bit",
    )
    parser.add_argument("--alt-label", default="alt", help="Label for comparison variant")

    # Prompts
    parser.add_argument(
        "--prompt", action="append",
        help="Prompt to benchmark; pass multiple times for a sweep",
    )
    parser.add_argument(
        "--prompts-file",
        help="File with one prompt per line; blank lines and # comments are ignored",
    )

    # Runtime config
    parser.add_argument("--max-tokens", type=int, default=120, help="Max new tokens per prompt")
    parser.add_argument("--k", type=int, default=4, help="Experts per token")
    parser.add_argument("--slot-bank", type=int, default=16, help="Slot-bank size")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--cache-io-split", type=int, default=1, help="pread fanout chunks")
    parser.add_argument(
        "--compare-prefetch", action="store_true",
        help="Run both baseline and temporal-prefetch for each prompt",
    )
    parser.add_argument(
        "--prefetch-temporal", action="store_true",
        help="Enable temporal prefetch for the run",
    )
    parser.add_argument(
        "--python-slot-bank", action="store_true",
        help="Use Python slot-bank instead of native C",
    )

    # Output
    parser.add_argument(
        "--jsonl",
        default=str(Path(__file__).resolve().parent.parent.parent / "results" / "bench_capture.jsonl"),
        help="JSONL output file",
    )
    parser.add_argument("--append-jsonl", action="store_true", help="Append to JSONL")
    parser.add_argument(
        "--collect-routing-dir",
        help="Directory for per-variant binary routing traces",
    )
    parser.add_argument("--append-routing", action="store_true", help="Append to routing traces")
    parser.add_argument(
        "--preview-tokens", type=int, default=0,
        help="Print the first N generated tokens as preview",
    )
    parser.add_argument("--stream", action="store_true", help="Stream generated text")
    return parser.parse_args()


def _read_prompts(cli: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if cli.prompts_file:
        for raw_line in Path(cli.prompts_file).read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    if cli.prompt:
        prompts.extend(cli.prompt)
    if not prompts:
        prompts.extend(DEFAULT_PROMPTS)
    return prompts


class RoutingSampleWriter:
    """Write binary routing traces: (layer_idx, K, hidden_bytes, expert_ids) per token."""

    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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


def _routing_path_for_variant(cli: argparse.Namespace, variant: dict) -> Path | None:
    routing_dir = getattr(cli, "collect_routing_dir", None)
    if not routing_dir:
        return None
    return Path(routing_dir) / f"{cli.label}_{variant['label']}.bin"


def _variant_specs(cli: argparse.Namespace) -> list[dict]:
    primary_label = "2bit" if cli.expert_bits_2 else "4bit"
    specs = [
        {
            "label": primary_label,
            "mlx": cli.mlx,
            "experts": cli.experts,
            "expert_bits": 2 if cli.expert_bits_2 else None,
        }
    ]
    if cli.mlx_alt and cli.experts_alt:
        specs.append(
            {
                "label": cli.alt_label,
                "mlx": cli.mlx_alt,
                "experts": cli.experts_alt,
                "expert_bits": 2 if cli.alt_expert_bits_2 else None,
            }
        )
    return specs


def _group_summary(records: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["variant"], record["mode"]), []).append(record)

    summary_rows = []
    for (variant, mode), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "variant": variant,
                "mode": mode,
                "runs": len(rows),
                "avg_prefill_tps": statistics.mean(r["prefill_tps"] for r in rows),
                "avg_decode_tps": statistics.mean(r["decode_tps"] for r in rows),
                "avg_hit_rate": statistics.mean(r["slot_hit_rate"] for r in rows),
                "avg_full_hit_rate": statistics.mean(r["slot_full_hit_rate"] for r in rows),
                "avg_misses": statistics.mean(r["slot_misses"] for r in rows),
            }
        )
    return summary_rows


def _run_one(
    *,
    cli: argparse.Namespace,
    variant: dict,
    prompt: str,
    mode: str,
    routing_writer: RoutingSampleWriter | None = None,
):
    bundle = load_model_bundle(
        mlx_model_dir=variant["mlx"],
        experts_dir=variant["experts"],
        routed_top_k=cli.k,
        cache_io_split=cli.cache_io_split,
        slot_bank_size=cli.slot_bank,
        slot_bank_native=not cli.python_slot_bank,
        expert_bits=variant["expert_bits"],
    )

    if routing_writer is not None:
        set_routing_sample_callback(bundle.model, routing_writer.write)

    generated_ids: list[int] = []

    def on_token(token_id: int) -> None:
        generated_ids.append(token_id)
        if cli.stream:
            text = decode_incremental(bundle.tokenizer, generated_ids[-1:])
            print(text, end="", flush=True)

    stats = generate_with_stats(
        bundle=bundle,
        prompt=prompt,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        seed=cli.seed,
        on_token=on_token,
        slot_bank_temporal_prefetch=(mode == "prefetch"),
    )

    if routing_writer is not None:
        set_routing_sample_callback(bundle.model, None)

    slot_stats = collect_slot_bank_stats(bundle.model)
    text_preview = ""
    if cli.preview_tokens > 0 and generated_ids:
        text_preview = decode_incremental(bundle.tokenizer, generated_ids[: cli.preview_tokens])

    return {
        "label": cli.label,
        "variant": variant["label"],
        "mode": mode,
        "prompt": prompt,
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generated_tokens,
        "prefill_seconds": stats.prefill_seconds,
        "decode_seconds": stats.decode_seconds,
        "prefill_tps": stats.prefill_tokens_per_second,
        "decode_tps": stats.decode_tokens_per_second,
        "slot_bank_size": cli.slot_bank,
        "slot_calls": int(slot_stats["calls"]),
        "slot_requests": int(slot_stats["requests"]),
        "slot_misses": int(slot_stats["misses"]),
        "slot_hit_rate": float(slot_stats["hit_rate"]),
        "slot_full_hit_rate": float(slot_stats["full_hit_rate"]),
        "text_preview": text_preview,
        "routing_file": str(routing_writer.path) if routing_writer is not None else "",
        "routing_samples": routing_writer.samples if routing_writer is not None else 0,
    }


def main() -> int:
    cli = parse_args()
    prompts = _read_prompts(cli)
    variants = _variant_specs(cli)
    modes = ["baseline", "prefetch"] if cli.compare_prefetch else ["prefetch" if cli.prefetch_temporal else "baseline"]

    jsonl_path = Path(cli.jsonl)
    if cli.jsonl and not cli.append_jsonl and jsonl_path.exists():
        jsonl_path.unlink()
    if cli.jsonl:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[bench-capture] label={cli.label} prompts={len(prompts)} "
        f"variants={len(variants)} jsonl={cli.jsonl}",
        flush=True,
    )

    records: list[dict] = []
    for variant in variants:
        routing_writer: RoutingSampleWriter | None = None
        routing_path = _routing_path_for_variant(cli, variant)
        if routing_path is not None:
            routing_writer = RoutingSampleWriter(routing_path, append=cli.append_routing)
        try:
            print(
                f"\n[bench-capture] variant={variant['label']} mlx={variant['mlx']} "
                f"experts={variant['experts']} k={cli.k} slot_bank={cli.slot_bank}",
                flush=True,
            )
            for prompt_index, prompt in enumerate(prompts, start=1):
                print(f"[{variant['label']}] prompt {prompt_index}/{len(prompts)}: {prompt!r}", flush=True)
                for mode in modes:
                    record = _run_one(
                        cli=cli,
                        variant=variant,
                        prompt=prompt,
                        mode=mode,
                        routing_writer=routing_writer if mode == modes[0] else None,
                    )
                    records.append(record)
                    print(
                        f"  [{mode}] decode={record['decode_tps']:.2f} tok/s "
                        f"prefill={record['prefill_tps']:.2f} tok/s "
                        f"hit={record['slot_hit_rate']:.3f} "
                        f"full_hit={record['slot_full_hit_rate']:.3f} "
                        f"misses={record['slot_misses']}",
                        flush=True,
                    )
                    if record["text_preview"]:
                        print(f"  preview: {record['text_preview']!r}", flush=True)
                    if cli.jsonl:
                        with jsonl_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            if routing_writer is not None:
                print(
                    f"[bench-capture] routing_samples={routing_writer.samples} "
                    f"routing_file={routing_writer.path}",
                    flush=True,
                )
                routing_writer.close()

    summary_rows = _group_summary(records)
    print("\n[bench-capture] Summary", flush=True)
    print(
        "| variant | mode | runs | avg decode tok/s | avg prefill tok/s | avg hit rate | avg full-hit | avg misses |",
        flush=True,
    )
    print("|---------|------|-----:|-----------------:|------------------:|------------:|-----------:|-----------:|", flush=True)
    for row in summary_rows:
        print(
            f"| {row['variant']} | {row['mode']} | {row['runs']} | "
            f"{row['avg_decode_tps']:.2f} | {row['avg_prefill_tps']:.2f} | "
            f"{row['avg_hit_rate']:.4f} | {row['avg_full_hit_rate']:.4f} | "
            f"{row['avg_misses']:.1f} |",
            flush=True,
        )

    if records:
        best = max(records, key=lambda r: r["decode_tps"])
        print(
            f"\n[bench-capture] best: {best['variant']}:{best['mode']} "
            f"decode={best['decode_tps']:.2f} tok/s "
            f"hit={best['slot_hit_rate']:.4f} prompt={best['prompt']!r}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
