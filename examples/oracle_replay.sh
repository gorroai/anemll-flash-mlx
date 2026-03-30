#!/usr/bin/env bash
set -euo pipefail

python3 tools/diagnostics/bench_slot_bank_oracle_hits.py \
  --mlx /path/to/mlx-community-Qwen3.5-397B-A17B-4bit \
  --experts /path/to/packed_experts_2bit \
  --prompt "What is Apple Neural Engine, how is it different from GPU, tell me its history" \
  --max-tokens 340 \
  --k 4 \
  --temperature 0 \
  --slot-bank 64 \
  --trace-mode slot-bank \
  --2-bit \
  --one-step-prefetch \
  --stream \
  --verbose
