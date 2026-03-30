#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_qwen35.py \
  --mlx /path/to/mlx-community-Qwen3.5-397B-A17B-4bit \
  --experts /path/to/packed_experts_2bit \
  --prompt "What is Apple Neural Engine, how is it different from GPU, tell me its history" \
  --max-tokens 340 \
  --k 4 \
  --temperature 0 \
  --slot-bank 64 \
  --slot-bank-native \
  --2-bit \
  --stream
