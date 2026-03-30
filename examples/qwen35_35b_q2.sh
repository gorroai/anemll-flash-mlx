#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_qwen35.py \
  --mlx /path/to/mlx-Qwen3.5-35B-A3B-4bit \
  --experts /path/to/qwen3.5-35b-a3b-mlx-2bit \
  --prompt "What is Apple Neural Engine" \
  --max-tokens 120 \
  --k 4 \
  --temperature 0 \
  --slot-bank 16 \
  --slot-bank-native \
  --2-bit \
  --stream
