#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m py_compile \
  "$ROOT_DIR/flash_moe_mlx/__init__.py" \
  "$ROOT_DIR/flash_moe_mlx/expert_io.py" \
  "$ROOT_DIR/flash_moe_mlx/model.py" \
  "$ROOT_DIR/flash_moe_mlx/upstream_gated_delta.py" \
  "$ROOT_DIR/flash_moe_mlx/upstream_switch_layers.py" \
  "$ROOT_DIR/scripts/run_qwen35.py" \
  "$ROOT_DIR/scripts/export_tiered_35b_2bit.py" \
  "$ROOT_DIR/scripts/export_mixed_sidecar.py" \
  "$ROOT_DIR/tools/diagnostics/bench_slot_bank_oracle_hits.py" \
  "$ROOT_DIR/tools/diagnostics/bench_slot_commit.py" \
  "$ROOT_DIR/tools/diagnostics/bench_capture.py" \
  "$ROOT_DIR/tests/test_imports.py"

python3 "$ROOT_DIR/tests/test_imports.py"
