# Migration Map

This map defines what should move from the research repo into `anemll-flash-mlx`.

## Copy Into New Repo

Runtime package:

- `flash_moe_mlx/__init__.py`
- `flash_moe_mlx/model.py`
- `flash_moe_mlx/expert_io.py`
- `flash_moe_mlx/upstream_gated_delta.py`
- `flash_moe_mlx/upstream_switch_layers.py`

Runtime entrypoint:

- `scripts/run_qwen35.py`

Native helper:

- `csrc/expert_io.c`

Conversion scripts:

- `scripts/export_tiered_35b_2bit.py`
- `scripts/export_mixed_sidecar.py`

Supported diagnostics:

- `tools/diagnostics/bench_slot_bank_oracle_hits.py`
- `tools/diagnostics/bench_slot_commit.py`
- `tools/diagnostics/bench_neo_capture.py`

## Leave In Research Repo

- handoff documents
- predictor datasets and training notes
- dead-end benches
- locality trace experiments
- backend-comparison notes

## Extraction Principle

Do not move files out of the parent repo yet.

Instead:

1. copy needed files into `anemll-flash-mlx/`
2. clean paths and docs there
3. test there
4. only then consider making it a standalone repo
