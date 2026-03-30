# QC Report — anemll-flash-mlx

Date: 2026-03-30
Model: Qwen3.5-35B-A3B-4bit (mlx-community, 4 shards, 256 experts/layer, k=4)
Machine: Apple Silicon (M-series)

## Benchmark Results

### Smoke Tests

| Test | Result |
|------|--------|
| `ci/smoke.sh` (py_compile + imports) | PASS |
| `export_mixed_sidecar.py --dry-run` (sharded safetensors) | PASS |
| Expert export (40 layers, 16.88 GiB) | PASS |

### Decode Speed (120 tokens, same prompt, temperature=0)

| Mode | Decode tok/s | Hit rate | Full-hit rate | Misses |
|------|-------------|----------|---------------|--------|
| `--resident` | **94.5** | 100% | 100% | 0 |
| `--resident-pread-mlx` | **101.6** | 100% | 100% | 0 |
| `--slot-bank 128 --prefetch-temporal --cache-io-split 4` | **43.5** | 82.7% | 56.8% | 3,729 |
| `--slot-bank 128 --cache-io-split 4` | **47.1** | 82.7% | 56.8% | 3,728 |
| `--slot-bank 64 --prefetch-temporal --cache-io-split 4` | **41.7** | 80.8% | 53.5% | 4,156 |
| `--slot-bank 64 --cache-io-split 4` | **43.1** | 80.8% | 53.5% | 4,156 |
| `--slot-bank 64` | **42.8** | 80.8% | 53.5% | 4,156 |
| `--slot-bank 16` | **17.1** | 59.5% | 20.8% | 8,743 |

### Prefill Speed

| Mode | Prefill tok/s |
|------|--------------|
| `--resident` | 19.2 |
| `--resident-pread-mlx` | 40.2 |
| `--slot-bank 64` | 22.3 |

### Key Observations

- **Resident ceiling**: 94-96 tok/s. All experts in MLX memory, no I/O.
- **Packed-bank ceiling** (`--resident-pread-mlx`): 101.6 tok/s — slightly faster than resident because it uses k=4 vs k=8 and indexed bank access.
- **Best SSD config**: slot-bank=128 at 47.1 tok/s (49% of resident ceiling). Hit rate 82.7%.
- **`--prefetch-temporal` does not help on 35B** — actually slightly slower (43.5 vs 47.1). The temporal prefetch adds overhead without enough hit-rate improvement on this model.
- **`--cache-io-split 4` is neutral on 35B** — expert size (1.69 MB) is not large enough for pread fanout to help. This flag matters more on 397B where experts are 6.75 MB.
- **slot-bank 16 is too small** — 59.5% hit rate, 17.1 tok/s. slot-bank 64+ is the practical minimum.
- **SSD bottleneck is clear**: ~4,000 misses at bank=64 means ~4,000 expert loads from SSD during 120-token generation. Each miss is a 1.69 MB pread.

### Sample Output

Prompt: "Explain the architecture of Mixture of Experts models in 3 sentences."

> Mixture of Experts (MoE) models consist of a large number of specialized
> neural network sub-networks, known as "experts," which are trained to
> perform specific tasks or handle specific data patterns. A gating network
> dynamically routes each input token to only a subset of these experts,
> typically selecting just the top-k most relevant ones to ensure
> computational efficiency. This sparse architecture allows the model to
> scale to billions or even trillions of parameters while maintaining
> inference costs comparable to a much smaller dense model.

All modes produce identical output (deterministic, temperature=0).

## Issues Found and Fixed

### Critical

1. **`export_mixed_sidecar.py` — sharded safetensors not supported**
   - Hardcoded `model.safetensors` path; fails on sharded models (35B has 4 shards)
   - Fixed: now reads `model.safetensors.index.json`, opens correct shard per layer
   - Verified with full 40-layer export (16.88 GiB)

### Medium

2. **Missing `.gitignore`** — `__pycache__/` and `.DS_Store` were tracked in the repo
   - Added `.gitignore` with standard Python exclusions
   - Removed existing `__pycache__/` and `.DS_Store` files

3. **`tokenizers` missing from `pyproject.toml`** — `model.py` imports `from tokenizers import Tokenizer`
   - Added to `[project.dependencies]`

4. **`requirements.txt` was redundant and incomplete** — duplicated `pyproject.toml` but missed `tokenizers`
   - Removed; `pyproject.toml` is the single source of truth

5. **Absolute filesystem paths in docs** — README, examples/README, scripts/README, docs/migration-map.md all had `/Users/anemll/...` paths
   - All converted to relative paths

### Low

6. **Typos in example prompts** — "differnt" and "it's history" in example scripts
   - Fixed to "different" and "its history"

7. **No UV support** — README only showed `pip install -r requirements.txt`
   - Added `uv sync` as recommended install method
   - Added `.python-version` file (3.11)

8. **README too long** — ~375 lines of design philosophy before Quick Start
   - Trimmed to ~160 lines, Quick Start at top, benchmark table added

## Files Changed

| File | Change |
|------|--------|
| `README.md` | Rewritten: UV, benchmarks, relative paths, trimmed |
| `pyproject.toml` | Added `tokenizers` dep |
| `scripts/export_mixed_sidecar.py` | Sharded safetensors support |
| `.gitignore` | Created |
| `.python-version` | Created (3.11) |
| `examples/README.md` | Relative paths |
| `scripts/README.md` | Relative paths |
| `docs/migration-map.md` | Relative paths |
| `examples/qwen35_397b_q2.sh` | Typo fix |
| `examples/oracle_replay.sh` | Typo fix |
| `requirements.txt` | Removed |
| `flash_moe_mlx/__pycache__/` | Removed |
| `.DS_Store`, `tools/.DS_Store` | Removed |
