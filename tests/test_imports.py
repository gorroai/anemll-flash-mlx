from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    import flash_moe_mlx
    from scripts import run_qwen35

    assert hasattr(flash_moe_mlx, "load_model_bundle")
    parser = run_qwen35.build_arg_parser()
    assert parser.prog
    print("import-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
