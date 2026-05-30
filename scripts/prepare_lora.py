from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from mcs_research.core.config import load_config
from mcs_research.lora.fsod_prepare import run_lora_prepare_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and optionally caption LoRA training inputs.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_lora_prepare_from_config(load_config(args.config))
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
