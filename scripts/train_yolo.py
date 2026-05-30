from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from mcs_research.core.config import load_config
from mcs_research.training.yolo import run_yolo_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Ultralytics YOLO from config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build train/eval kwargs without launching training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True
    payload = run_yolo_training(config)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
