from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from mcs_research.core.config import load_config
from mcs_research.core.files import write_json
from mcs_research.lora.fsod_prepare import run_lora_prepare_from_config
from mcs_research.generation.comfyui import run_comfyui_from_config
from mcs_research.training.yolo import run_yolo_training
from prepare_dataset import run_prepare_dataset
from filter_dataset import run_filter_dataset


def section_enabled(section: Any) -> bool:
    return isinstance(section, dict) and bool(section.get("enabled", True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal dataset -> LoRA -> generation -> filter -> train pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results: dict[str, Any] = {}

    if section_enabled(config.get("prepare_dataset")):
        results["prepare_dataset"] = run_prepare_dataset(config["prepare_dataset"])
    if section_enabled(config.get("prepare_lora")):
        results["prepare_lora"] = run_lora_prepare_from_config(config["prepare_lora"])
    if section_enabled(config.get("generate_comfyui")):
        results["generate_comfyui"] = run_comfyui_from_config(config["generate_comfyui"])
    if section_enabled(config.get("filter_dataset")):
        results["filter_dataset"] = run_filter_dataset(config["filter_dataset"])
    if section_enabled(config.get("train_yolo")):
        train_cfg = dict(config["train_yolo"])
        if args.dry_run_train:
            train_cfg["dry_run"] = True
        results["train_yolo"] = run_yolo_training(train_cfg)

    if config.get("summary_out"):
        write_json(Path(config["summary_out"]).resolve(), results)
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
