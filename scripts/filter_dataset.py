from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from _bootstrap import bootstrap

bootstrap()

from mcs_research.core.config import load_config
from mcs_research.filtering.synthetic import filter_config_from_mapping, filter_yolo_dataset


def split_specs_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if "source_root" in config:
        source_root = Path(config["source_root"]).resolve()
        splits = config.get("split_names", ["train", "valid"])
        return {
            str(split): {
                "real_image_dir": str(source_root / str(split) / "original" / "images"),
                "real_label_dir": str(source_root / str(split) / "original" / "labels"),
                "synth_root": str(source_root / str(split) / "synth"),
                "synth_label_dir": str(source_root / str(split) / "synth_labels"),
            }
            for split in splits
        }
    return {str(split): dict(spec) for split, spec in dict(config["splits"]).items()}


def run_filter_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    filter_section = dict(config.get("filter", config))
    class_names = config.get("class_names")
    return filter_yolo_dataset(
        output_root=Path(config["output_root"]).resolve(),
        splits=split_specs_from_config(config),
        cfg=filter_config_from_mapping(filter_section),
        class_names=[str(name) for name in class_names] if class_names is not None else None,
        data_yaml=Path(config["data_yaml"]).resolve() if config.get("data_yaml") else None,
        overwrite=bool(config.get("overwrite", True)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered real/synthetic YOLO datasets from config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_filter_dataset(load_config(args.config))
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
