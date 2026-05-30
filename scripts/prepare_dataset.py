from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from _bootstrap import bootstrap

bootstrap()

from mcs_research.core.config import load_config
from mcs_research.core.files import write_json
from mcs_research.datasets.voc import convert_voc_to_yolo
from mcs_research.datasets.yolo import convert_coco_to_yolo, summarize_yolo_split, write_data_yaml


def run_prepare_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(config.get("mode", "coco_to_yolo"))
    output_root = Path(config.get("output_root", "code_refactored/workdir/dataset")).resolve()
    image_storage = str(config.get("image_storage", "copy"))

    if mode == "coco_to_yolo":
        summary = convert_coco_to_yolo(
            annotation_path=Path(config["annotation_path"]).resolve(),
            image_root=Path(config["image_root"]).resolve(),
            output_root=output_root,
            split_name=str(config.get("split_name", "train")),
            image_storage=image_storage,
            overwrite_labels=bool(config.get("overwrite_labels", True)),
        )
        return {"mode": mode, "output_root": str(output_root), "splits": {summary["split_name"]: summary}}

    if mode == "coco_splits":
        split_summaries: dict[str, Any] = {}
        for split, split_cfg in dict(config["splits"]).items():
            split_cfg = dict(split_cfg)
            split_summaries[split] = convert_coco_to_yolo(
                annotation_path=Path(split_cfg["annotation_path"]).resolve(),
                image_root=Path(split_cfg.get("image_root", config["image_root"])).resolve(),
                output_root=output_root,
                split_name=str(split),
                image_storage=str(split_cfg.get("image_storage", image_storage)),
                overwrite_labels=bool(split_cfg.get("overwrite_labels", True)),
            )
        names = split_summaries[next(iter(split_summaries))]["class_names"]
        write_data_yaml(
            output_root / "data.yaml",
            names,
            train="train/images" if "train" in split_summaries else f"{next(iter(split_summaries))}/images",
            val="valid/images" if "valid" in split_summaries else f"{next(iter(split_summaries))}/images",
            test="test/images" if "test" in split_summaries else None,
        )
        payload = {"mode": mode, "output_root": str(output_root), "splits": split_summaries}
        write_json(output_root / "dataset_manifest.json", payload)
        return payload

    if mode == "voc_to_yolo":
        return convert_voc_to_yolo(
            devkit_root=Path(config["devkit_root"]).resolve(),
            output_root=output_root,
            train_sets=config.get("train_sets", [["2007", "trainval"], ["2012", "trainval"]]),
            valid_sets=config.get("valid_sets", [["2007", "test"]]),
            test_sets=config.get("test_sets", [["2007", "test"]]),
            include_difficult=bool(config.get("include_difficult", True)),
            image_storage=image_storage,
        )

    if mode == "summarize_yolo":
        summaries = {
            split: summarize_yolo_split(Path(spec["image_dir"]).resolve(), Path(spec["label_dir"]).resolve())
            for split, spec in dict(config["splits"]).items()
        }
        payload = {"mode": mode, "splits": summaries}
        if config.get("manifest_out"):
            write_json(Path(config["manifest_out"]).resolve(), payload)
        return payload

    raise ValueError(f"Unsupported prepare_dataset mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format dataset from config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_prepare_dataset(load_config(args.config))
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
