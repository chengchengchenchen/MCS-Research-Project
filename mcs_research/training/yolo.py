from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mcs_research.core.files import write_json


def to_jsonable(results: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            cleaned[str(key)] = value
            continue
        try:
            cleaned[str(key)] = float(value)
        except (TypeError, ValueError):
            cleaned[str(key)] = str(value)
    return cleaned


def build_train_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = {
        "data": str(Path(config.get("data", "data.yaml")).resolve()),
        "epochs": int(config.get("epochs", 100)),
        "imgsz": int(config.get("imgsz", 416)),
        "batch": int(config.get("batch", 32)),
        "workers": int(config.get("workers", 8)),
        "seed": int(config.get("seed", 1)),
        "project": str(Path(config.get("project", "runs_orig")).resolve()),
        "name": str(config.get("name", "yolov8s_e100_416")),
    }
    for optional_key in ("device", "patience", "cache", "resume"):
        if optional_key in config:
            kwargs[optional_key] = config[optional_key]
    return kwargs


def run_yolo_training(config: Mapping[str, Any]) -> dict[str, Any]:
    weights = str(config.get("weights", "yolov8s.pt"))
    dry_run = bool(config.get("dry_run", False))
    train_kwargs = build_train_kwargs(config)
    val_kwargs = {
        "data": train_kwargs["data"],
        "imgsz": int(config.get("eval_imgsz", train_kwargs["imgsz"])),
        "split": str(config.get("eval_split", "test")),
    }
    output = {
        "weights": weights,
        "train_kwargs": train_kwargs,
        "val_kwargs": val_kwargs,
        "dry_run": dry_run,
    }
    if dry_run:
        if config.get("val_out"):
            write_json(Path(config["val_out"]).resolve(), output)
        return output

    from ultralytics import YOLO

    model = YOLO(weights)
    train_result = model.train(**train_kwargs)
    val_result = model.val(**val_kwargs)
    results_dict = to_jsonable(getattr(val_result, "results_dict", {}) or {})
    output["train_result"] = str(train_result)
    output["results_dict"] = results_dict
    if config.get("val_out"):
        write_json(Path(config["val_out"]).resolve(), output)
    return output
