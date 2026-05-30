from __future__ import annotations

import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcs_research.core.files import clear_dir, ensure_dir, list_images, load_json, materialize_file, write_json
from mcs_research.datasets.yolo import convert_coco_to_yolo, write_data_yaml
from mcs_research.lora.captioning import (
    DEFAULT_PROMPT,
    DEFAULT_STYLE_TOKEN,
    build_prompt,
    ensure_model_available,
    generate_caption,
)
from mcs_research.datasets.yolo import format_label_counts, load_class_names, yolo_label_counts


def verify_nested_image_sets(fsod_json_root: Path, kshots: Sequence[int]) -> None:
    previous_ids: set[int] | None = None
    previous_k: int | None = None
    for k in sorted(kshots):
        payload = load_json(fsod_json_root / f"{k}shot_novel.json")
        current_ids = {int(image["id"]) for image in payload.get("images", [])}
        if previous_ids is not None and not previous_ids.issubset(current_ids):
            raise ValueError(f"FSOD image sets are not nested: {previous_k}shot is not a subset of {k}shot")
        previous_ids = current_ids
        previous_k = k


def is_retryable_caption_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return True
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def generate_caption_with_retry(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: Path,
    timeout: int,
    temperature: float,
    retries: int,
    retry_wait: float,
) -> str:
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return generate_caption(
                base_url=base_url,
                model=model,
                prompt=prompt,
                image_path=image_path,
                timeout=timeout,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts or not is_retryable_caption_error(exc):
                raise
            print(f"[retry] {image_path.name} attempt {attempt}/{attempts} failed: {exc}", file=sys.stderr)
            time.sleep(retry_wait)
    raise RuntimeError(f"Caption generation failed for {image_path}: {last_error}")


def prepare_fsod_lora_inputs(
    *,
    seed: int,
    kshots: Sequence[int],
    coco_train_root: Path,
    fsod_json_root: Path,
    output_root: Path,
    overwrite: bool = False,
    image_storage: str = "copy",
) -> dict[str, Any]:
    seed_root = fsod_json_root / f"seed{seed}"
    output_seed_root = output_root / f"seed{seed}"
    verify_nested_image_sets(seed_root, kshots)
    summaries: dict[str, Any] = {}
    for k in sorted(dict.fromkeys(kshots)):
        subset_root = output_seed_root / f"{k}shot"
        if overwrite:
            clear_dir(subset_root)
        else:
            ensure_dir(subset_root)
        summary = convert_coco_to_yolo(
            annotation_path=seed_root / f"{k}shot_novel.json",
            image_root=coco_train_root,
            output_root=subset_root,
            split_name="imageset",
            image_storage=image_storage,
        )
        # LoRA tools expect flat images/labels plus data.yaml.
        flat_images = subset_root / "images"
        flat_labels = subset_root / "labels"
        ensure_dir(flat_images)
        ensure_dir(flat_labels)
        for image_path in list_images(subset_root / "imageset" / "images"):
            materialize_file(image_path, flat_images / image_path.name, "copy")
            label_path = subset_root / "imageset" / "labels" / f"{image_path.stem}.txt"
            if label_path.exists():
                materialize_file(label_path, flat_labels / label_path.name, "copy")
        write_data_yaml(subset_root / "data.yaml", summary["class_names"], train="images", val="images")
        summaries[str(k)] = summary
    manifest = {"seed": seed, "kshots": list(kshots), "output_root": str(output_seed_root), "subsets": summaries}
    write_json(output_seed_root / "lora_prepare_manifest.json", manifest)
    return manifest


def caption_lora_subset(
    *,
    subset_root: Path,
    model: str,
    base_url: str,
    style_token: str = DEFAULT_STYLE_TOKEN,
    prompt_template: str = DEFAULT_PROMPT,
    timeout: int = 600,
    temperature: float = 0.3,
    retries: int = 2,
    retry_wait: float = 15.0,
    limit: int = 0,
    overwrite_captions: bool = False,
) -> dict[str, Any]:
    image_dir = subset_root / "images"
    label_dir = subset_root / "labels"
    data_yaml = subset_root / "data.yaml"
    class_names = load_class_names(data_yaml)
    images = list_images(image_dir)
    if limit > 0:
        images = images[:limit]

    written = skipped = 0
    failures: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists() and not overwrite_captions:
            skipped += 1
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        prompt = build_prompt(prompt_template, format_label_counts(yolo_label_counts(label_path), class_names))
        try:
            caption = generate_caption_with_retry(
                base_url=base_url,
                model=model,
                prompt=prompt,
                image_path=image_path,
                timeout=timeout,
                temperature=temperature,
                retries=retries,
                retry_wait=retry_wait,
            )
            caption_text = f"{style_token}, {caption}" if style_token else caption
            caption_path.write_text(caption_text + "\n", encoding="utf-8")
            written += 1
            print(f"[caption] {subset_root.name} {index}/{len(images)} wrote {caption_path.name}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"image": str(image_path), "label": str(label_path), "error": str(exc)})
            print(f"[caption] failed {image_path.name}: {exc}", file=sys.stderr)

    summary = {
        "subset_root": str(subset_root),
        "images": len(images),
        "captions_written": written,
        "captions_skipped": skipped,
        "failures": failures,
    }
    write_json(subset_root / "caption_summary.json", summary)
    return summary


def run_lora_prepare_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    kshots = [int(item) for item in config.get("kshots", [5, 10, 30])]
    output_root = Path(config["output_root"]).resolve()
    prepare_summary: dict[str, Any] | None = None
    if not config.get("skip_prepare", False):
        prepare_summary = prepare_fsod_lora_inputs(
            seed=seed,
            kshots=kshots,
            coco_train_root=Path(config["coco_train_root"]).resolve(),
            fsod_json_root=Path(config["fsod_json_root"]).resolve(),
            output_root=output_root,
            overwrite=bool(config.get("overwrite", False)),
            image_storage=str(config.get("image_storage", "copy")),
        )

    caption_summaries: dict[str, Any] = {}
    if not config.get("skip_caption", False):
        caption_cfg = dict(config.get("caption", {}))
        ensure_model_available(
            str(caption_cfg.get("base_url", "http://127.0.0.1:11434")),
            str(caption_cfg.get("model", "qwen3-vl:32b")),
            int(caption_cfg.get("timeout", 600)),
        )
        for k in kshots:
            subset_root = output_root / f"seed{seed}" / f"{k}shot"
            caption_summaries[str(k)] = caption_lora_subset(
                subset_root=subset_root,
                model=str(caption_cfg.get("model", "qwen3-vl:32b")),
                base_url=str(caption_cfg.get("base_url", "http://127.0.0.1:11434")),
                style_token=str(caption_cfg.get("style_token", DEFAULT_STYLE_TOKEN)),
                prompt_template=str(caption_cfg.get("prompt", DEFAULT_PROMPT)),
                timeout=int(caption_cfg.get("timeout", 600)),
                temperature=float(caption_cfg.get("temperature", 0.3)),
                retries=int(caption_cfg.get("retries", 2)),
                retry_wait=float(caption_cfg.get("retry_wait", 15.0)),
                limit=int(caption_cfg.get("limit", 0)),
                overwrite_captions=bool(caption_cfg.get("overwrite_captions", False)),
            )

    return {"prepare": prepare_summary, "caption": caption_summaries}
