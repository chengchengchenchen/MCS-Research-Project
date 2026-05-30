from __future__ import annotations

import ast
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from mcs_research.core.files import ensure_dir, list_images, load_json, write_json
from mcs_research.datasets.yolo import format_label_counts, load_class_names, yolo_label_counts

API = "http://127.0.0.1:8000"
CLIENT_ID = "python"
BATCH_SIZE = 8

NODE_ID_LOADIMAGE = "7"
NODE_ID_SAVEIMAGE = "30"
NODE_ID_YOLO_PROMPT = "35"
NODE_ID_LABEL_PATH = "40"
NODE_ID_DATA_YAML = "41"
NODE_ID_SAVE_DIR = "42"
NODE_ID_BATCH_SIZE = "43"
REQUIRED_NODE_IDS = (
    NODE_ID_LOADIMAGE,
    NODE_ID_SAVEIMAGE,
    NODE_ID_YOLO_PROMPT,
    NODE_ID_LABEL_PATH,
    NODE_ID_DATA_YAML,
    NODE_ID_SAVE_DIR,
    NODE_ID_BATCH_SIZE,
)


def load_prompt(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return payload.get("prompt", payload)


def submit(prompt: Mapping[str, Any], api: str, client_id: str) -> str:
    response = requests.post(f"{api}/prompt", json={"prompt": prompt, "client_id": client_id}, timeout=60)
    response.raise_for_status()
    return str(response.json()["prompt_id"])


def wait_done(prompt_id: str, api: str, timeout: float = 600.0, poll: float = 0.5) -> dict[str, Any]:
    started = time.time()
    while True:
        response = requests.get(f"{api}/history/{prompt_id}", timeout=30)
        if response.ok:
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
        if time.time() - started > timeout:
            raise TimeoutError("Prompt execution timed out.")
        time.sleep(poll)


def clone_prompt(prompt: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(prompt))


def validate_prompt(prompt: Mapping[str, Any]) -> None:
    missing = [node_id for node_id in REQUIRED_NODE_IDS if node_id not in prompt]
    if missing:
        raise KeyError(f"Workflow is missing required node id(s): {', '.join(missing)}")


def normalize_caption_tokens(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def resolve_suffix(caption_path: Path, fallback_suffix: str, fixed_prefix: str, label_count_tokens: list[str]) -> tuple[str, str]:
    if not caption_path.exists():
        return fallback_suffix, "fallback:missing-caption"
    tokens = normalize_caption_tokens(caption_path.read_text(encoding="utf-8").strip())
    if not tokens:
        return fallback_suffix, "fallback:empty-caption"
    if fixed_prefix and tokens and tokens[0].casefold() == fixed_prefix.casefold():
        tokens = tokens[1:]
    lowered_counts = [token.casefold() for token in label_count_tokens]
    if lowered_counts and [token.casefold() for token in tokens[: len(lowered_counts)]] == lowered_counts:
        tokens = tokens[len(lowered_counts) :]
    suffix = ", ".join(tokens).strip()
    return (suffix, "caption") if suffix else (fallback_suffix, "fallback:empty-after-clean")


def run_comfyui_generation(
    *,
    prompt_path: Path,
    input_dir: Path,
    label_dir: Path,
    data_yaml: Path,
    output_dir: Path,
    api: str = API,
    client_id: str = CLIENT_ID,
    batch_size: int = BATCH_SIZE,
    caption_dir: Path | None = None,
    wait_timeout: float = 600.0,
    sleep_after_each: float = 0.2,
) -> dict[str, Any]:
    base_prompt = load_prompt(prompt_path)
    validate_prompt(base_prompt)
    caption_dir = caption_dir or input_dir
    class_names = load_class_names(data_yaml)

    yolo_inputs = base_prompt[NODE_ID_YOLO_PROMPT]["inputs"]
    fallback_suffix = str(yolo_inputs.get("fixed_suffix", ""))
    fixed_prefix = str(yolo_inputs.get("fixed_prefix", "")).strip()
    base_prompt[NODE_ID_DATA_YAML]["inputs"]["value"] = str(data_yaml.as_posix())
    base_prompt[NODE_ID_BATCH_SIZE]["inputs"]["value"] = int(batch_size)

    synth_img_dir = output_dir / "images"
    synth_lbl_dir = output_dir / "labels"
    ensure_dir(output_dir)
    ensure_dir(synth_img_dir)
    ensure_dir(synth_lbl_dir)

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(list_images(input_dir), start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            rows.append({"image": str(image_path), "status": "missing_label"})
            continue
        label_count_tokens = [
            item.strip()
            for item in format_label_counts(yolo_label_counts(label_path), class_names).split(",")
            if item.strip()
        ]
        suffix, suffix_source = resolve_suffix(
            caption_dir / f"{image_path.stem}.txt",
            fallback_suffix,
            fixed_prefix,
            label_count_tokens,
        )
        prompt = clone_prompt(base_prompt)
        prompt[NODE_ID_LOADIMAGE]["inputs"]["image"] = str(image_path.as_posix())
        prompt[NODE_ID_LABEL_PATH]["inputs"]["value"] = str(label_path.as_posix())
        prompt[NODE_ID_YOLO_PROMPT]["inputs"]["fixed_suffix"] = suffix

        run_dir = output_dir / image_path.stem
        ensure_dir(run_dir)
        prompt[NODE_ID_SAVE_DIR]["inputs"]["value"] = str(run_dir.as_posix())
        prompt[NODE_ID_SAVEIMAGE]["inputs"]["file_name"] = image_path.stem

        prompt_id = submit(prompt, api=api, client_id=client_id)
        wait_done(prompt_id, api=api, timeout=wait_timeout)
        generated = sorted(run_dir.glob(f"{image_path.stem}_*.png"))
        for generated_path in generated:
            shutil.copy2(generated_path, synth_img_dir / generated_path.name)
            shutil.copy2(label_path, synth_lbl_dir / f"{generated_path.stem}.txt")
        rows.append(
            {
                "index": index,
                "image": str(image_path),
                "run_dir": str(run_dir),
                "prompt_id": prompt_id,
                "generated": len(generated),
                "suffix_source": suffix_source,
                "status": "ok",
            }
        )
        time.sleep(sleep_after_each)

    summary = {"output_dir": str(output_dir), "images": len(rows), "rows": rows}
    write_json(output_dir / "generation_summary.json", summary)
    return summary


def run_comfyui_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return run_comfyui_generation(
        prompt_path=Path(config["prompt_path"]).resolve(),
        input_dir=Path(config["input_dir"]).resolve(),
        label_dir=Path(config["label_dir"]).resolve(),
        data_yaml=Path(config["data_yaml"]).resolve(),
        output_dir=Path(config["output_dir"]).resolve(),
        api=str(config.get("api", API)),
        client_id=str(config.get("client_id", CLIENT_ID)),
        batch_size=int(config.get("batch_size", BATCH_SIZE)),
        caption_dir=Path(config["caption_dir"]).resolve() if config.get("caption_dir") else None,
        wait_timeout=float(config.get("wait_timeout", 600.0)),
        sleep_after_each=float(config.get("sleep_after_each", 0.2)),
    )
