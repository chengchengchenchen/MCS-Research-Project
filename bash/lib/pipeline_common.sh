#!/usr/bin/env bash

set -euo pipefail

code_refactored_root_from_this_file() {
  local lib_dir
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  local root_dir
  root_dir="$(cd "$lib_dir/../.." && pwd -P)"
  if (cd "$root_dir" && pwd -W) >/dev/null 2>&1; then
    (cd "$root_dir" && pwd -W)
  else
    printf '%s\n' "$root_dir"
  fi
}

: "${PYTHON:=python}"
: "${CODE_REFACTORED_ROOT:=$(code_refactored_root_from_this_file)}"
: "${PROJECT_ROOT:=$CODE_REFACTORED_ROOT}"
: "${DATA_ROOT:=$CODE_REFACTORED_ROOT/data}"
: "${SEED:=1}"
: "${IMAGE_STORAGE:=copy}"
: "${WEIGHTS:=yolov8s.pt}"
: "${EPOCHS:=100}"
: "${IMG_SIZE:=416}"
: "${BATCH:=32}"
: "${WORKERS:=8}"
: "${DRY_RUN:=0}"
: "${DOWNLOAD:=1}"
: "${RUN_GENERATION:=0}"
: "${INCLUDE_SYNTH:=$RUN_GENERATION}"
: "${COMFYUI_API:=http://127.0.0.1:8000}"
: "${COMFYUI_CLIENT_ID:=bash_pipeline}"
: "${GEN_BATCH_SIZE:=8}"
: "${INCLUDE_DIFFICULT:=1}"

export PYTHON PROJECT_ROOT CODE_REFACTORED_ROOT DATA_ROOT SEED IMAGE_STORAGE WEIGHTS EPOCHS IMG_SIZE BATCH WORKERS
export DRY_RUN DOWNLOAD RUN_GENERATION INCLUDE_SYNTH COMFYUI_API COMFYUI_CLIENT_ID GEN_BATCH_SIZE INCLUDE_DIFFICULT

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

download_file() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    printf '[skip] %s exists\n' "$dest"
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    printf '+ curl -L --fail --retry 3 -o %q <url>\n' "$dest"
    curl -L --fail --retry 3 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    printf '+ wget -O %q <url>\n' "$dest"
    wget -O "$dest" "$url"
  else
    URL="$url" DEST="$dest" "$PYTHON" - <<'PY'
import os
import urllib.request

urllib.request.urlretrieve(os.environ["URL"], os.environ["DEST"])
PY
  fi
}

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  run "$PYTHON" -m zipfile -e "$zip_path" "$dest_dir"
}

extract_tar() {
  local tar_path="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  run tar -xf "$tar_path" -C "$dest_dir"
}

yolo_dataset_exists() {
  local root="$1"
  [[ -f "$root/data.yaml" && -d "$root/train/images" && -d "$root/train/labels" ]]
}

find_yolo_root() {
  local root="$1"
  if yolo_dataset_exists "$root"; then
    printf '%s\n' "$root"
    return 0
  fi
  local yaml_path
  yaml_path="$(find "$root" -type f -name data.yaml -print 2>/dev/null | head -n 1 || true)"
  if [[ -n "$yaml_path" ]] && yolo_dataset_exists "$(dirname "$yaml_path")"; then
    dirname "$yaml_path"
    return 0
  fi
  return 1
}

write_coco_prepare_config() {
  local config_path="$1"
  local annotation_path="$2"
  local image_root="$3"
  local output_root="$4"
  mkdir -p "$(dirname "$config_path")"
  CONFIG_PATH="$config_path" ANNOTATION_PATH="$annotation_path" IMAGE_ROOT="$image_root" OUTPUT_ROOT="$output_root" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "mode": "coco_to_yolo",
    "annotation_path": os.environ["ANNOTATION_PATH"],
    "image_root": os.environ["IMAGE_ROOT"],
    "output_root": os.environ["OUTPUT_ROOT"],
    "split_name": "train",
    "image_storage": os.environ.get("IMAGE_STORAGE", "copy"),
}
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_voc_prepare_config() {
  local config_path="$1"
  local devkit_root="$2"
  local output_root="$3"
  mkdir -p "$(dirname "$config_path")"
  CONFIG_PATH="$config_path" DEVKIT_ROOT="$devkit_root" OUTPUT_ROOT="$output_root" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "mode": "voc_to_yolo",
    "devkit_root": os.environ["DEVKIT_ROOT"],
    "output_root": os.environ["OUTPUT_ROOT"],
    "image_storage": os.environ.get("IMAGE_STORAGE", "copy"),
    "include_difficult": os.environ.get("INCLUDE_DIFFICULT", "1").lower() in {"1", "true", "yes", "on"},
    "train_sets": [["2007", "trainval"], ["2012", "trainval"]],
    "valid_sets": [["2007", "test"]],
    "test_sets": [["2007", "test"]],
}
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_generate_config() {
  local config_path="$1"
  local source_root="$2"
  local output_root="$3"
  mkdir -p "$(dirname "$config_path")"
  CONFIG_PATH="$config_path" SOURCE_ROOT="$source_root" OUTPUT_ROOT="$output_root" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

prompt_path = os.environ.get("COMFYUI_WORKFLOW") or os.environ.get("COMFYUI_PROMPT_PATH")
if not prompt_path:
    raise SystemExit("COMFYUI_WORKFLOW or COMFYUI_PROMPT_PATH is required when RUN_GENERATION=1")
source_root = Path(os.environ["SOURCE_ROOT"])
payload = {
    "prompt_path": prompt_path,
    "input_dir": str(source_root / "train" / "images"),
    "label_dir": str(source_root / "train" / "labels"),
    "data_yaml": str(source_root / "data.yaml"),
    "output_dir": os.environ["OUTPUT_ROOT"],
    "api": os.environ.get("COMFYUI_API", "http://127.0.0.1:8000"),
    "client_id": os.environ.get("COMFYUI_CLIENT_ID", "bash_pipeline"),
    "batch_size": int(os.environ.get("GEN_BATCH_SIZE", "8")),
    "wait_timeout": float(os.environ.get("COMFYUI_WAIT_TIMEOUT", "600")),
    "sleep_after_each": float(os.environ.get("COMFYUI_SLEEP_AFTER_EACH", "0.2")),
}
caption_dir = os.environ.get("CAPTION_DIR")
if caption_dir:
    payload["caption_dir"] = caption_dir
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_filter_config() {
  local config_path="$1"
  local source_root="$2"
  local output_root="$3"
  local synth_root="$4"
  mkdir -p "$(dirname "$config_path")"
  CONFIG_PATH="$config_path" SOURCE_ROOT="$source_root" OUTPUT_ROOT="$output_root" SYNTH_ROOT="$synth_root" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

source_root = Path(os.environ["SOURCE_ROOT"])
synth_root = Path(os.environ["SYNTH_ROOT"])
include_synth = env_bool("INCLUDE_SYNTH")
splits = {}
for split in ("train", "valid", "test"):
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        continue
    spec = {
        "real_image_dir": str(image_dir),
        "real_label_dir": str(label_dir),
    }
    if include_synth and split == "train":
        spec["synth_root"] = str(synth_root)
        spec["synth_label_dir"] = str(synth_root / "labels")
    splits[split] = spec
if "train" not in splits:
    raise SystemExit(f"Missing YOLO train split under {source_root}")

payload = {
    "output_root": os.environ["OUTPUT_ROOT"],
    "data_yaml": str(source_root / "data.yaml"),
    "splits": splits,
    "filter": {
        "include_real": True,
        "include_synth": include_synth,
        "synth_keep_ratio": float(os.environ.get("SYNTH_KEEP_RATIO", "0.3")),
        "synth_keep_topk": int(os.environ.get("SYNTH_KEEP_TOPK", "-1")),
        "synth_min_keep": int(os.environ.get("SYNTH_MIN_KEEP", "0")),
        "filter_quantile": float(os.environ.get("FILTER_QUANTILE", "25.0")),
        "weight_clip": float(os.environ.get("WEIGHT_CLIP", "0.2")),
        "weight_dino": float(os.environ.get("WEIGHT_DINO", "0.5")),
        "weight_bbox_mean": float(os.environ.get("WEIGHT_BBOX_MEAN", "0.25")),
        "weight_bbox_min": float(os.environ.get("WEIGHT_BBOX_MIN", "0.05")),
        "synth_exact_count": int(os.environ.get("SYNTH_EXACT_COUNT", "0")),
        "retention_mode": os.environ.get("RETENTION_MODE", "global"),
        "retention_stem_cap": int(os.environ.get("RETENTION_STEM_CAP", "0")),
        "sample_total": int(os.environ.get("SAMPLE_TOTAL", "0")),
        "sample_seed": int(os.environ.get("SEED", "1")),
        "image_storage": os.environ.get("IMAGE_STORAGE", "copy"),
    },
}
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_train_config() {
  local config_path="$1"
  local filtered_root="$2"
  local run_root="$3"
  local dataset_name="$4"
  mkdir -p "$(dirname "$config_path")"
  CONFIG_PATH="$config_path" FILTERED_ROOT="$filtered_root" RUN_ROOT="$run_root" DATASET_NAME="$dataset_name" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

filtered_root = Path(os.environ["FILTERED_ROOT"])
default_eval_split = "test" if (filtered_root / "test" / "images").exists() else "valid"
if not (filtered_root / default_eval_split / "images").exists():
    default_eval_split = "train"
name = os.environ.get("RUN_NAME") or f"{os.environ['DATASET_NAME']}_seed{os.environ.get('SEED', '1')}"
payload = {
    "data": str(filtered_root / "data.yaml"),
    "weights": os.environ.get("WEIGHTS", "yolov8s.pt"),
    "epochs": int(os.environ.get("EPOCHS", "100")),
    "imgsz": int(os.environ.get("IMG_SIZE", "416")),
    "batch": int(os.environ.get("BATCH", "32")),
    "workers": int(os.environ.get("WORKERS", "8")),
    "seed": int(os.environ.get("SEED", "1")),
    "project": os.environ["RUN_ROOT"],
    "name": name,
    "eval_split": os.environ.get("EVAL_SPLIT", default_eval_split),
    "val_out": str(Path(os.environ["RUN_ROOT"]) / f"{name}_eval.json"),
}
for key in ("DEVICE", "PATIENCE", "CACHE", "RESUME"):
    if key in os.environ and os.environ[key] != "":
        payload[key.lower()] = os.environ[key]
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

maybe_run_generation() {
  local source_root="$1"
  local generated_root="$2"
  local config_path="$3"
  if ! is_true "$RUN_GENERATION"; then
    printf '[skip] generation disabled; set RUN_GENERATION=1 to call ComfyUI\n'
    return 0
  fi
  write_generate_config "$config_path" "$source_root" "$generated_root"
  run "$PYTHON" "$CODE_REFACTORED_ROOT/scripts/generate_comfyui.py" --config "$config_path"
}

run_filter_and_train() {
  local source_root="$1"
  local work_dir="$2"
  local dataset_name="$3"
  local config_dir="$work_dir/configs"
  local generated_root="$work_dir/generated"
  local filtered_root="$work_dir/filtered"
  local runs_root="$work_dir/yolo_runs"
  mkdir -p "$config_dir" "$runs_root"

  maybe_run_generation "$source_root" "$generated_root" "$config_dir/generate_comfyui.json"
  write_filter_config "$config_dir/filter_dataset.json" "$source_root" "$filtered_root" "$generated_root"
  run "$PYTHON" "$CODE_REFACTORED_ROOT/scripts/filter_dataset.py" --config "$config_dir/filter_dataset.json"
  write_train_config "$config_dir/train_yolo.json" "$filtered_root" "$runs_root" "$dataset_name"
  local train_args=("$PYTHON" "$CODE_REFACTORED_ROOT/scripts/train_yolo.py" --config "$config_dir/train_yolo.json")
  if is_true "$DRY_RUN"; then
    train_args+=(--dry-run)
  fi
  run "${train_args[@]}"
}
