#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/lib/pipeline_common.sh"

: "${SHOT:=5}"
: "${COCO_ROOT:=$DATA_ROOT/coco}"
: "${WORK_DIR:=$CODE_REFACTORED_ROOT/workdir/coco_fsod_${SHOT}shot_seed${SEED}}"

DOWNLOAD_DIR="$DATA_ROOT/_downloads/coco"
ANNOTATION_PATH="${FSOD_JSON_PATH:-$COCO_ROOT/fsod/seed${SEED}/${SHOT}shot_novel.json}"
DATASET_ROOT="$WORK_DIR/dataset"
CONFIG_DIR="$WORK_DIR/configs"

mkdir -p "$CONFIG_DIR"

if [[ ! -f "$ANNOTATION_PATH" ]]; then
  if ! is_true "$DOWNLOAD"; then
    die "Missing FSOD annotation $ANNOTATION_PATH and DOWNLOAD=0"
  elif [[ -n "${FSOD_JSON_URL:-}" ]]; then
    download_file "$FSOD_JSON_URL" "$ANNOTATION_PATH"
  else
    die "Missing FSOD annotation $ANNOTATION_PATH. Set FSOD_JSON_URL or FSOD_JSON_PATH before downloading COCO."
  fi
fi

if [[ ! -d "$COCO_ROOT/train2017" ]]; then
  is_true "$DOWNLOAD" || die "Missing $COCO_ROOT/train2017 and DOWNLOAD=0"
  download_file "http://images.cocodataset.org/zips/train2017.zip" "$DOWNLOAD_DIR/train2017.zip"
  extract_zip "$DOWNLOAD_DIR/train2017.zip" "$COCO_ROOT"
fi

if [[ ! -f "$COCO_ROOT/annotations/instances_train2017.json" ]]; then
  is_true "$DOWNLOAD" || die "Missing COCO annotations and DOWNLOAD=0"
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DOWNLOAD_DIR/annotations_trainval2017.zip"
  extract_zip "$DOWNLOAD_DIR/annotations_trainval2017.zip" "$COCO_ROOT"
fi

write_coco_prepare_config "$CONFIG_DIR/prepare_dataset.json" "$ANNOTATION_PATH" "$COCO_ROOT/train2017" "$DATASET_ROOT"
run "$PYTHON" "$CODE_REFACTORED_ROOT/scripts/prepare_dataset.py" --config "$CONFIG_DIR/prepare_dataset.json"
run_filter_and_train "$DATASET_ROOT" "$WORK_DIR" "coco_fsod_${SHOT}shot"
