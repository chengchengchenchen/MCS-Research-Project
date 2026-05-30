#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/lib/pipeline_common.sh"

get_env_or_default() {
  local name="$1"
  local default_value="$2"
  local value="${!name:-}"
  printf '%s\n' "${value:-$default_value}"
}

: "${DATASET_NAME:?DATASET_NAME is required}"
: "${RAW_SLUG:?RAW_SLUG is required}"
: "${RF100_ROOT:=$DATA_ROOT/rf100}"
: "${WORK_DIR:=$CODE_REFACTORED_ROOT/workdir/rf100_${DATASET_NAME}_seed${SEED}}"

DATASET_ENV="$(printf '%s' "$DATASET_NAME" | tr '[:lower:]-' '[:upper:]_')"
URL_VAR="RF100_${DATASET_ENV}_ZIP_URL"
DATASET_URL="${!URL_VAR-}"
DATASET_URL="${DATASET_URL:-${RF100_ZIP_URL:-}}"
API_KEY_VAR="RF100_${DATASET_ENV}_API_KEY"
WORKSPACE_VAR="RF100_${DATASET_ENV}_WORKSPACE"
PROJECT_VAR="RF100_${DATASET_ENV}_PROJECT"
VERSION_VAR="RF100_${DATASET_ENV}_VERSION"
ROBOFLOW_API_KEY_VALUE="$(get_env_or_default "$API_KEY_VAR" "${ROBOFLOW_API_KEY:-}")"
ROBOFLOW_WORKSPACE_VALUE="$(get_env_or_default "$WORKSPACE_VAR" "${ROBOFLOW_WORKSPACE:-}")"
ROBOFLOW_PROJECT_VALUE="$(get_env_or_default "$PROJECT_VAR" "${ROBOFLOW_PROJECT:-$RAW_SLUG}")"
ROBOFLOW_VERSION_VALUE="$(get_env_or_default "$VERSION_VAR" "${ROBOFLOW_VERSION:-}")"
LOCAL_SOURCE="$RF100_ROOT/$DATASET_NAME"
EXTRACT_ROOT="$RF100_ROOT/_raw/$DATASET_NAME"
DOWNLOAD_DIR="$DATA_ROOT/_downloads/rf100"

SOURCE_ROOT=""
if yolo_dataset_exists "$LOCAL_SOURCE"; then
  SOURCE_ROOT="$LOCAL_SOURCE"
elif [[ -n "$DATASET_URL" ]]; then
  is_true "$DOWNLOAD" || die "Missing $LOCAL_SOURCE and DOWNLOAD=0"
  ZIP_PATH="$DOWNLOAD_DIR/${DATASET_NAME}.zip"
  download_file "$DATASET_URL" "$ZIP_PATH"
  extract_zip "$ZIP_PATH" "$EXTRACT_ROOT"
  SOURCE_ROOT="$(find_yolo_root "$EXTRACT_ROOT" || true)"
elif [[ -n "$ROBOFLOW_API_KEY_VALUE" && -n "$ROBOFLOW_WORKSPACE_VALUE" && -n "$ROBOFLOW_VERSION_VALUE" ]]; then
  is_true "$DOWNLOAD" || die "Missing $LOCAL_SOURCE and DOWNLOAD=0"
  ZIP_PATH="$DOWNLOAD_DIR/${DATASET_NAME}_roboflow_yolov8.zip"
  DATASET_URL="https://universe.roboflow.com/${ROBOFLOW_WORKSPACE_VALUE}/${ROBOFLOW_PROJECT_VALUE}/dataset/${ROBOFLOW_VERSION_VALUE}/download/yolov8?key=${ROBOFLOW_API_KEY_VALUE}"
  download_file "$DATASET_URL" "$ZIP_PATH"
  extract_zip "$ZIP_PATH" "$EXTRACT_ROOT"
  SOURCE_ROOT="$(find_yolo_root "$EXTRACT_ROOT" || true)"
fi

if [[ -z "$SOURCE_ROOT" ]]; then
  die "Missing RF100 YOLO dataset for $DATASET_NAME. Expected $LOCAL_SOURCE, or set $URL_VAR/RF100_ZIP_URL, or set Roboflow API variables."
fi

printf '[dataset] %s -> %s\n' "$DATASET_NAME" "$SOURCE_ROOT"
run_filter_and_train "$SOURCE_ROOT" "$WORK_DIR" "rf100_${DATASET_NAME}"
