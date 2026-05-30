#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/lib/pipeline_common.sh"

: "${VOC_DEVKIT_ROOT:=$DATA_ROOT/VOCdevkit}"
: "${WORK_DIR:=$CODE_REFACTORED_ROOT/workdir/voc_seed${SEED}}"

DOWNLOAD_DIR="$DATA_ROOT/_downloads/voc"
DATASET_ROOT="$WORK_DIR/dataset"
CONFIG_DIR="$WORK_DIR/configs"

mkdir -p "$CONFIG_DIR"

if [[ ! -d "$VOC_DEVKIT_ROOT/VOC2007" || ! -d "$VOC_DEVKIT_ROOT/VOC2012" ]]; then
  is_true "$DOWNLOAD" || die "Missing VOCdevkit and DOWNLOAD=0"
  download_file "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" "$DOWNLOAD_DIR/VOCtrainval_06-Nov-2007.tar"
  download_file "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" "$DOWNLOAD_DIR/VOCtest_06-Nov-2007.tar"
  download_file "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" "$DOWNLOAD_DIR/VOCtrainval_11-May-2012.tar"
  extract_tar "$DOWNLOAD_DIR/VOCtrainval_06-Nov-2007.tar" "$DATA_ROOT"
  extract_tar "$DOWNLOAD_DIR/VOCtest_06-Nov-2007.tar" "$DATA_ROOT"
  extract_tar "$DOWNLOAD_DIR/VOCtrainval_11-May-2012.tar" "$DATA_ROOT"
fi

write_voc_prepare_config "$CONFIG_DIR/prepare_dataset.json" "$VOC_DEVKIT_ROOT" "$DATASET_ROOT"
run "$PYTHON" "$CODE_REFACTORED_ROOT/scripts/prepare_dataset.py" --config "$CONFIG_DIR/prepare_dataset.json"
run_filter_and_train "$DATASET_ROOT" "$WORK_DIR" "voc"
