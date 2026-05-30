#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DATASET_NAME="cotton"
export RAW_SLUG="cotton-20xz5"
exec "$SCRIPT_DIR/run_rf100_dataset.sh" "$@"
