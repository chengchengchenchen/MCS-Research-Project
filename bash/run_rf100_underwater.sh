#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DATASET_NAME="underwater"
export RAW_SLUG="underwater-objects-5v7p8"
exec "$SCRIPT_DIR/run_rf100_dataset.sh" "$@"
