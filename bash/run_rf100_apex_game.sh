#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DATASET_NAME="apex_game"
export RAW_SLUG="apex-videogame"
exec "$SCRIPT_DIR/run_rf100_dataset.sh" "$@"
