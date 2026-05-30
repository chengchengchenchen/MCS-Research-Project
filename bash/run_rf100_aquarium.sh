#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DATASET_NAME="aquarium"
export RAW_SLUG="aquarium-qlnqy"
exec "$SCRIPT_DIR/run_rf100_dataset.sh" "$@"
