#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DATASET_NAME="road_traffic"
export RAW_SLUG="road-traffic"
exec "$SCRIPT_DIR/run_rf100_dataset.sh" "$@"
