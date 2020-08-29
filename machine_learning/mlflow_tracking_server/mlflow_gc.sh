#!/bin/bash
#
# Run mlflow gc.
# Usage: bash mlflow_gc.sh

set -eu

readonly SCRIPT_DIR=$(cd $(dirname $0); pwd)
readonly PROJECT_DIR=$SCRIPT_DIR
readonly BACKEND_STORE_URI="sqlite:///mlruns/tracking.db"

mlflow gc --backend-store-uri=$BACKEND_STORE_URI
