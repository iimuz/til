#!/bin/sh
#
# Run jupyterlab
# Usage: bash run_jupyter.sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(cd $SCRIPT_DIR/..; pwd)
CONFIG="$PROJECT_DIR/scripts/jupyter_notebook_config.py"

jupyter lab --config=$CONFIG
