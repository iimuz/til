#!/bin/bash
#
# GCP関連を操作するためのスクリプト

set -eu

if ! type gcloud > /dev/null 2>&1; then
  echo "gcloud command not found."
  exit 1
fi

if ! type gsutil > /dev/null 2>&1; then
  echo "gsutil command not found."
  exit 1
fi

# script settings
readonly SCRIPT_PATH=${BASH_SOURCE:-0}
readonly SCRIPT_DIR=$(cd $(dirname $SCRIPT_PATH); pwd)
readonly PROJECT_DIR=$(cd $SCRIPT_DIR/..; pwd)

# GCP settings
readonly GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-"hogehoge"}
readonly GCLOUD_PROJECT=${GCLOUD_PROJECT_ID:-$(gcloud config get-value project 2> /dev/null)}
readonly GCLOUD_REGION=${GCLOUD_REGION:-$(gcloud config get-value compute/region 2> /dev/null)}

function _create_gcs_bucket() {
  if ! type gsutil > /dev/null 2>&1; then
    echo "gsutil command not found."
    exit 1;
  fi

  NAME="gs://$GCS_BUCKET_NAME"
  PROJECT=$GCLOUD_PROJECT
  STORAGE_CLASS="Standard"
  LOCATION=$GCLOUD_REGION

  gsutil mb -p $GCLOUD_PROJECT -c $STORAGE_CLASS -l $LOCATION -b on $NAME
}

readonly SUB_COMMAND=${1}
shift
readonly OPTIONS=$*

case "$SUB_COMMAND" in
  "gcs" ) _create_gcs_bucket $OPTIONS;;
esac
