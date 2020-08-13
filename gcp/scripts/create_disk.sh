#!/bin/sh
#
# GCE にディスクを生成します。
#
# 参考資料:
#  - Google Cloud ゾーン永続ディスクの追加またはサイズ変更
#    - https://cloud.google.com/compute/docs/disks/add-persistent-disk)

DISK_NAME=dev-home
DISK_SIZE=10  # GB
DISK_TYPE=pd-standard  # or pd-ssd

gcloud compute disks create $DISK_NAME \
  --size $DISK_SIZE \
  --type $DISK_TYPE \
  --zone asia-northeast1-b
