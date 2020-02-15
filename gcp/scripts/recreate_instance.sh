#!/bin/bash
#
# GCE のインスタンスをディスクのみ既存のデータを利用して再構築します。
# gcloud コマンドのデフォルト値を参照し、
# リージョンなどを設定する前提としています。

# * preemptible f1-micro
#     * 0.6 GB Memory
#     * 0.006 $ / min, 4.17 $ / month
# * preemptible g1-small
#     * 1.7 GB Memory
#     * 0.011 $ / min, 7.82 $ / month
# * preemptible n1-standard-1
#     * 3.75 GB Memory
#     * 0.014 $ / min, 10.19 $ / month
INSTANCE_NAME=${1:-dev}
MACHINE_TYPE=${2:-n1-standard-1}
DISK_NAME=${3:-${INSTANCE_NAME}}

echo "create: name: $INSTANCE_NAME, type: $MACHINE_TYPE, disk: $DISK_NAME"
gcloud compute instances delete $INSTANCE_NAME --keep-disks=all
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --disk=name=${DISK_NAME},device-name=${DISK_NAME},mode=rw,boot=yes \
  --reservation-affinity=any

