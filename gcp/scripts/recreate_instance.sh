#!/bin/bash
#
# GCE のインスタンスをディスクのみ既存のデータを利用して再構築します。
# gcloud コマンドのデフォルト値を参照し、
# リージョンなどを設定する前提としています。
# * preemptible f1-micro: 1 vCPU, 0.614 GB Memory, 0.006 $/hour, 4.17 $/month
# * preemptible e2-micro: 2 vCPU, 1.0 GB Memory, 0.004 $/hour, 2.87 $/month
# * preemptible e2-medium: 2 vCPU, 4.0 GB Memory, 0.014 $/hour, 9.93 $/month
# * preemptible n1-standard-1: 2 vCPU, 7.5 GB Memory, 0.027 $/hour, 19.35 $/month
# * preemptible n2-standard-2: 2 vCPU, 8 GB Memory, 0.032 $/hour, 22.72 $/month
# * preemptible e2-standard-2: 2 vCPU, 8.0 GB Memory, 0.027/hour, 19.35 $/month
# * preemptible n1-standard-4: 4 vCPU, 15 GB Memory, 0.054 $/hour, 38.69 $/month
# * preemptible n2-standard-4: 4 vCPU, 16 GB Memory, 0.063 $/hour, 45.45 $/month
# * preemptible e2-standard-4: 4 vCPU, 16.0 GB Memory, 0.052/hour, 37.65 $/month
# * Tesla T4: 0.112 $/hour, 80.30 $/month
# * Storage: 0.0007 $/(hour * 10 GB), 0.52 $/(month * 10 GB)
# * example
#   * preemptible e2-medium, 50GB Storage: 0.016 $/hour, 12.01 $/month
#   * preemptible e2-standard-2, 50GB Storage: 0.029 $/hour, 21.43 $/month
#   * preemptible e2-standard-4, 50GB Storage: 0.055 $/hour, 37.65 $/month
#   * preemptible n1-standard-2, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month
#   * preemptible n1-standard-4, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month

INSTANCE_NAME=${1:-gpu}
MACHINE_TYPE=${2:-e2-medium}
DISK_NAME=${3:-${INSTANCE_NAME}}

echo "create: name: $INSTANCE_NAME, type: $MACHINE_TYPE, disk: $DISK_NAME"
gcloud compute instances delete $INSTANCE_NAME --keep-disks=all
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --disk=name=${DISK_NAME},device-name=${DISK_NAME},mode=rw,boot=yes \
  --reservation-affinity=any \
  --preemptible
