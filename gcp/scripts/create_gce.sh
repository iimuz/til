#!/bin/sh
#
# GCE インスタンスを生成します。
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
MACHINE_TYPE=${2:-f1-micro}
DISK_SIZE=${3:-10GB}

echo "create: name: $INSTANCE_NAME, type: $MACHINE_TYPE, disk: $DISK_SIZE"
gcloud compute \
  instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --image=ubuntu-1804-bionic-v20190204 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=$INSTANCE_NAME

# インスタンスの sshd が起動するまで適当に待つ
echo "wait starting insntance sshd for 20 sec..."
sleep 20s
gcloud compute scp init_gce.sh $INSTANCE_NAME:/home/$USER/

