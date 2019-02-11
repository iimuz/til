#!/bin/sh
#
# GCE インスタンスを生成します。
# gcloud コマンドのデフォルト値を参照し、
# リージョンなどを設定する前提としています。

INSTANCE_NAME=instance-1
DISK_SIZE=10GB
MACHINE_TYPE=f1-micro

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

