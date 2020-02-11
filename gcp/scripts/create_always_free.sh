#!/bin/sh
#
# us-west1 などで always free プランでのインスタンスを生成します。
# 設定値は、現状で free プランで可能な最大値にしています。
# zone/region は、 active な configurations を always free プランが可能な west1 などにしてください。

INSTANCE_NAME=free
MACHINE_TYPE=f1-micro
DISK_SIZE=30GB
echo "create: name: $INSTANCE_NAME, type: $MACHINE_TYPE, disk: $DISK_SIZE"

gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --maintenance-policy=MIGRATE \
  --image=ubuntu-1804-bionic-v20200129a \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=$INSTANCE_NAME \
  --reservation-affinity=any

echo "wait starting insntance sshd for 20 sec..."
sleep 20s
gcloud compute scp init_gce.sh $INSTANCE_NAME:/home/$USER/
