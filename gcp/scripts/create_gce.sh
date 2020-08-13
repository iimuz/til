#!/bin/sh
#
# GCE インスタンスを生成します。
# gcloud コマンドのデフォルト値を参照し、
# リージョンなどを設定する前提としています。

# * preemptible f1-micro: 1 vCPU, 0.614 GB Memory, 0.006 $/hour, 4.17 $/month
# * preemptible n1-standard-1: 1 vCPU, 3.75 GB Memory, 0.014 $/hour, 10.19 $/month
# * preemptible e2-micro: 2 vCPU, 1.0 GB Memory, 0.004 $/hour, 2.87 $/month
# * preemptible e2-medium: 2 vCPU, 4.0 GB Memory, 0.014 $/hour, 9.93 $/month
# * preemptible e2-standard-2: 2 vCPU, 8.0 GB Memory, 0.027/hour, 19.35 $/month
# * Tesla T4: 0.112 $/hour, 80.30 $/month
# * Storage: 0.0007 $/(hour * 10 GB), 0.52 $/(month * 10 GB)
# * example
#   * preemptible e2-medium, 50GB Storage: 0.016 $/hour, 12.01 $/month
#   * preemptible e2-standard, 50GB Storage: 0.029 $/hour, 21.43 $/month
#   * preemptible n1-standard-2, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month
INSTANCE_NAME=${1:-gpu}
MACHINE_TYPE=${2:-n1-standard-4}
DISK_SIZE=${3:-200GB}
ATATCH_DISK=${4:-gpu-home}
USE_GPU=${5:-true}

gpu_option=""
if "${USE_GPU}"; then
  echo "use gpu..."
  gpu_option="--accelerator=type=nvidia-tesla-t4,count=1"
fi

echo "create: name: $INSTANCE_NAME, type: $MACHINE_TYPE, disk: $DISK_SIZE"
gcloud compute \
  instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --image=ubuntu-1804-bionic-v20200701 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=$INSTANCE_NAME \
  --disk=name=$ATATCH_DISK,device-name=$ATATCH_DISK,mode=rw,boot=no \
  $gpu_option

# インスタンスの sshd が起動するまで適当に待つ
echo "wait starting insntance sshd for 20 sec..."
sleep 20s
gcloud compute scp init_gce.sh $INSTANCE_NAME:/home/$USER/
gcloud compute scp init_home.sh $INSTANCE_NAME:/home/$USER/
