#!/bin/bash
#
# GCPを操作するためのコマンド群をまとめたスクリプト.

set -eu

readonly SCRIPT_PATH=${0}
readonly SCRIPT_DIR=$(cd $(dirname $SCRIPT_PATH); pwd)

# スクリプトのコマンドの説明。
# 詳細説明は各コマンドの説明に記載する。
function _usage() {
  cat <<EOF
$(basename $0) is a tool for gcp.

Usage:
  $(basename $0) [command] [options]

Commands:
  help:     print this.
  init:     initialize command for instance.
  instance: command related to gce instance.
  disk:     command related to gce instance disk.
EOF
}

# オプションが引数を持っているか確認する。
# 引数が設定されていれば正常終了し、引数がなければエラーで終了する。
function _check_option_arg() {
  if [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
    echo "option requires an argument -- $1" 1>&2
    exit 1
  fi
}

# GCEのdisk関連コマンドのエントリポイント
function _disk() {
  readonly SUB_COMMAND=$1
  shift
  readonly SUB_OPTIONS="$@"

  case "$SUB_COMMAND" in
    "help" ) _disk_usage;;
    "create" ) _disk_create $SUB_OPTIONS;;
  esac
}

# GCEのディスクコマンドのヘルプ.
function _disk_usage() {
  cat <<EOF
$(basename $0) disk command is a tool for gce disk.

Usage:
$(basename $0) disk [command] [options]

Commands:
help: print this.
create: create gce disk.

Notes:
- Google Cloud ゾーン永続ディスクの追加またはサイズ変更
  - https://cloud.google.com/compute/docs/disks/add-persistent-disk)
EOF
}

# GCEのディスク生成
function _disk_create() {
  disk_name="dev"
  disk_size="50"
  disk_type="pd-standard"
  zone="asia-northeast1-b"

  param=""
  for OPT in "$@"
  do
    case "$OPT" in
      "--help" | "-h" )
        _disk_create_usage
        exit 0
        ;;
      "--name" )
        _check_option_arg $1 $2
        disk_name="$2"
        shift 2
        ;;
      "--size" )
        _check_option_arg $1 $2
        disk_size="$2"
        shift 2
        ;;
      "--ssd" )
        disk_type="pd-ssd"
        shift 1
        ;;
      "--zone" )
        _check_option_arg $1 $2
        zone="$2"
        shift 2
        ;;
      "--" | "-" )
        shift 1
        param="$param $@"
        break;;
      "-"* )
        echo "illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
        exit 1
        ;;
      * )
        if [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]; then
          param="$param $1"
          shift 1
        fi
        ;;
    esac
  done

  gcloud compute disks create $disk_name \
    --size $disk_size \
    --type $disk_type \
    --zone $zone
}

# GCE diskの生成コマンドのヘルプ.
function _disk_create_usage() {
  cat <<EOF
$(basename $0) disk create command creates gce disk.

Usage:
$(basename $0) disk create [options]

Commands:
--help, -h:  print this.
--name name: disk name.
--size size: disk size (GB).
--ssd:       use ssd storage. if not, use hdd.
--zone zone: set zone.
EOF
}

# インスタンスの初期化コマンドのエントリポイント.
function _init() {
  readonly SUB_COMMAND=$1
  shift
  readonly SUB_OPTIONS="$@"

  case "$SUB_COMMAND" in
    "docker-gpu" ) _init_docker_gpu;;
    "dotfiles" ) _init_dotfiles;;
    "gpu" ) _init_gpu;;
    "help" ) _init_usage;;
    "swap" ) _init_swap;;
    "update" ) _init_update;;
  esac
}

# インスタンスの初期化関連コマンドのヘルプ.
function _init_usage() {
  cat <<EOF
$(basename $0) instance command is a tool for gce instance.

Usage:
$(basename $0) instance [command] [options]

Commands:
docker-gpu: install docker tools for nvidia gpu.
dotfiles:   set dotfiles.
gpu:        initialize gpu driver.
help:       print this.
swap:       create swap file and activate.
update:     initialize instance.
EOF
}

# docker 環境で gpu が有効にできないときの追加インストール
function _init_docker_gpu() {
  # install nvidia container
  curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
  sudo apt-get update

  sudo apt-get install -y nvidia-container-runtime

  docker run --rm -it --gpus=all ubuntu:18.04 nvidia-smi
}

# dotfilesの最低限の設定を実施.
function _init_dotfiles() {
  readonly DOT_PARENT_DIR="src/github.com/iimuz"

  sudo apt install -y --no-install-recommends unzip
  mkdir -p $DOT_PARENT_DIR
  pushd $DOT_PARENT_DIR
  git clone https://github.com/iimuz/dotfiles.git
  pushd dotfiles
  bash setup.sh
  popd
  popd
}


# Ubuntu 院スタンにおいてGPUドライバをインストールする.
# - reference: `https://cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=ja`
function _init_gpu() {
  # Ubuntu 18.04
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  rm cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

  # install driver
  sudo apt-get update
  sudo apt-get install -y cuda

  # check driver
  nvidia-smi
}

# swapファイルを生成し有効化します.
function _init_swap() {
  readonly SWAPFILE=/swapfile

  sudo dd if=/dev/zero of=$SWAPFILE bs=1M count=4000
  sudo chmod 600 $SWAPFILE
  sudo mkswap $SWAPFILE
  sudo swapon $SWAPFILE

  sudo echo "\n$SWAPFILE none swap sw 0 0\n" >> /etc/fstab
}

# インスタンスがUbuntuと仮定して生成後の初期動作
function _init_update() {
  sudo apt update
  sudo apt upgrade -y
  sudo apt autoremove -y
  sudo apt clean
}

# インスタンスの生成関連のコマンドのエントリポイント。
function _instance() {
  readonly SUB_COMMAND=$1
  shift
  readonly SUB_OPTIONS="$@"

  case "$SUB_COMMAND" in
    "change" ) _instance_change $SUB_OPTIONS;;
    "create" ) _instance_create $SUB_OPTIONS;;
    "free" ) _instance_create_always_free;;
    "help" ) _instance_usage;;
    "home" ) _instance_home $SUB_OPTIONS;;
    "recreate" ) _instance_recreate $SUB_OPTIONS;;
  esac
}

# GCEコマンドの説明。
function _instance_usage() {
  cat <<EOF
$(basename $0) instance command is a tool for gce instance.

Usage:
$(basename $0) instance [command] [options]

Commands:
change:          change instance machine type. (options: --name name, --type type)
create:          create gce instance.
free:            create always free instance.
help:            print this.
home:            set the disk as home direcotry. (option: --device name)
recreate:        recreate gce instance.
EOF
}

# 生成済みのインスタンスのマシンタイプのみ変更する。
# 変更するためにはインスタンスを停止する必要があります。
function _instance_change() {
  instance_name="dev"
  machine_type="n1-standard-4"

  param=""
  for OPT in "$@"
  do
    case "$OPT" in
      "--name" )
        _check_option_arg $1 $2
        instance_name=$2
        shift 2
        ;;
      "--type" )
        _check_option_arg $1 $2
        machine_type=$2
        shift 2
        ;;
      "--" | "-" )
        shift 1
        param="$param $@"
        break;;
      "-"* )
        echo "illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
        exit 1
        ;;
    esac
  done

  echo "chnage: name: $instance_name, type: $machine_type"
  gcloud compute instances set-machine-type $instance_name --machine-type $machine_type
}

# GCEにインスタンスを生成。
function _instance_create() {
  instance_name="dev"
  machine_type="n1-standard-4"
  disk_size="50GB"
  gpu_option=""
  attach_disk=""
  preemptible="--preemptible"

  param=""
  for OPT in "$@"
  do
    case "$OPT" in
      "--attach-disk" )
        _check_option_arg $1 $2
        attach_disk="--disk=name=$2,device-name=$2,mode=rw,boot=no"
        shift 2
        ;;
      "--gpu" )
        gpu_option="--accelerator=type=nvidia-tesla-t4,count=1"
        shift 1
        ;;
      "--help" | "-h" )
        _instance_create_usage
        exit 0
        ;;
      "--name" )
        _check_option_arg $1 $2
        instance_name=$2
        shift 2
        ;;
      "--no-preemptible" )
        preemptible=""
        shift 1
        ;;
      "--size" )
        _check_option_arg $1 $2
        disk_size=$2
        shift 2
        ;;
      "--type" )
        _check_option_arg $1 $2
        machine_type=$2
        shift 2
        ;;
      "--" | "-" )
        shift 1
        param="$param $@"
        break;;
      "-"* )
        echo "illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
        exit 1
        ;;
    esac
  done

  echo "create: name: $instance_name, type: $machine_type, disk: $disk_size, gpu: $gpu_option, attach disk: $attach_disk"
  gcloud compute \
    instances create $instance_name \
    --machine-type=$machine_type \
    --subnet=default \
    --network-tier=PREMIUM \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --image=ubuntu-1804-bionic-v20210129 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=$disk_size \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=$instance_name \
    $preemptible \
    $gpu_option \
    $attach_disk

  echo "wait starting insntance sshd for 20 sec..."
  sleep 20s
  gcloud compute scp $SCRIPT_PATH $INSTANCE_NAME:/home/$USER/
}

# instance createのヘルプ.
function _instance_create_usage() {
  cat <<EOF
$(basename $0) instance create command creates gce instance.

Usage:
$(basename $0) instance create [options]

Commands:
--attach-disk name: attach {name} disk. e.g. misc
--gpu:              use gpu.
--help, -h:         print this.
--name name:        instance {name}. e.g. dev
--no-preemptible:   do not preemptible instance.
--size size:        disk {size}. e.g. 50GB
--type type:        machine type. e.g. n1-standard-4

Notes:
gcloud コマンドのデフォルト値を参照し、リージョンなどを設定する前提としています。

- preemptible f1-micro: 1 vCPU, 0.614 GB Memory, 0.006 $/hour, 4.17 $/month
- preemptible n1-standard-1: 1 vCPU, 3.75 GB Memory, 0.014 $/hour, 10.19 $/month
- preemptible e2-micro: 2 vCPU, 1.0 GB Memory, 0.004 $/hour, 2.87 $/month
- preemptible e2-medium: 2 vCPU, 4.0 GB Memory, 0.014 $/hour, 9.93 $/month
- preemptible e2-standard-2: 2 vCPU, 8.0 GB Memory, 0.027/hour, 19.35 $/month
- Tesla T4: 0.112 $/hour, 80.30 $/month
- Storage: 0.0007 $/(hour * 10 GB), 0.52 $/(month * 10 GB)
- example
  - preemptible e2-medium, 50GB Storage: 0.016 $/hour, 12.01 $/month
  - preemptible e2-standard, 50GB Storage: 0.029 $/hour, 21.43 $/month
  - preemptible n1-standard-2, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month
EOF
}

# us-west1 などで always free プランでのインスタンスを生成します。
# 設定値は、現状で free プランで可能な最大値にしています。
# zone/region は、 active な configurations を always free プランが可能な west1 などにしてください。
function _instance_create_always_free() {
  readonly INSTANCE_NAME=free
  readonly MACHINE_TYPE=f1-micro
  readonly DISK_SIZE=30GB

  _instance_create \
    --name $INSTANCE_NAME \
    --type $MACHINE_TYPE \
    --size $DISK_SIZE \
    --no-preemptible
}

# instanceを生成後、指定したディスクをホームディレクトリとして設定し再起動します。
function _instance_home() {
  device_name="home"

  param=""
  for OPT in "$@"
  do
    case "$OPT" in
      "--help" | "-h" )
        echo "no help."
        exit 0
        ;;
      "--name" )
        _check_option_arg $1 $2
        device_name=$2
        shift 2
        ;;
      "--" | "-" )
        shift 1
        param="$param $@"
        break;;
      "-"* )
        echo "illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
        exit 1
        ;;
    esac
  done

  cd /
  sudo rm -rf /home/*
  echo UUID=`sudo blkid -s UUID -o value /dev/disk/by-id/$device_name` /home ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
  sudo reboot
}

# 生成済みのインスタンスを削除して、同一ディスクで別のインスタンスを生成する。
function _instance_recreate() {
  instance_name="dev"
  machine_type="n1-standard-4"
  disk_size="50GB"
  gpu_option=""
  attach_disk=""
  disk_name=""
  preemptible="--preemptible"

  param=""
  for OPT in "$@"
  do
    case "$OPT" in
      "--attach-disk" )
        _check_option_arg $1 $2
        attach_disk="--disk=name=$2,device-name=$2,mode=rw,boot=no"
        shift 2
        ;;
      "--disk-name" )
        _check_option_arg $1 $2
        disk_name=$2
        shift 2
        ;;
      "--gpu" )
        gpu_option="--accelerator=type=nvidia-tesla-t4,count=1"
        shift 1
        ;;
      "--help" | "-h" )
        _instance_recreate_usage
        exit 0
        ;;
      "--name" )
        _check_option_arg $1 $2
        instance_name=$2
        shift 2
        ;;
      "--no-preemptible" )
        preemptible=""
        shift 1
        ;;
      "--type" )
        _check_option_arg $1 $2
        machine_type=$2
        shift 2
        ;;
      "--" | "-" )
        shift 1
        param="$param $@"
        break;;
      "-"* )
        echo "illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
        exit 1
        ;;
    esac
  done
  if [ $disk_name = "" ]; then
    disk_name=$instance_name
  fi

  echo "recreate: name: $instance_name, type: $machine_type, disk: $disk_name"
  gcloud compute instances delete $instance_name --keep-disks=all
  gcloud compute instances create $instance_name \
    --machine-type=$machine_type \
    --subnet=default \
    --network-tier=PREMIUM \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --disk=name=${disk_name},device-name=${disk_name},mode=rw,boot=yes \
    --reservation-affinity=any \
    $preemptible \
    $attach_disk \
    $gpu_option
}

# instance recreateのヘルプ.
function _instance_recreate_usage() {
  cat <<EOF
$(basename $0) instance create command creates gce instance.

Usage:
$(basename $0) instance create [options]

Commands:
--attach-disk name: attach {name} disk. e.g. misc
--disk-name name:   disk name. if using different disk name from instance name.
--gpu:              use gpu.
--help, -h:         print this.
--name name:        instance {name}. e.g. dev
--no-preemptible:   do not preemptible instance.
--type type:        machine type. e.g. n1-standard-4

Notes:
gcloud コマンドのデフォルト値を参照し、リージョンなどを設定する前提としています。

- preemptible f1-micro: 1 vCPU, 0.614 GB Memory, 0.006 $/hour, 4.17 $/month
- preemptible e2-micro: 2 vCPU, 1.0 GB Memory, 0.004 $/hour, 2.87 $/month
- preemptible e2-medium: 2 vCPU, 4.0 GB Memory, 0.014 $/hour, 9.93 $/month
- preemptible n1-standard-1: 2 vCPU, 7.5 GB Memory, 0.027 $/hour, 19.35 $/month
- preemptible n2-standard-2: 2 vCPU, 8 GB Memory, 0.032 $/hour, 22.72 $/month
- preemptible e2-standard-2: 2 vCPU, 8.0 GB Memory, 0.027/hour, 19.35 $/month
- preemptible n1-standard-4: 4 vCPU, 15 GB Memory, 0.054 $/hour, 38.69 $/month
- preemptible n2-standard-4: 4 vCPU, 16 GB Memory, 0.063 $/hour, 45.45 $/month
- preemptible e2-standard-4: 4 vCPU, 16.0 GB Memory, 0.052/hour, 37.65 $/month
- Tesla T4: 0.112 $/hour, 80.30 $/month
- Storage: 0.0007 $/(hour x 10 GB), 0.52 $/(month x 10 GB)
- example
  - preemptible e2-medium, 50GB Storage: 0.016 $/hour, 12.01 $/month
  - preemptible e2-standard-2, 50GB Storage: 0.029 $/hour, 21.43 $/month
  - preemptible e2-standard-4, 50GB Storage: 0.055 $/hour, 37.65 $/month
  - preemptible n1-standard-2, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month
  - preemptible n1-standard-4, 50GB Storage, Tesla T4: 0.14 $/hour, 102.25 $/month
EOF
}

# スクリプトの実行
readonly COMMAND=$1
shift
readonly OPTIONS="$@"

case "$COMMAND" in
  "help" ) _usage;;
  "init" ) _init $OPTIONS;;
  "instance" ) _instance $OPTIONS;;
  "disk" ) _disk $OPTIONS;;
esac
