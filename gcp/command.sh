#!/bin/bash
#
# GCPを操作するためのコマンド群をまとめたスクリプト.

set -eu

# スクリプトのコマンドの説明。
# 詳細説明は各コマンドの説明に記載する。
function _usage() {
  cat <<EOF
$(basename $0) is a tool for gcp.

Usage:
  $(basename $0) [command] [options]

Commands:
  help:     print this.
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


# GCE関連のコマンドのエントリポイント。
function _instance() {
  readonly SUB_COMMAND=$1
  shift
  readonly SUB_OPTIONS="$@"

  case "$SUB_COMMAND" in
    "help" ) _instance_usage;;
    "create" ) _instance_create $SUB_OPTIONS;;
  esac
}

# GCEコマンドの説明。
function _instance_usage() {
  cat <<EOF
$(basename $0) instance command is a tool for gce instance.

Usage:
  $(basename $0) instance [command] [options]

Commands:
  help: print this.
  create: create gce instance.
EOF
}

# GCEにインスタンスを生成。
function _instance_create() {
  instance_name="dev"
  machine_type="n1-standard-4"
  disk_size="50GB"
  gpu_option=""
  attach_disk=""

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
      "--size" )
        _check_option_arg $1 $2
        disk_size=$2
        shift 2
        ;;
      "--type" )
        _check_option_arg $1 $2
        machine_type=$2
        shfit 2
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

  echo "create: name: $instance_name, type: $machine_type, disk: $disk_size, gpu: $gpu_option, attach disk: $attach_disk"
  gcloud compute \
    instances create $instance_name \
    --machine-type=$machine_type \
    --subnet=default \
    --network-tier=PREMIUM \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --image=ubuntu-2004-focal-v20201211 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=$disk_size \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=$instance_name \
    $gpu_option \
    $attach_disk

  echo "wait starting insntance sshd for 20 sec..."
  sleep 20s
  gcloud compute scp scripts/init_gce.sh $INSTANCE_NAME:/home/$USER/
}

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

# スクリプトの実行
readonly COMMAND=$1
shift
readonly OPTIONS="$@"

case "$COMMAND" in
  "help" ) _usage;;
  "instance" ) _instance $OPTIONS;;
  "disk" ) _disk $OPTIONS;;
esac
