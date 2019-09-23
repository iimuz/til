#!/bin/sh
#
# 生成済みのインスタンスのマシンタイプを変更します。
# 変更するためには、一度インスタンスを停止する必要があります。
#
# See: https://cloud.google.com/compute/docs/instances/changing-machine-type-of-stopped-instance

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

echo "chnage: name: $INSTANCE_NAME, type: $MACHINE_TYPE"
gcloud compute instances set-machine-type $INSTANCE_NAME --machine-type $MACHINE_TYPE

