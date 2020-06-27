#!/bin/bash

watch_dir=/content/torch_time_series_forecasting
ssh_key=$HOME/.ssh/google_compute_engine
ip=disii@free.us-west1-b.develop-231513
src_dir=/content/torch_time_series_forecasting
dst_dir=/home/disii/src/github.com/iimuz/til/machine_learning

apt-get install -y --no-install-recommends inotify-tools
while inotifywait -e create,delete,modify,move -r -q ${watch_dir}; do
  rsync -achu --progress --no-o --no-g --delete -e "ssh -i $ssh_key" $src_dir ${ip}:$dst_dir
done
