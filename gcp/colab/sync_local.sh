#!/bin/bash

watch_dir=/workspace/hoge
ssh_key=$HOME/.ssh/google_compute_engine
ip=username@hoge.us-west1-b.project
src_dir=/workspace/hoge/
dst_dir=/path/to/output/dir

apt-get install -y --no-install-recommends inotify-tools
while inotifywait -e create,delete,modify,move -r -q ${watch_dir}; do
  rsync -achu --progress --no-o --no-g --delete -e "ssh -i $ssh_key" $src_dir ${ip}:$dst_dir
done
