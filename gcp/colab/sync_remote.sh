#!/bin/bash

ssh_option="-p 20022"
src_dir=/path/to/source
ip=root@localhost
dst_dir=/workspace

rsync -achu --no-o --no-g --delete --progress -e "ssh $ssh_option" $src_dir ${ip}:$dst_dir
rsync -achu --no-o --no-g --delete --progress -e "ssh $ssh_option" sync_local.sh ${ip}:$dst_dir
