#!/bin/bash

ssh_option="-p 20022"
src_dir=$HOME/src/github.com/iimuz/til/machine_learning/torch_time_series_forecasting
ip=root@localhost
dst_dir=/content

rsync -achu --no-o --no-g --delete --progress -e "ssh $ssh_option" $src_dir ${ip}:$dst_dir
rsync -achu --no-o --no-g --delete --progress -e "ssh $ssh_option" sync_local.sh ${ip}:$dst_dir
rsync -achu --no-o --no-g --delete --progress -e "ssh $ssh_option" $HOME/src/github.com/iimuz/config/.config/vscode/colab.code-workspace ${ip}:$dst_dir
