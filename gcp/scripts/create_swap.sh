#!/bin/bash
#
# swap ファイルを作成し、有効化します。

swapfile=/swapfile

sudo dd if=/dev/zero of=$swapfile bs=1M count=4000
sudo chmod 600 $swapfile
sudo mkswap $swapfile
sudo swapon $swapfile

sudo echo "\n$swapfile none swap sw 0 0\n" >> /etc/fstab
