#!/bin/bash
#
# 接続されているディスクをホームディレクトリとして設定します。

DEVICE=google-gpu-home

cd /
sudo rm -rf /home/*
echo UUID=`sudo blkid -s UUID -o value /dev/disk/by-id/$DEVICE` /home ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
sudo reboot
