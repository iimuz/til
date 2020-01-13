#!/bin/bash
#
# 接続されているディスクをホームディレクトリとして設定します。

DEVICE_ID=sdb

cd /
sudo rm -rf /home/*
echo UUID=`sudo blkid -s UUID -o value /dev/$DEVICE_ID` /home ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
sudo reboot

