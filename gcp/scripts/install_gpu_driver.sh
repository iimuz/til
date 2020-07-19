#!/bin/sh
#
# GCE の GPU 有効インスタンスにおいて、 GPU Driver をインストールする。
# - reference: `https://cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=ja`

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
