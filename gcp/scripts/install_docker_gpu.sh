#!/bin/sh
#
# docker 環境で gpu が有効にできないときの追加インストール

# install nvidia container
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update

sudo apt-get install -y nvidia-container-runtime

docker run --rm -it --gpus=all ubuntu:18.04 nvidia-smi
