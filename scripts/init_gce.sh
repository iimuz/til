#!/bin/bash
#
# GCE インスタンス起動時の初期化処理を実行します。
# 対象のインスタンスは Ubuntu であると仮定しています。

sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt clean

sudo apt install -y --no-install-recommeds git

mkdir -p src/github.com/iimuz && pushd $_
git clone https://github.com/iimuz/dotfiles.git
pushd dotfiles
bash setup_ubuntu.sh
popd
popd

