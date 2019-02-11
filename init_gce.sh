#!/bin/bash
#
# GCE インスタンス起動時の初期化処理を実行します。

apt update
apt upgrade -y
apt autoremove -y
apt clean

apt install -y --no-install-recommeds git

mkdir -p src/github.com/iimuz && pushd $_
git clone https://github.com/iimuz/dotfiles.git
pushd dotfiles
bash setup_gce.sh
popd
popd

