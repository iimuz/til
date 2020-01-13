#!/bin/bash
#
# GCE インスタンス起動時の初期化処理を実行します。
# 対象のインスタンスは Ubuntu であると仮定しています。

sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt clean

