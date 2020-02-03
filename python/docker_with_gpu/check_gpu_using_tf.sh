#!/bin/bash
#
# tensorflow のイメージを利用して、 tensorflow が GPU を認識していることを確認します。

docker run \
  --rm \
  -t \
  --gpus all \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="$(pwd)",target=/workspace,readonly \
  -w /workspace \
  tensorflow/tensorflow:2.1.0-gpu-py3 \
  python check_gpu_using_tf.py
