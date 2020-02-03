#!/bin/bash

docker run \
  --rm \
  -t \
  --gpus all \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="$(pwd)",target=/workspace,readonly \
  -w /workspace \
  pytorch/pytorch:1.4-cuda10.1-cudnn7-devel \
  python torch_check_gpu.py
