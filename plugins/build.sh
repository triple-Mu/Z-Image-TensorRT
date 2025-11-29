#!/bin/bash

TENSORRT_ROOT="/data/TensorRT-10.13.2.6" # must be a real path
CUDNN_ROOT="" # must be a real path
ARCH="arch=compute_80,code=sm_80"

cd plugins

git clone https://github.com/NVIDIA/cudnn-frontend.git

nvcc cudnn_attention.cpp \
  -I$PWD \
  -I${TENSORRT_ROOT}/include \
  -L${TENSORRT_ROOT}/lib \
  -I${CUDNN_ROOT}/include \
  -L${CUDNN_ROOT}/lib \
  -I$PWD/cudnn-frontend/include \
  -lcudart -lcudnn -lnvinfer \
  -shared \
  -Xcompiler "-std=c++17 -fPIC -DNDEBUG" \
  -gencode=${ARCH} \
  -O3 \
  --use_fast_math \
  -o libcudnn_attention_plugin.so