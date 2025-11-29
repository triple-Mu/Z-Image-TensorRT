# Z-Image-TensorRT

Z-Image's DiT inference with TensorRT-10

## ENV

The project was tested in the following environment:

- Ubuntu 18.04
- NVIDIA Driver 525.125.06
- [`CUDA`](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run)
  11.8
- [`Python`](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_25.9.1-3-Linux-x86_64.sh) 3.10.19
- [
  `PyTorch`](https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=715d3b039a629881f263c40d1fb65edac6786da13bfba221b353ef2371c4da86)
  2.6.0+cu118
- [`Diffusers`](https://github.com/huggingface/diffusers/commit/152f7ca357c066c4af3d1a58cdf17662ef5a2f87) 0.36.0.dev0
- [
  `ONNX`](https://files.pythonhosted.org/packages/8d/eb/30159bb6a108b03f2b7521410369a5bd8d296be3fbf0b30ab7acd9ef42ad/onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl)
  1.19.1
- [
  `TensorRT`](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz)
  10.13.0.35
- [`cudnn-frontend`](https://github.com/NVIDIA/cudnn-frontend/commit/11b51e9c5ad6cc71cd66cb873e34bc922d97d547) 1.16.0

```shell
# Create conda env
conda create -n z-image python=3.10
conda activate z-image

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# Install Diffusers
pip install git+https://github.com/huggingface/diffusers.git@fc337d585309c4b032e8d0180bea683007219df1
# Install ONNX
pip install onnx==1.19.1

# Install TensorRT
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xf TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz
pip install TensorRT-10.13.2.6/python/tensorrt-10.13.2.6-cp310-none-linux_x86_64.whl
export PATH=${PWD}/TensorRT-10.13.2.6/bin:$PATH
export LD_LIBRARY_PATH=${PWD}/TensorRT-10.13.2.6/lib:$LD_LIBRARY_PATH

# Install cudnn-frontend
# tensorrt-plugin is coming soon
```

## CONVERT TO ONNX

Clone the project first:

```shell
git clone https://github.com/triple-Mu/Z-Image-TensorRT.git
cd Z-Image-TensorRT
```

Here are some scripts to test exporting onnx:

- [`1-export-dit-directly.py`](./step_by_step/1-export-dit-directly.py)

Empty script, proceed to step 2.

- [`2-remove-complex-op.py`](./step_by_step/2-remove-complex-op.py)

```shell
python step_by_step/2-remove-complex-op.py --model_path Tongyi-MAI/Z-Image-Turbo --onnx_path transformer_step2.onnx
```

- [`3-merge-qkv-projection.py`](./step_by_step/3-merge-qkv-projection.py)

```shell
python step_by_step/3-merge-qkv-projection.py --model_path Tongyi-MAI/Z-Image-Turbo --onnx_path transformer_step3.onnx
```

Advanced: Merging QKV GEMM reduces kernel launches and increases throughput.

- [`4-cudnn-attention-plugin.py`](./step_by_step/4-cudnn-attention-plugin.py)

```shell
python step_by_step/4-cudnn-attention-plugin.py --model_path Tongyi-MAI/Z-Image-Turbo --onnx_path transformer_step4.onnx
```

Advanced: Replacing sdpa with cudnn-attention, it results in a significant improvement on A100 GPU.

## CONVERT TO TensorRT

After convert `ZImageTransformer2DModel` to ONNX, the tensorrt engine can be built by `trtexec`.

Refer to [`2-build_engine.sh`](./scripts/2-build_engine.sh)

Set up `TENSORRT_ROOT` `ONNX_PATH` and `ENGINE_PATH` first, and the min/opt/max shape also can be modified by yourself.

Then run:

```shell
bash scripts/2-build_engine.sh
```

## RUNNING TensorRT Pipeline!

After convert ONNX to Engine, the pipeline can be built with Diffusers's pipeline.

Refer to [`run_trt_pipeline.py`](./run_trt_pipeline.py)

Run:

```shell
python run_trt_pipeline.py --model_path Tongyi-MAI/Z-Image-Turbo --trt_path transformer_step2.engine
```

Then the example output image will be saved at [`example.png`](./example.png).

## CUDNN-ATTENTION Plugin!

Build CUDNN-Attention plugin!

Refer to [`build.sh`](./plugins/build.sh)

Set up `TENSORRT_ROOT` `CUDNN_ROOT` and `ARCH` first.

Then run:

```shell
bash plugins/build.sh
```

Build engine with CUDNN-Attention plugin:

Refer to [`4-build_engine_cudnn_attention.sh`](./scripts/4-build_engine_cudnn_attention.sh)

Inference:

```shell
python run_trt_pipeline.py --model_path Tongyi-MAI/Z-Image-Turbo --trt_path transformer_step4.engine --plugin_path ./libcudnn_attention_plugin.so
```
