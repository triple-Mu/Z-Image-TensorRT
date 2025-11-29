from typing import Tuple, Any, Optional, List, Dict, Union

import argparse
import ctypes
import tensorrt as trt
import torch
import torch.nn as nn
from diffusers import ZImageTransformer2DModel
from diffusers.pipelines import ZImagePipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class ZImageEmbedRope(nn.Module):
    def __init__(
            self,
            axes_dims: Tuple[int, int, int] = (32, 48, 48),
            axes_lens: Tuple[int, int, int] = (1536, 512, 512),
            theta: float = 256.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.theta = theta

        for dim, max_len, m in zip(axes_dims, axes_lens, 'thw'):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(max_len, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            self.register_buffer(f'cos_{m}', freqs.cos())
            self.register_buffer(f'sin_{m}', freqs.sin())

    def forward(self, grid_size: Tuple[int, int, int], start_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        F, H, W = grid_size

        cos_t = self.cos_t[start_idx: start_idx + F]
        sin_t = self.sin_t[start_idx: start_idx + F]
        cos_h = self.cos_h[:H]
        sin_h = self.sin_h[:H]
        cos_w = self.cos_w[:W]
        sin_w = self.sin_w[:W]

        cos_t = cos_t.view(F, 1, 1, -1).expand(F, H, W, -1)
        sin_t = sin_t.view(F, 1, 1, -1).expand(F, H, W, -1)
        cos_h = cos_h.view(1, H, 1, -1).expand(F, H, W, -1)
        sin_h = sin_h.view(1, H, 1, -1).expand(F, H, W, -1)
        cos_w = cos_w.view(1, 1, W, -1).expand(F, H, W, -1)
        sin_w = sin_w.view(1, 1, W, -1).expand(F, H, W, -1)

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)
        cos = cos.flatten(0, 2)
        sin = sin.flatten(0, 2)

        return cos.float(), sin.float()


class ZImageTRTModel(ZImageTransformer2DModel):
    dtype_mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.FP8: torch.float8_e4m3fn,
    }
    dtype_mapping_reverse = {
        torch.float32: trt.DataType.FLOAT,
        torch.float16: trt.DataType.HALF,
        torch.bfloat16: trt.DataType.BF16,
        torch.int8: trt.DataType.INT8,
        torch.int32: trt.DataType.INT32,
        torch.int64: trt.DataType.INT64,
        torch.bool: trt.DataType.BOOL,
        torch.uint8: trt.DataType.UINT8,
        torch.float8_e4m3fn: trt.DataType.FP8,
    }

    def __init__(
            self,
            engine_file: str,
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16,
            plugin_file: Optional[str] = None,
    ):
        super(ZImageTransformer2DModel, self).__init__()
        self.plugin_handle = None if plugin_file is None else ctypes.cdll.LoadLibrary(plugin_file)
        torch.cuda.set_device(device)
        trt_logger: trt.Logger = trt.Logger(trt.Logger.VERBOSE)
        assert trt.init_libnvinfer_plugins(trt_logger, '')

        runtime: trt.Runtime
        with open(engine_file, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.STATIC)
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = context
        self.pos_embed = ZImageEmbedRope(
            axes_dims=(32, 48, 48),
            axes_lens=(1536, 512, 512),
            theta=256.0,
        ).to(dtype=torch.float32, device=device)
        self._device = device
        self._dtype = dtype
        self.stream = torch.cuda.Stream(device)
        self.in_channels = self.out_channels = 16
        config = lambda: None
        config.in_channels = 16
        config.guidance_embeds = False
        self._config = config
        self.register_parameter(
            'fake_empty_tensor',
            nn.Parameter(
                torch.empty(0, dtype=dtype, device=device),
                requires_grad=False,
            )
        )

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @torch.inference_mode()
    def forward(
            self,
            x: List[torch.Tensor],
            t: torch.Tensor,
            cap_feats: List[torch.Tensor],
            patch_size: int = 2,
            f_patch_size: int = 1,
    ) -> Tuple[List[torch.Tensor], Dict]:
        assert len(x) == 1 and len(cap_feats) == 1, 'only one input tensor is supported!'

        x_size = [x_i.shape[1:] for x_i in x]
        x = torch.stack(x, dim=0)
        cap_feats = torch.stack(cap_feats, dim=0)
        assert x.device == cap_feats.device == t.device == self.device

        cap_len = cap_feats.shape[1]
        pad_cap_len = ((cap_len + 127) // 128) * 128

        x = x.unflatten(2, (-1, f_patch_size)).unflatten(4, (-1, patch_size)).unflatten(6, (-1, patch_size))
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.flatten(4)
        x_grid_size = x.shape[1:4]
        x = x.flatten(1, 3)

        x_freqs_cis = self.pos_embed(grid_size=x_grid_size, start_idx=pad_cap_len + 1)
        cap_freqs_cis = self.pos_embed(grid_size=(cap_len, 1, 1), start_idx=1)

        hidden_states = x.contiguous()
        encoder_hidden_states = cap_feats.contiguous()
        timestep = t.contiguous()

        hidden_states = hidden_states.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('hidden_states')],
        )
        timestep = timestep.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('timestep')],
        )
        encoder_hidden_states = encoder_hidden_states.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('encoder_hidden_states')],
        )
        img_rope_real = x_freqs_cis[0].contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('img_rope_real')],
            device=self.device,
        )
        img_rope_imag = x_freqs_cis[1].contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('img_rope_imag')],
            device=self.device,
        )
        txt_rope_real = cap_freqs_cis[0].contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('txt_rope_real')],
            device=self.device,
        )
        txt_rope_imag = cap_freqs_cis[1].contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('txt_rope_imag')],
            device=self.device,
        )

        assert self.context.set_tensor_address('hidden_states', hidden_states.data_ptr())
        assert self.context.set_input_shape('hidden_states', tuple(hidden_states.shape))
        assert self.context.set_tensor_address('timestep', timestep.data_ptr())
        assert self.context.set_input_shape('timestep', tuple(timestep.shape))
        assert self.context.set_tensor_address('encoder_hidden_states', encoder_hidden_states.data_ptr())
        assert self.context.set_input_shape('encoder_hidden_states', tuple(encoder_hidden_states.shape))
        assert self.context.set_tensor_address('img_rope_real', img_rope_real.data_ptr())
        assert self.context.set_input_shape('img_rope_real', tuple(img_rope_real.shape))
        assert self.context.set_tensor_address('img_rope_imag', img_rope_imag.data_ptr())
        assert self.context.set_input_shape('img_rope_imag', tuple(img_rope_imag.shape))
        assert self.context.set_tensor_address('txt_rope_real', txt_rope_real.data_ptr())
        assert self.context.set_input_shape('txt_rope_real', tuple(txt_rope_real.shape))
        assert self.context.set_tensor_address('txt_rope_imag', txt_rope_imag.data_ptr())
        assert self.context.set_input_shape('txt_rope_imag', tuple(txt_rope_imag.shape))
        assert self.context.all_shape_inputs_specified and self.context.all_binding_shapes_specified

        out_hidden_states = torch.empty(
            tuple(self.context.get_tensor_shape('out_hidden_states')),
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('out_hidden_states')],
            device=self.device,
        )
        assert self.context.set_tensor_address('out_hidden_states', out_hidden_states.data_ptr())
        assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        unified = out_hidden_states.unbind(dim=0)
        out_hidden_states = self.unpatchify(list(unified), x_size, patch_size, f_patch_size)

        return out_hidden_states, {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen-Image model path or hf model id")
    parser.add_argument("--trt_path", type=str, required=True,
                        help="trt engine path of Qwen-Image\'s dit")
    parser.add_argument("--plugin_path", type=str, default=None,
                        help="trt engine path of Qwen-Image\'s dit")
    return parser.parse_args()


@torch.inference_mode()
def main(args: argparse.Namespace):
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    transformer: ZImageTRTModel = ZImageTRTModel(
        args.trt_path,
        device=device,
        dtype=dtype,
        plugin_file=args.plugin_path,
    )

    pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        transformer=None,
    )
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer = transformer

    pipe.to(device=device)
    pipe.set_progress_bar_config(disable=False)

    prompt = 'Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.'

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator('cuda').manual_seed(42),
    ).images[0]

    image.save("example.png")


if __name__ == '__main__':
    main(parse_args())
