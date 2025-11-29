from typing import Tuple, Optional
import os
import tempfile
import argparse

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusers
import diffusers.models.transformers.transformer_z_image
from diffusers import ZImageTransformer2DModel
from diffusers.models.attention_processor import Attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Tongyi-MAI/Z-Image-Turbo',
                        help="Z-Image-Turbo model path or hf model id")
    parser.add_argument("--onnx_path", type=str, default='transformer.onnx',
                        help="ONNX path of exported Z-Image-Turbo\'s dit")
    return parser.parse_args()


def apply_rotary_emb_zimage(x_in: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x_fp32 = x_in.float()  # improve precision
    x_fp32 = x_fp32.unflatten(-1, (-1, 2))
    a, b = x_fp32.unbind(-1)  # [b, s, n, d//2]
    c, d = freqs_cis  # [s, d//2]
    c = c[None, :, None, :]  # [1, s, 1, d//2]
    d = d[None, :, None, :]  # [1, s, 1, d//2]
    real = (a * c - b * d).to(x_in.dtype)
    imag = (b * c + a * d).to(x_in.dtype)
    y = torch.stack([real, imag], dim=-1)
    return y.flatten(-2)


def _scaled_dot_product_cudnn_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
) -> torch.Tensor:
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    output = F.scaled_dot_product_attention(query, key, value)
    return output.permute(0, 2, 1, 3)


class TRTCUDNNAttention(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: torch.Graph,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
    ) -> torch.Tensor:
        return _scaled_dot_product_cudnn_attention(query, key, value)

    @staticmethod
    def symbolic(
            g,
            query: torch.Value,
            key: torch.Value,
            value: torch.Value,
    ) -> torch.Value:
        out = g.op(
            'TRT::CUDNNAttention',
            query,
            key,
            value,
            outputs=1,
        )
        return out


scaled_dot_product_cudnn_attention = TRTCUDNNAttention.apply


class ZImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # [b, seq_len, 3 * h]
        qkv = attn.to_qkv(hidden_states)
        # [b, seq_len, 3, n, d]
        qkv = qkv.unflatten(-1, (3, attn.heads, -1))
        # [b, seq_len, n, d]
        query, key, value = qkv.unbind(2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        dtype = query.dtype

        # Apply RoPE
        query = apply_rotary_emb_zimage(query, freqs_cis)
        key = apply_rotary_emb_zimage(key, freqs_cis)

        # Cast to correct dtype
        query, key = query.to(dtype), key.to(dtype)

        # Compute joint attention
        hidden_states = scaled_dot_product_cudnn_attention(
            query,
            key,
            value,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class OptDit(nn.Module):
    def __init__(self, dit: ZImageTransformer2DModel):
        super().__init__()
        for m in dit.modules():
            if isinstance(m, Attention):
                m.set_processor(ZImageAttnProcessor())
                m.fuse_projections()
        self.dit = dit

    def forward(
            self,
            hidden_states: torch.Tensor,  # bf16 [batch, img_seq_len, 64]
            encoder_hidden_states: torch.Tensor,  # bf16 [batch, txt_seq_len, 2560]
            timestep: torch.Tensor,  # bf16 [batch]
            img_rope_real: torch.Tensor,  # fp32 [img_seq_len, 128//2]
            img_rope_imag: torch.Tensor,  # fp32 [img_seq_len, 128//2]
            txt_rope_real: torch.Tensor,  # fp32 [txt_seq_len, 128//2]
            txt_rope_imag: torch.Tensor,  # fp32 [txt_seq_len, 128//2]
    ) -> torch.Tensor:
        img_seq_len = hidden_states.size(1)

        hidden_states = self.dit.all_x_embedder['2-1'](hidden_states)
        encoder_hidden_states = self.dit.cap_embedder(encoder_hidden_states)

        timestep = timestep * self.dit.t_scale
        timestep = self.dit.t_embedder(timestep)
        adaln_input = timestep.type_as(hidden_states)

        # image
        for layer in self.dit.noise_refiner:
            hidden_states = layer(hidden_states, None, [img_rope_real, img_rope_imag], adaln_input)

        # text
        for layer in self.dit.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, None, [txt_rope_real, txt_rope_imag])

        # unified
        unified = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        unified_rope_real = torch.cat([img_rope_real, txt_rope_real], dim=0)
        unified_rope_imag = torch.cat([img_rope_imag, txt_rope_imag], dim=0)

        for layer in self.dit.layers:
            unified = layer(unified, None, [unified_rope_real, unified_rope_imag], adaln_input)

        unified = self.dit.all_final_layer['2-1'](unified, adaln_input)
        return unified[:, :img_seq_len]


@torch.inference_mode()
def main(args: argparse.Namespace):
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    transformer: ZImageTransformer2DModel = ZImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder='transformer',
        torch_dtype=dtype,
    )

    transformer.eval()
    transformer.to(dtype=dtype, device=device)

    img_width = 1024
    img_height = 1024

    batch_size = 1
    img_seq_len = img_width // 16 * img_height // 16
    txt_seq_len = 256
    in_channels = transformer.config.in_channels * 2 * 2  # 16 * 2 * 2 = 64
    txt_attention_dim = transformer.config.cap_feat_dim  # 2560
    attention_head_dim = transformer.config.dim // transformer.config.n_heads  # 128

    hidden_states = torch.randn(
        (batch_size, img_seq_len, in_channels),
        dtype=dtype,
        device=device,
    )
    timestep = torch.randint(
        0, 1000,
        (batch_size,),
        device=device,
    ).to(dtype=dtype)
    encoder_hidden_states = torch.randn(
        (batch_size, txt_seq_len, txt_attention_dim),
        dtype=dtype,
        device=device,
    )
    img_rope_real = torch.randn(
        (img_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    img_rope_imag = torch.randn(
        (img_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    txt_rope_real = torch.randn(
        (txt_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    txt_rope_imag = torch.randn(
        (txt_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )

    transformer: OptDit = OptDit(transformer)

    out_hidden_states = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_rope_real=img_rope_real,
        img_rope_imag=img_rope_imag,
        txt_rope_real=txt_rope_real,
        txt_rope_imag=txt_rope_imag,
    )

    print(f'{out_hidden_states.shape}\n', end='')

    with tempfile.TemporaryDirectory() as d:
        temp_path = f'{d}/{os.path.basename(args.onnx_path)}'
        torch.onnx.export(
            transformer,
            (
                hidden_states,  # hidden_states
                encoder_hidden_states,  # encoder_hidden_states
                timestep,  # timestep
                img_rope_real,  # img_rope_real
                img_rope_imag,  # img_rope_imag
                txt_rope_real,  # txt_rope_real
                txt_rope_imag,  # txt_rope_imag
            ),
            temp_path,
            opset_version=17,
            input_names=[
                'hidden_states',
                'encoder_hidden_states',
                'timestep',
                'img_rope_real',
                'img_rope_imag',
                'txt_rope_real',
                'txt_rope_imag',
            ],
            output_names=['out_hidden_states'],
            dynamic_axes={
                'hidden_states': {1: 'img_seq_len'},
                'encoder_hidden_states': {1: 'txt_seq_len'},
                'img_rope_real': {0: 'img_seq_len'},
                'img_rope_imag': {0: 'img_seq_len'},
                'txt_rope_real': {0: 'txt_seq_len'},
                'txt_rope_imag': {0: 'txt_seq_len'},
                'out_hidden_states': {1: 'img_seq_len'},
            }
        )
        onnx_model = onnx.load(temp_path)
        onnx.save(
            onnx_model,
            args.onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=args.onnx_path.replace('.onnx', '.onnx.data'),
        )


if __name__ == '__main__':
    main(parse_args())
