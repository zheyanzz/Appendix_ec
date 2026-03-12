

import logging

from typing import Any


import torch

import torch.nn as nn

from torch import Tensor


from ..conditioning.structure import CannyEncoder3D

from ..conditioning.motion import MotionInjectionProcessor, MotionInjectionModule


logger = logging.getLogger(__name__)


class DiTBackbone(nn.Module):


    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg


        from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX


        model_id = "THUDM/CogVideoX-5b-I2V"


        self.transformer = CogVideoXTransformer3DModel.from_pretrained(

            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,

        )

        self.vae = AutoencoderKLCogVideoX.from_pretrained(

            model_id, subfolder="vae", torch_dtype=torch.bfloat16,

        )

        self.vae.requires_grad_(False)  


        C_lat = self.vae.config.latent_channels

        self.canny_enc = CannyEncoder3D(C_lat=C_lat)


        

        self.transformer.enable_gradient_checkpointing()


        self._original_processors: dict[int, Any] = {}


        

        self.D_tok = self.transformer.config.inner_dim  

        self.H_heads = self.transformer.config.num_attention_heads  

        self.D_h = self.D_tok // self.H_heads  


    @torch.no_grad()

    def encode_video(self, frames: Tensor) -> Tensor:

        x = frames.permute(0, 2, 1, 3, 4)  

        z = self.vae.encode(x).latent_dist.sample()

        return z * self.vae.config.scaling_factor


    def decode_latents(self, z: Tensor) -> Tensor:

        z = z / self.vae.config.scaling_factor

        out = self.vae.decode(z).sample

        return out.permute(0, 2, 1, 3, 4)


    def set_motion_processors(

        self,

        motion_module: MotionInjectionModule,

        flow_inputs: dict,

        T_prime: int,

        N_src_spatial: int,

        attention_backend: str = "auto",

    ):

        self._original_processors = {}

        motion_cfg = getattr(self.cfg, "motion_injection", None) or self.cfg.model.motion_injection

        motion_layers = motion_cfg.motion_layers


        n_motion_heads = motion_cfg.n_motion_heads

        if n_motion_heads == -1:

            n_motion_heads = self.H_heads


        for layer_idx in motion_layers:

            block = self.transformer.transformer_blocks[layer_idx]

            self._original_processors[layer_idx] = block.attn1.processor

            block.attn1.set_processor(

                MotionInjectionProcessor(

                    motion_module,

                    flow_inputs,

                    T_prime=T_prime,

                    N_src_spatial=N_src_spatial,

                    n_motion_heads=n_motion_heads,

                    H_heads=self.H_heads,

                    D_h=self.D_h,

                    attention_backend=attention_backend,

                )

            )


    def restore_processors(self):

        for layer_idx, proc in self._original_processors.items():

            self.transformer.transformer_blocks[layer_idx].attn1.set_processor(proc)

        self._original_processors = {}


    def forward_with_hooks(

        self,

        Z_full: Tensor,

        style_attn_module,

        S_tokens: Tensor,

        flow_inputs: dict,

        timestep: Tensor,

        motion_module: MotionInjectionModule,

        T_prime: int = 1,

        N_src_spatial: int = 5400,

        attention_backend: str = "auto",

    ) -> Tensor:

        style_attn_module.T_prime = T_prime

        style_attn_module.N_src = N_src_spatial


        

        self.set_motion_processors(

            motion_module, flow_inputs,

            T_prime=T_prime, N_src_spatial=N_src_spatial,

            attention_backend=attention_backend,

        )


        

        handles = []

        style_cfg = getattr(self.cfg, "style_attention", None) or self.cfg.model.style_attention


        def _style_hook(_, __, out, idx: int):

            if isinstance(out, tuple):

                if not out:

                    return out

                hidden = style_attn_module.apply_to_output(out[0], S_tokens, idx)

                return (hidden, *out[1:])

            return style_attn_module.apply_to_output(out, S_tokens, idx)


        for layer_idx in style_cfg.style_layers:

            block = self.transformer.transformer_blocks[layer_idx]

            h = block.register_forward_hook(

                lambda mod, inp, out, idx=layer_idx: _style_hook(mod, inp, out, idx)

            )

            handles.append(h)


        

        pred_noise = self.transformer(Z_full, timestep=timestep).sample


        

        self.restore_processors()

        for h in handles:

            h.remove()


        return pred_noise


DiTFoundationBackbone = DiTBackbone


__all__ = ["DiTBackbone", "DiTFoundationBackbone"]

