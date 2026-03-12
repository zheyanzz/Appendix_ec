

import torch

import torch.nn as nn

from torch import Tensor


from .injection import MotionInjectionModule

from ...infrastructure.attention import get_attention_backend


class MotionInjectionProcessor(nn.Module):


    def __init__(

        self,

        motion_module: MotionInjectionModule,

        flow_inputs: dict,

        T_prime: int,

        N_src_spatial: int,

        n_motion_heads: int,

        H_heads: int = 48,

        D_h: int = 64,

        attention_backend: str = "auto",

    ):

        super().__init__()

        self.motion_module = motion_module

        self.flow_inputs = flow_inputs

        self.T_prime = T_prime

        self.N_src_spatial = N_src_spatial  

        self.n_motion_heads = n_motion_heads

        self.H_heads = H_heads

        self.D_h = D_h

        self.attention_backend_name = attention_backend

        self.attn_backend = get_attention_backend(attention_backend)


    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):

        B, seq, D_tok = hidden_states.shape

        N_s = self.N_src_spatial

        min_seq = self.T_prime * N_s

        if seq < min_seq:

            raise ValueError(

                f"Sequence too short for T'={self.T_prime}, N_src={N_s}: got {seq}"

            )

        has_canny_tokens = seq >= 2 * min_seq


        

        Q = attn.to_q(hidden_states)

        K = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)

        V = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)


        Q = Q.reshape(B, seq, self.H_heads, self.D_h).permute(0, 2, 1, 3)

        K = K.reshape(B, -1, self.H_heads, self.D_h).permute(0, 2, 1, 3)

        V = V.reshape(B, -1, self.H_heads, self.D_h).permute(0, 2, 1, 3)


        

        motion_batch = self.flow_inputs["M"]

        region_batch = self.flow_inputs["W"]


        for h in range(self.n_motion_heads):

            for b in range(B):

                M_b = motion_batch[b] if motion_batch.dim() == 4 else motion_batch

                W_b = region_batch[b] if region_batch.dim() == 4 else region_batch

                for t in range(self.T_prime):

                    c_start = t * N_s

                    c_end = (t + 1) * N_s

                    if has_canny_tokens:

                        can_start = self.T_prime * N_s + t * N_s

                        can_end = self.T_prime * N_s + (t + 1) * N_s

                        K_frame = torch.cat(

                            [K[b, h, c_start:c_end], K[b, h, can_start:can_end]], dim=0

                        )

                        V_frame = torch.cat(

                            [V[b, h, c_start:c_end], V[b, h, can_start:can_end]], dim=0

                        )

                    else:

                        K_frame = K[b, h, c_start:c_end]

                        V_frame = V[b, h, c_start:c_end]


                    K_new, V_new = self.motion_module.inject_head(

                        h, K_frame, V_frame,

                        t,

                        M_b,

                        W_b,

                    )


                    K[b, h, c_start:c_end] = K_new[:N_s]

                    V[b, h, c_start:c_end] = V_new[:N_s]

                    if has_canny_tokens:

                        K[b, h, can_start:can_end] = K_new[N_s:]

                        V[b, h, can_start:can_end] = V_new[N_s:]


        

        out = self.attn_backend(Q, K, V)

        out = out.permute(0, 2, 1, 3).reshape(B, seq, D_tok)


        

        out = attn.to_out[0](out)

        if len(attn.to_out) > 1:

            out = attn.to_out[1](out)


        return out

