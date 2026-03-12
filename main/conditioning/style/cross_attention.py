

import torch

import torch.nn as nn

from torch import Tensor


from ...infrastructure.attention import get_attention_backend


class StyleCrossAttention(nn.Module):


    def __init__(

        self,

        D_sty: int,

        D_h: int,

        D_tok: int,

        N_src: int,

        attention_backend: str = "auto",

    ):

        super().__init__()

        self.D_h = D_h

        self.D_tok = D_tok

        self.N_src = N_src  

        self.attention_backend_name = attention_backend

        self.attn_backend = get_attention_backend(attention_backend)


        self.W_Q = nn.Linear(D_tok, D_h, bias=False)

        self.W_K = nn.Linear(D_sty, D_h, bias=False)

        self.W_V = nn.Linear(D_sty, D_h, bias=False)

        self.W_O = nn.Linear(D_h, D_tok, bias=False)


        

        self.T_prime: int = 1


    def forward(

        self, hidden_states: Tensor, S_tokens: Tensor, t_prime: int

    ) -> Tensor:

        B = hidden_states.shape[0]


        if S_tokens.dim() == 4:

            s = S_tokens[:, t_prime]  

        elif S_tokens.dim() == 3:

            s_idx = t_prime if S_tokens.shape[0] > 1 else 0

            s = S_tokens[s_idx].unsqueeze(0).expand(B, -1, -1)  

        else:

            s = S_tokens.unsqueeze(0).expand(B, -1, -1)


        Q_h = self.W_Q(hidden_states)  

        Ks = self.W_K(s)               

        Vs = self.W_V(s)               


        attn = self.attn_backend(Q_h, Ks, Vs)


        return self.W_O(attn)


    def apply_to_output(

        self, output: Tensor, S_tokens: Tensor, layer_idx: int

    ) -> Tensor:

        T_prime = self.T_prime

        N_s = self.N_src


        output = output.clone()


        for t in range(T_prime):

            c_start = t * N_s

            c_end = (t + 1) * N_s


            content_tokens = output[:, c_start:c_end, :]

            style_out = self.forward(content_tokens, S_tokens, t_prime=t)


            output[:, c_start:c_end, :] = output[:, c_start:c_end, :] + style_out


        return output

