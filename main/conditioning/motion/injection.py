

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class MotionInjectionModule(nn.Module):

    def __init__(self, n_motion_heads: int, D_h: int, N_src: int):
        super().__init__()
        self.n_motion_heads = n_motion_heads
        self.D_h = D_h
        self.N_src = N_src


        self.motion_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * D_h),
            )
            for _ in range(n_motion_heads)
        ])

        self.region_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3))
            for _ in range(n_motion_heads)
        ])

    @staticmethod

    def _project_feature_orthogonal(content: Tensor, residual: Tensor) -> Tensor:
        d_feat = content.shape[1]
        if d_feat <= 1:
            return residual


        q, _ = torch.linalg.qr(content.transpose(0, 1), mode="reduced")


        if q.shape[1] >= d_feat:
            q = q[:, : d_feat - 1]
        if q.numel() == 0:
            return residual

        return residual - (residual @ q) @ q.transpose(0, 1)


    @staticmethod

    def _project_token_orthogonal(content: Tensor, residual: Tensor) -> Tensor:
        q, _ = torch.linalg.qr(content, mode="reduced")
        return residual - q @ (q.transpose(0, 1) @ residual)


    def inject_head(

        self,
        h_idx: int,
        K_h: Tensor,
        V_h: Tensor,
        t_prime: int,
        M: Tensor,
        W: Tensor,

    ) -> tuple[Tensor, Tensor]:
        N_src = self.N_src
        if K_h.shape[0] < N_src or V_h.shape[0] < N_src:
            raise ValueError(
                f"Expected at least {N_src} tokens, got K={K_h.shape[0]}, V={V_h.shape[0]}"

            )

        M_t = M[t_prime]  
        W_t = W[t_prime]  


        raw = self.motion_mlp[h_idx](M_t)  
        dK, dV = raw.chunk(2, dim=-1)      


        

        mix = torch.softmax(self.region_weights[h_idx], dim=0)
        w_mix = W_t @ mix
        dK_mod = w_mix.unsqueeze(-1) * dK
        dV_mod = w_mix.unsqueeze(-1) * dV


        

        K_c = K_h[:N_src]
        V_c = V_h[:N_src]
        dK_hat = self._project_feature_orthogonal(K_c, dK_mod)

        dV_hat = self._project_feature_orthogonal(V_c, dV_mod)
        dK_hat = self._project_token_orthogonal(K_c, dK_hat)
        dV_hat = self._project_token_orthogonal(V_c, dV_hat)


        

        K_prime = K_h.clone()
        V_prime = V_h.clone()
        K_prime[:N_src] = K_prime[:N_src] + dK_hat
        V_prime[:N_src] = V_prime[:N_src] + dV_hat


        return K_prime, V_prime

