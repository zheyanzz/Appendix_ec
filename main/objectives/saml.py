

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor


class SpatialAdaptiveMotionLoss(nn.Module):


    def __init__(self):

        super().__init__()

        self.raw_omega_hi = nn.Parameter(torch.tensor(1.0))

        self.raw_omega_edge = nn.Parameter(torch.tensor(1.0))

        self.raw_omega_lo = nn.Parameter(torch.tensor(0.5))


    def forward(

        self,

        pred_noise: Tensor,

        flow_inputs: dict,

        T_prime: int,

        P_grid_h: int,

        P_grid_w: int,

    ) -> Tensor:

        omega_hi = F.softplus(self.raw_omega_hi)

        omega_edge = F.softplus(self.raw_omega_edge)

        omega_lo = F.softplus(self.raw_omega_lo)


        B = pred_noise.shape[0]

        W_all = flow_inputs["W"]

        if W_all.dim() == 3:

            W_all = W_all.unsqueeze(0).expand(B, -1, -1, -1)


        f_bar_all = flow_inputs["f_bar"]

        if f_bar_all.dim() == 4:

            f_bar_all = f_bar_all.unsqueeze(0).expand(B, -1, -1, -1, -1)


        losses = []


        for t_prime_idx in range(1, T_prime):

            eps_t = pred_noise[:, t_prime_idx]      

            eps_tm1 = pred_noise[:, t_prime_idx - 1]


            if flow_inputs["has_displacement"]:

                grid = self._make_warp_grid(

                    f_bar_all[:, t_prime_idx], eps_t.shape, P_grid_h, P_grid_w

                )

                eps_warped = F.grid_sample(

                    eps_tm1, grid, mode="bilinear",

                    align_corners=True, padding_mode="border",

                )

            else:

                eps_warped = eps_tm1  


            D = (eps_t - eps_warped).pow(2).sum(dim=1)  


            W_t = W_all[:, t_prime_idx].reshape(B, P_grid_h, P_grid_w, 3)

            R = (

                omega_hi * W_t[..., 0]

                + omega_edge * W_t[..., 1]

                + omega_lo * W_t[..., 2]

            )


            loss_t = (R * D).mean(dim=(-2, -1))

            losses.append(loss_t)


        if not losses:

            return torch.tensor(0.0, device=pred_noise.device)


        return torch.stack(losses).mean()


    @staticmethod

    def _make_warp_grid(

        f_bar_t: Tensor, target_shape: tuple, P_h: int, P_w: int

    ) -> Tensor:

        if f_bar_t.dim() == 3:

            f_bar_t = f_bar_t.unsqueeze(0)


        B = target_shape[0]

        device = f_bar_t.device


        f_norm = torch.stack(

            [

                f_bar_t[:, 0] * 2.0 / max(P_w - 1, 1),

                f_bar_t[:, 1] * 2.0 / max(P_h - 1, 1),

            ],

            dim=-1,

        )  


        base = F.affine_grid(

            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),

            [B, 1, P_h, P_w],

            align_corners=True,

        )


        return base + f_norm

