from __future__ import annotations




import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor


class StyleTemporalEncoder(nn.Module):


    def __init__(self, D_clip: int = 512, D_sty: int = 1024, P_sty: int = 8):

        super().__init__()

        self.D_sty = D_sty

        self.P_sty = P_sty


        self.conv1 = nn.Conv3d(D_clip, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.norm1 = nn.GroupNorm(8, 512)

        self.conv2 = nn.Conv3d(512, D_sty, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.norm2 = nn.GroupNorm(8, D_sty)

        self.act = nn.GELU()


    def forward(

        self,

        Y_tilde: Tensor,

        T_prime: int,

        cut_boundaries: list[int] | None = None,

    ) -> Tensor:

        x = Y_tilde.permute(1, 0, 2, 3).unsqueeze(0)


        if cut_boundaries:

            for t_star in cut_boundaries:

                idx = t_star - 1

                if 0 <= idx < x.shape[2]:

                    x[0, :, idx, :, :] = 0.0


        x = self.act(self.norm1(self.conv1(x)))

        x = self.act(self.norm2(self.conv2(x)))

        x = F.adaptive_avg_pool3d(x, (T_prime, self.P_sty, self.P_sty))


        S = x.squeeze(0).permute(1, 0, 2, 3)

        return S

