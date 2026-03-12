

import torch

import torch.nn as nn

from torch import Tensor


class CannyEncoder3D(nn.Module):


    def __init__(self, C_lat: int = 16):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),

            nn.SiLU(),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.SiLU(),

            nn.Conv3d(64, C_lat, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),

        )

        

        


    def forward(self, canny_frames: Tensor) -> Tensor:

        x = canny_frames.permute(0, 2, 1, 3, 4)  

        return self.net(x)  

