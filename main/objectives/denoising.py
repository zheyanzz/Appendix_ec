

import torch

import torch.nn.functional as F

from torch import Tensor


class DenoisingLoss:


    def __call__(self, pred_noise: Tensor, gt_noise: Tensor) -> Tensor:

        return F.mse_loss(pred_noise, gt_noise)

