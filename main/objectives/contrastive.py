

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor


class ContrastiveContentLoss(nn.Module):


    def __init__(self, clip_model=None, tau: float = 0.07):

        super().__init__()

        self.clip = clip_model

        self.tau = tau


    def set_clip(self, clip_model):

        self.clip = clip_model


    def _clip_cls(self, frames: Tensor, with_grad: bool) -> Tensor:

        x = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)

        if with_grad:

            features = self.clip.encode_image(x)

        else:

            with torch.no_grad():

                features = self.clip.encode_image(x)

        return F.normalize(features, dim=-1)


    def forward(

        self,

        pred_frames: Tensor,

        src_frames: Tensor,

        sty_frames: Tensor,

    ) -> Tensor:

        if self.clip is None:

            raise RuntimeError("CLIP model is not set. Call set_clip() first.")


        phi_pred = self._clip_cls(pred_frames, with_grad=True)

        phi_src = self._clip_cls(src_frames, with_grad=False)

        phi_sty = self._clip_cls(sty_frames, with_grad=False)


        sim_pos = (phi_pred * phi_src).sum(-1) / self.tau

        sim_neg = (phi_pred * phi_sty).sum(-1) / self.tau

        logits = torch.stack([sim_pos, sim_neg], dim=-1)

        target = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)

        loss = F.cross_entropy(logits, target)


        return loss

