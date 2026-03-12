from __future__ import annotations




import torch

import torch.nn as nn

from torch import Tensor


class StyleAppearanceLoss(nn.Module):


    def __init__(self, clip_model=None, layers: list[int] | None = None):

        super().__init__()

        self.clip = clip_model

        self.layers = layers or [4, 8, 12, 16]

        self._features: dict[int, Tensor] = {}

        self._hooks = []


    def set_clip(self, clip_model):

        self.clip = clip_model

        self._register_hooks()


    def _register_hooks(self):

        for h in self._hooks:

            h.remove()

        self._hooks = []


        if self.clip is None:

            return


        transformer = self.clip.visual.transformer

        for layer_idx in self.layers:

            if layer_idx < len(transformer.resblocks):

                block = transformer.resblocks[layer_idx]

                h = block.register_forward_hook(

                    lambda mod, inp, out, idx=layer_idx: self._save_features(idx, out)

                )

                self._hooks.append(h)


    def _save_features(self, layer_idx: int, output: Tensor):

        self._features[layer_idx] = output


    def _extract_features(self, frames: Tensor, with_grad: bool) -> dict[int, Tensor]:

        import torch.nn.functional as F


        self._features = {}

        x = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)


        if with_grad:

            self.clip.encode_image(x)

        else:

            with torch.no_grad():

                self.clip.encode_image(x)


        result = {}

        for layer_idx, feat in self._features.items():

            

            if feat.dim() == 3:

                if feat.shape[0] != frames.shape[0]:

                    feat = feat.permute(1, 0, 2)  

                

                feat = feat[:, 1:]  

                result[layer_idx] = feat.permute(0, 2, 1)  


        return result


    def forward(

        self,

        pred_frames: Tensor,

        style_frames: Tensor,

    ) -> Tensor:

        if self.clip is None:

            raise RuntimeError("CLIP model is not set. Call set_clip() first.")


        feat_pred = self._extract_features(pred_frames, with_grad=True)

        feat_style = self._extract_features(style_frames, with_grad=False)


        loss = torch.tensor(0.0, device=pred_frames.device)


        for l_idx in self.layers:

            if l_idx not in feat_pred or l_idx not in feat_style:

                continue


            F_pred = feat_pred[l_idx]    

            F_style = feat_style[l_idx]  


            C_l, L_l = F_pred.shape[1], F_pred.shape[2]


            G_pred = (1.0 / (C_l * L_l)) * torch.bmm(F_pred, F_pred.transpose(1, 2))

            G_style = (1.0 / (C_l * L_l)) * torch.bmm(F_style, F_style.transpose(1, 2))


            loss = loss + (G_pred - G_style).pow(2).sum(dim=(-2, -1)).mean()


        return loss

