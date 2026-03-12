from __future__ import annotations
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

class TransitionDetector:

    def __init__(
        self,
        clip_model=None,
        scales: list[int] | None = None,
        alpha: float = 1.0,
        tau: int = 2,
        W_cls: int = 5,
    ):
        self.clip = clip_model
        self.scales = scales or [64, 128, 224]
        self.alpha = alpha
        self.tau = tau
        self.W_cls = W_cls

    def set_clip(self, clip_model):
        self.clip = clip_model

    @torch.no_grad()
    def detect(self, style_frames: Tensor) -> tuple[list[set[int]], dict]:

        N = style_frames.shape[0]
        device = style_frames.device

        S = {}  
        for s in self.scales:
            energies = []
            for t in range(N):
                frame = style_frames[t:t+1]  
                I_s = F.interpolate(frame, size=(s, s), mode="bilinear", align_corners=False)
                F_t = self._clip_features(I_s)  
                C_feat, L = F_t.shape
                G = (1.0 / (C_feat * L)) * (F_t @ F_t.T)  
                energy = torch.norm(G, p="fro").item()
                energies.append(energy)
            S[s] = energies
        delta_S = {}
        B_per_scale = {}
        for s in self.scales:
            diffs = [abs(S[s][t] - S[s][t-1]) for t in range(1, N)]
            delta_S[s] = diffs
            if len(diffs) == 0:
                B_per_scale[s] = []
                continue
            mu = sum(diffs) / len(diffs)
            var = sum((d - mu) ** 2 for d in diffs) / len(diffs)
            sigma = var ** 0.5
            theta = mu + self.alpha * sigma
            B_per_scale[s] = [1 if d > theta else 0 for d in diffs]


        B_final = []

        for t_idx in range(N - 1):
            votes = sum(B_per_scale[s][t_idx] for s in self.scales if t_idx < len(B_per_scale[s]))
            B_final.append(1 if votes >= self.tau else 0)

        clusters = []
        visited = [False] * len(B_final)
        for t_idx in range(len(B_final)):
            if B_final[t_idx] == 1 and not visited[t_idx]:

                frame_num = t_idx + 2  
                cluster = {frame_num}
                visited[t_idx] = True
                for j in range(t_idx + 1, min(t_idx + self.W_cls, len(B_final))):
                    if B_final[j] == 1 and not visited[j]:
                        cluster.add(j + 2)
                        visited[j] = True
                clusters.append(cluster)

        return clusters, delta_S

    def _clip_features(self, image: Tensor) -> Tensor:
        if self.clip is None:
            raise RuntimeError("CLIP model not set. Call set_clip() first.")

        visual = self.clip.visual
        x = visual.conv1(image)  
        D, gh, gw = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(1, D, -1).permute(0, 2, 1)  
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0)
        x = torch.cat([cls_token + visual.positional_embedding[:1],

                        x + visual.positional_embedding[1:1+x.shape[1]]], dim=1)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  

        patch_tokens = x[0, 1:]  
        return patch_tokens.T  

