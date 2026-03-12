from __future__ import annotations




import torch

import torch.nn.functional as F

from torch import Tensor


class TSTMLoss:


    def __call__(

        self,

        boundary_logits: Tensor | None,

        gt_boundary_mask: Tensor | None,

        policy_logits: Tensor | None,

        gt_policies: Tensor | None,

    ) -> Tensor:

        device = "cpu"


        

        L_det = torch.tensor(0.0)

        if boundary_logits is not None and gt_boundary_mask is not None:

            device = boundary_logits.device

            gt_float = gt_boundary_mask.float()

            n_pos = gt_float.sum().clamp(min=1.0)

            n_neg = (gt_float.numel() - n_pos).clamp(min=1.0)

            pos_weight = (n_neg / n_pos).to(device)

            L_det = F.binary_cross_entropy_with_logits(

                boundary_logits, gt_float,

                pos_weight=pos_weight,

            )


        

        L_cls = torch.tensor(0.0, device=device)

        if policy_logits is not None and policy_logits.shape[0] > 0 and gt_policies is not None:

            L_cls = F.cross_entropy(policy_logits, gt_policies)


        return L_det + L_cls

