from __future__ import annotations




import logging


import torch

import torch.nn as nn

from torch import Tensor


logger = logging.getLogger(__name__)


class PolicyClassifier(nn.Module):


    def __init__(

        self,

        delta_min: int = 10,

        W_feat: int = 5,

        hidden_dim: int = 32,

        scales: list[int] | None = None,

    ):

        super().__init__()

        self.delta_min = delta_min

        self.W_feat = W_feat

        self.scales = scales or [64, 128, 224]


        

        self.mlp = nn.Sequential(

            nn.Linear(4, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, 2),

        )


        

        self.boundary_mlp = nn.Sequential(

            nn.Linear(4, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, 1),

        )


        

        self.last_boundary_logits: Tensor | None = None

        self.last_policy_logits: Tensor | None = None


    def classify(

        self,

        clusters: list[set[int]],

        delta_S: dict,

        N: int | None = None,

    ) -> tuple[list[int], list[str], bool]:

        if not clusters:

            self.last_boundary_logits = None

            self.last_policy_logits = None

            return [], [], True


        

        if N is None:

            first_scale = self.scales[0]

            N = len(delta_S[first_scale]) + 1


        s_max = max(self.scales)


        transitions = []

        policies = []

        all_logits = []


        for cluster in clusters:

            

            t_k = max(

                cluster,

                key=lambda t: sum(

                    delta_S[s][t - 2] if 0 <= t - 2 < len(delta_S[s]) else 0.0

                    for s in self.scales

                ),

            )


            

            if transitions and min(abs(t_k - t_star) for t_star in transitions) < self.delta_min:

                continue


            

            w_start = max(0, t_k - 2 - self.W_feat)  

            w_end = min(len(delta_S[s_max]), t_k - 2 + self.W_feat + 1)

            v_fine = delta_S[s_max][w_start:w_end]


            if len(v_fine) == 0:

                continue


            v_tensor = torch.tensor(v_fine, dtype=torch.float32)


            

            theta_local = v_tensor.median()

            dur = float((v_tensor > theta_local).sum().item())

            peak = float(v_tensor.max().item())

            mean_val = float(v_tensor.mean().item())

            gamma = peak / max(mean_val, 1e-8)


            x_k = torch.tensor([dur, gamma, peak, mean_val], dtype=torch.float32)

            x_k = x_k.to(next(self.mlp.parameters()).device)


            logits = self.mlp(x_k.unsqueeze(0))  

            all_logits.append(logits)

            p_k = "cut" if logits[0, 0] > logits[0, 1] else "fade"


            transitions.append(t_k)

            policies.append(p_k)


        

        if transitions:

            sorted_pairs = sorted(zip(transitions, policies, all_logits), key=lambda x: x[0])

            transitions = [p[0] for p in sorted_pairs]

            policies = [p[1] for p in sorted_pairs]

            all_logits = [p[2] for p in sorted_pairs]


        

        if all_logits:

            self.last_policy_logits = torch.cat(all_logits, dim=0)  

        else:

            self.last_policy_logits = torch.zeros(0, 2, device=next(self.mlp.parameters()).device)


        

        device = next(self.mlp.parameters()).device

        boundary_logits = torch.zeros(N, device=device)

        for t_idx in range(1, N):  

            

            w_start = max(0, t_idx - 1 - self.W_feat)

            w_end = min(len(delta_S[s_max]), t_idx - 1 + self.W_feat + 1)

            v_local = delta_S[s_max][w_start:w_end]

            if len(v_local) == 0:

                continue

            v_t = torch.tensor(v_local, dtype=torch.float32)

            theta_local = v_t.median()

            dur = float((v_t > theta_local).sum().item())

            peak = float(v_t.max().item())

            mean_val = float(v_t.mean().item())

            gamma = peak / max(mean_val, 1e-8)

            x_t = torch.tensor([dur, gamma, peak, mean_val], dtype=torch.float32, device=device)

            boundary_logits[t_idx] = self.boundary_mlp(x_t.unsqueeze(0)).squeeze()

        self.last_boundary_logits = boundary_logits


        is_uniform = len(transitions) == 0

        return transitions, policies, is_uniform

