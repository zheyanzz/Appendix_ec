from __future__ import annotations




import math


import torch

from torch import Tensor


class TSTM:


    def __init__(

        self,

        delta: int = 3,

        sigma: float = 1.0,

        lambda_decay: float = 1.5,

        w_fade: int = 5,

    ):

        self.delta = delta

        self.sigma = sigma

        self.lambda_decay = lambda_decay

        self.w_fade = w_fade


    def __call__(

        self,

        Y: Tensor,

        transitions: list[int],

        policies: list[str],

    ) -> Tensor:

        N = Y.shape[0]

        device = Y.device


        if not transitions:

            

            return self._smooth_all_norm(Y)


        

        boundaries = [1] + transitions + [N + 1]

        segments = []

        for i in range(len(boundaries) - 1):

            segments.append((boundaries[i], boundaries[i + 1]))


        

        frame_policies = ["norm"] * N

        for j, (t_star, policy) in enumerate(zip(transitions, policies)):

            if policy == "cut":

                for t in range(max(1, t_star - self.delta), min(N + 1, t_star + self.delta + 1)):

                    frame_policies[t - 1] = "cut"

            elif policy == "fade":

                for t in range(max(1, t_star - self.w_fade), min(N + 1, t_star + self.w_fade + 1)):

                    frame_policies[t - 1] = "fade"


        Y_tilde = torch.zeros_like(Y)


        for t in range(1, N + 1):  

            idx = t - 1  

            policy = frame_policies[idx]


            if policy == "norm":

                Y_tilde[idx] = self._smooth_norm(Y, t, N, segments)

            elif policy == "cut":

                Y_tilde[idx] = self._smooth_cut(Y, t, N, transitions, policies)

            elif policy == "fade":

                Y_tilde[idx] = self._smooth_fade(Y, t, N, transitions, policies, segments)


        return Y_tilde


    def _smooth_all_norm(self, Y: Tensor) -> Tensor:

        N = Y.shape[0]

        Y_tilde = torch.zeros_like(Y)

        for t in range(1, N + 1):

            weights = []

            indices = []

            for tau in range(-self.delta, self.delta + 1):

                t_tau = t + tau

                if 1 <= t_tau <= N:

                    w = math.exp(-tau ** 2 / (2 * self.sigma ** 2))

                    weights.append(w)

                    indices.append(t_tau - 1)

            w_sum = sum(weights)

            weights = [w / w_sum for w in weights]

            for w, i in zip(weights, indices):

                Y_tilde[t - 1] += w * Y[i]

        return Y_tilde


    def _get_segment(self, t: int, segments: list[tuple[int, int]]) -> int:

        for i, (start, end) in enumerate(segments):

            if start <= t < end:

                return i

        return len(segments) - 1


    def _smooth_norm(self, Y: Tensor, t: int, N: int, segments) -> Tensor:

        seg_idx = self._get_segment(t, segments)

        seg_start, seg_end = segments[seg_idx]


        weights = []

        indices = []

        for tau in range(-self.delta, self.delta + 1):

            t_tau = t + tau

            if 1 <= t_tau <= N and seg_start <= t_tau < seg_end:

                w = math.exp(-tau ** 2 / (2 * self.sigma ** 2))

                weights.append(w)

                indices.append(t_tau - 1)


        if not weights:

            return Y[t - 1]


        w_sum = sum(weights)

        result = torch.zeros_like(Y[0])

        for w, i in zip(weights, indices):

            result += (w / w_sum) * Y[i]

        return result


    def _smooth_cut(self, Y: Tensor, t: int, N: int,

                    transitions: list[int], policies: list[str]) -> Tensor:

        

        nearest_cut = None

        for t_star, pol in zip(transitions, policies):

            if pol == "cut" and abs(t - t_star) <= self.delta:

                nearest_cut = t_star

                break


        if nearest_cut is None:

            return Y[t - 1]


        if t < nearest_cut:

            

            weights = []

            indices = []

            for tau in range(-self.delta, 1):

                t_tau = t + tau

                if 1 <= t_tau <= N:

                    w = math.exp(-abs(tau) / self.lambda_decay)

                    weights.append(w)

                    indices.append(t_tau - 1)

        else:

            

            weights = []

            indices = []

            for tau in range(0, self.delta + 1):

                t_tau = t + tau

                if 1 <= t_tau <= N:

                    w = math.exp(-abs(tau) / self.lambda_decay)

                    weights.append(w)

                    indices.append(t_tau - 1)


        if not weights:

            return Y[t - 1]


        w_sum = sum(weights)

        result = torch.zeros_like(Y[0])

        for w, i in zip(weights, indices):

            result += (w / w_sum) * Y[i]

        return result


    def _smooth_fade(self, Y: Tensor, t: int, N: int,

                     transitions: list[int], policies: list[str],

                     segments: list[tuple[int, int]]) -> Tensor:

        

        nearest_fade = None

        fade_idx = None

        for j, (t_star, pol) in enumerate(zip(transitions, policies)):

            if pol == "fade" and abs(t - t_star) <= self.w_fade:

                nearest_fade = t_star

                fade_idx = j

                break


        if nearest_fade is None:

            return Y[t - 1]


        

        alpha_t = 0.5 * (1.0 - math.cos(

            math.pi * (t - (nearest_fade - self.w_fade)) / (2 * self.w_fade)

        ))

        alpha_t = max(0.0, min(1.0, alpha_t))


        

        seg_i = fade_idx  

        seg_i1 = fade_idx + 1  


        H_i = self._segment_mean(Y, t, N, segments[seg_i] if seg_i < len(segments) else None)

        H_i1 = self._segment_mean(Y, t, N, segments[seg_i1] if seg_i1 < len(segments) else None)


        return (1.0 - alpha_t) * H_i + alpha_t * H_i1


    def _segment_mean(self, Y: Tensor, t: int, N: int,

                      segment: tuple[int, int] | None) -> Tensor:

        if segment is None:

            return Y[t - 1]


        seg_start, seg_end = segment

        indices = []

        for tau in range(-self.delta, self.delta + 1):

            t_tau = t + tau

            if 1 <= t_tau <= N and seg_start <= t_tau < seg_end:

                indices.append(t_tau - 1)


        if not indices:

            return Y[t - 1]


        return torch.stack([Y[i] for i in indices]).mean(dim=0)

