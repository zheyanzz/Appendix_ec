

import logging


import cv2

import numpy as np

import torch

from torch import Tensor


from .base import BaseFlowExtractor


logger = logging.getLogger(__name__)


class FallbackFlowExtractor(BaseFlowExtractor):


    def __init__(self, method: str = "tvl1", device: str = "cuda"):

        assert method in ("tvl1", "frame_diff")

        self.method = method

        self.device = device

        self.has_displacement = method == "tvl1"


        if method == "tvl1":

            logger.info("Using TV-L1 optical flow (fallback)")

        else:

            logger.warning(

                "Using frame-difference proxy. has_displacement=False; "

                "SAML warp will be disabled."

            )


    def compute_flow(self, frames: Tensor) -> Tensor:

        if self.method == "tvl1":

            return self._tvl1(frames)

        return self._frame_diff(frames)


    def _tvl1(self, frames: Tensor) -> Tensor:

        N, C, H, W = frames.shape

        

        frames_np = (frames * 255).byte().cpu().numpy()  


        tvl1 = cv2.optflow.createOptFlow_DualTVL1()

        flow_list = [np.zeros((H, W, 2), dtype=np.float32)]


        for t in range(1, N):

            gray_prev = cv2.cvtColor(frames_np[t - 1].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

            gray_curr = cv2.cvtColor(frames_np[t].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

            flow = tvl1.calc(gray_prev, gray_curr, None)  

            flow_list.append(flow)


        flow_np = np.stack(flow_list, axis=0)  

        flow_tensor = torch.from_numpy(flow_np).permute(0, 3, 1, 2).float()  

        return flow_tensor.to(self.device)


    def _frame_diff(self, frames: Tensor) -> Tensor:

        N, C, H, W = frames.shape

        flow_list = [torch.zeros(1, 2, H, W)]


        for t in range(1, N):

            diff = (frames[t] - frames[t - 1]).pow(2).sum(dim=0).sqrt()  

            f = torch.stack([diff, diff], dim=0).unsqueeze(0)  

            flow_list.append(f)


        flow = torch.cat(flow_list, dim=0)  

        logger.warning("Frame-diff flow active — SAML warp disabled this step.")

        return flow.to(self.device)

