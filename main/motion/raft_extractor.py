

import logging

import os

import sys


import torch

from torch import Tensor


from .base import BaseFlowExtractor


logger = logging.getLogger(__name__)


class RAFTFlowExtractor(BaseFlowExtractor):


    has_displacement = True


    def __init__(self, model_path: str, device: str = "cuda",

                 small: bool = False, iters: int = 20):

        self.device = device

        self.iters = iters


        

        raft_core = os.path.join(

            os.path.dirname(__file__), "..", "..", "third_party", "RAFT", "core"

        )

        raft_core = os.path.abspath(raft_core)

        if raft_core not in sys.path:

            sys.path.insert(0, raft_core)


        from raft import RAFT  

        from argparse import Namespace


        args = Namespace(

            model=model_path,

            small=small,

            mixed_precision=False,

            alternate_corr=False,

        )

        self.model = torch.nn.DataParallel(RAFT(args))

        state = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(state)

        self.model = self.model.module.to(device)

        self.model.eval()

        for p in self.model.parameters():

            p.requires_grad_(False)

        logger.info("RAFT loaded from %s (iters=%d, small=%s)", model_path, iters, small)


    @torch.no_grad()

    def compute_flow(self, frames: Tensor) -> Tensor:

        N, C, H, W = frames.shape

        frames_dev = frames.to(self.device)

        

        frames_norm = frames_dev * 2.0 - 1.0


        flow_list = [torch.zeros(1, 2, H, W, device=self.device)]

        for t in range(1, N):

            _, flow_up = self.model(

                frames_norm[t - 1 : t],

                frames_norm[t : t + 1],

                iters=self.iters,

                test_mode=True,

            )

            flow_list.append(flow_up)  


        return torch.cat(flow_list, dim=0)  

