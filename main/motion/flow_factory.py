

import importlib.util

import logging

import os


from .base import BaseFlowExtractor


logger = logging.getLogger(__name__)


def raft_available(path: str) -> bool:

    return os.path.exists(path) and importlib.util.find_spec("raft") is not None


def tvl1_available() -> bool:

    try:

        import cv2

        return hasattr(cv2, "optflow")

    except ImportError:

        return False


def get_flow_extractor(cfg) -> BaseFlowExtractor:

    backend = getattr(cfg.flow, "backend", "raft")

    device = getattr(cfg, "device", "cuda")


    if backend == "raft":

        raft_ckpt = cfg.flow.raft.checkpoint

        if raft_available(raft_ckpt):

            from .raft_extractor import RAFTFlowExtractor

            return RAFTFlowExtractor(

                raft_ckpt, device, cfg.flow.raft.small, cfg.flow.raft.iters

            )

        logger.warning("RAFT requested but unavailable; trying TV-L1.")


    if backend in ("raft", "tvl1") and tvl1_available():

        from .fallback_extractor import FallbackFlowExtractor

        logger.warning("Falling back to TV-L1 optical flow")

        return FallbackFlowExtractor("tvl1", device)


    from .fallback_extractor import FallbackFlowExtractor

    logger.warning("Falling back to frame-difference proxy. SAML warp disabled.")

    return FallbackFlowExtractor("frame_diff", device)

