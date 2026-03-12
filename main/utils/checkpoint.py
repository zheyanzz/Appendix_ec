

import logging

import os


import torch


logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, step: int, stage: int, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "step": step,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

    }
    torch.save(state, path)
    logger.info("Saved checkpoint to %s (step=%d, stage=%d)", path, step, stage)


def load_checkpoint(model, optimizer=None, path: str = "") -> dict:
    if not os.path.exists(path):
        logger.warning("No checkpoint found at %s", path)
        return {"step": 0, "stage": 1}


    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])


    logger.info("Loaded checkpoint from %s (step=%d, stage=%d)",
                path, state["step"], state["stage"])
    return {"step": state["step"], "stage": state["stage"]}

