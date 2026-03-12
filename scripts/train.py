"""Curriculum training entry point for QuadStyle.

Supports multi-GPU via HuggingFace Accelerate:
    accelerate launch scripts/train.py --stage 1
    accelerate launch --multi_gpu --num_processes 4 scripts/train.py --stage 2

Single-GPU fallback:
    python scripts/train.py --stage 1
"""

import argparse
import logging
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadstyle.foundation import QuadStyleSystem
from quadstyle.data import VideoDataset, StyleDataset
from quadstyle.training import Trainer
from quadstyle.utils.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(stage: int = 1) -> OmegaConf:
    """Load and merge all config files for the given stage."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

    cfg = OmegaConf.create({})

    # Model configs
    model_cfgs = {}
    for name in [
        "wave_clip",
        "motion_injection",
        "style_attention",
        "canny_encoder",
        "tstm",
        "transition_detector",
        "attention_infra",
    ]:
        path = os.path.join(base_dir, "model", f"{name}.yaml")
        if os.path.exists(path):
            model_cfgs[name] = OmegaConf.load(path)
    cfg.model = OmegaConf.create(model_cfgs)

    # Flow configs
    flow_cfgs = {}
    for name in ["raft", "tvl1", "frame_diff"]:
        path = os.path.join(base_dir, "flow", f"{name}.yaml")
        if os.path.exists(path):
            flow_cfgs[name] = OmegaConf.load(path)
    cfg.flow = OmegaConf.create(flow_cfgs)
    cfg.flow.backend = "raft"

    # Loss weights
    loss_path = os.path.join(base_dir, "loss", "weights.yaml")
    if os.path.exists(loss_path):
        cfg.loss = OmegaConf.load(loss_path)

    # Stage-specific training config
    stage_path = os.path.join(base_dir, "train", f"stage{stage}.yaml")
    if os.path.exists(stage_path):
        cfg.train = OmegaConf.load(stage_path)
    else:
        cfg.train = OmegaConf.create({
            "lr": 1e-4, "weight_decay": 1e-2, "warmup_steps": 1000,
            "gradient_accumulation_steps": 4,
        })

    # Load all stage configs for loss scheduler stage_overrides
    training_stages = []
    for s in range(1, 5):
        sp = os.path.join(base_dir, "train", f"stage{s}.yaml")
        if os.path.exists(sp):
            training_stages.append(OmegaConf.load(sp))
    cfg.training_stages = OmegaConf.create(training_stages)

    # Apply loss overrides from stage config
    if hasattr(cfg.train, "loss_overrides"):
        for k, v in cfg.train.loss_overrides.items():
            cfg.loss[k] = v

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def main():
    parser = argparse.ArgumentParser(description="QuadStyle Training")
    parser.add_argument("--stage", type=int, default=1, help="Starting stage (1-4)")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--total_steps", type=int, default=250000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.stage)

    logger.info("Initializing QuadStyle model...")
    model = QuadStyleSystem(cfg)

    if args.resume:
        info = load_checkpoint(model, path=args.resume)
        start_step = info["step"]
        logger.info("Resumed from step %d", start_step)

    # Data loaders
    data_cfg = cfg.train.get("data", {})
    content_dataset = VideoDataset(
        data_dirs=list(data_cfg.get("content_dirs", [])),
        num_frames=data_cfg.get("num_frames", 25),
        resolution=tuple(data_cfg.get("resolution", [480, 720])),
    )
    style_dataset = StyleDataset(
        data_dirs=list(data_cfg.get("style_dirs", [])),
        num_frames=data_cfg.get("num_frames", 25),
        resolution=tuple(data_cfg.get("resolution", [480, 720])),
    )

    content_loader = DataLoader(
        content_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    style_loader = DataLoader(
        style_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )

    # Trainer handles Accelerate setup internally
    trainer = Trainer(model, cfg)

    if args.resume:
        trainer.global_step = start_step

    logger.info("Starting training from stage %d...", args.stage)

    # Checkpoint save callback
    os.makedirs(args.save_dir, exist_ok=True)

    def save_fn(model, optimizer, step, stage):
        path = os.path.join(args.save_dir, f"quadstyle_step{step}.ckpt")
        save_checkpoint(model, optimizer, step, stage, path)
        logger.info("Saved checkpoint: %s", path)

    trainer.train(
        content_loader, style_loader,
        total_steps=args.total_steps,
        save_fn=save_fn,
        save_every=args.save_every,
    )

    # Save final checkpoint
    save_checkpoint(
        trainer.model, trainer.optimizer, trainer.global_step,
        trainer.current_stage,
        os.path.join(args.save_dir, "quadstyle_final.ckpt"),
    )


if __name__ == "__main__":
    main()
