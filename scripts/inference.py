"""Single-video stylization inference entry point.

Usage:
    python scripts/inference.py \
        --source content_video.mp4 \
        --style style_video.mp4 \
        --output stylized.mp4 \
        --checkpoint weights/quadstyle.ckpt
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadstyle.foundation import QuadStyleSystem
from quadstyle.utils.video_io import load_video, save_video
from quadstyle.utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_inference_config(device: str = "cuda") -> OmegaConf:
    """Load and merge all config files for inference (uses stage4 as base)."""
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

    # Loss weights (needed for model init)
    loss_path = os.path.join(base_dir, "loss", "weights.yaml")
    if os.path.exists(loss_path):
        cfg.loss = OmegaConf.load(loss_path)

    # Training config (needed for model init, use stage4)
    stage_path = os.path.join(base_dir, "train", "stage4.yaml")
    if os.path.exists(stage_path):
        cfg.train = OmegaConf.load(stage_path)
    else:
        cfg.train = OmegaConf.create({"lr": 5e-5, "weight_decay": 1e-2})

    cfg.device = device
    return cfg


def main():
    parser = argparse.ArgumentParser(description="QuadStyle Inference")
    parser.add_argument("--source", type=str, required=True, help="Source video path")
    parser.add_argument("--style", type=str, required=True, help="Style reference video/image path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output path")
    parser.add_argument("--checkpoint", type=str, default="weights/quadstyle.ckpt")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Build config
    cfg = load_inference_config(device=device)

    # Initialize model
    logger.info("Initializing QuadStyle model...")
    model = QuadStyleSystem(cfg)
    load_checkpoint(model, path=args.checkpoint)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load source video
    logger.info("Loading source video: %s", args.source)
    source = load_video(args.source, max_frames=args.num_frames)
    if source.shape[-2] != args.height or source.shape[-1] != args.width:
        source = F.interpolate(
            source, size=(args.height, args.width),
            mode="bilinear", align_corners=False,
        )
    source = source.unsqueeze(0).to(device=device, dtype=dtype)  # [1, N, 3, H, W]

    # Load style video/images
    logger.info("Loading style reference: %s", args.style)
    style = load_video(args.style, max_frames=args.num_frames)
    if style.shape[-2] != args.height or style.shape[-1] != args.width:
        style = F.interpolate(
            style, size=(args.height, args.width),
            mode="bilinear", align_corners=False,
        )
    # Pad style to match source frame count if needed
    if style.shape[0] < source.shape[1]:
        pad_n = source.shape[1] - style.shape[0]
        style = torch.cat([style, style[-1:].expand(pad_n, -1, -1, -1)], dim=0)
    style = style.unsqueeze(0).to(device=device, dtype=dtype)  # [1, N, 3, H, W]

    # Generate
    logger.info("Generating stylized video (%d denoising steps)...", args.num_steps)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda")):
        output = model.generate(
            source, style,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video(output[0].float().cpu(), args.output, fps=args.fps)
    logger.info("Saved stylized video to %s (%d frames at %d fps)",
                args.output, output.shape[1], args.fps)


if __name__ == "__main__":
    main()
