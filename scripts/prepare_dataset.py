"""Build dataset and generate JSON annotation sidecars for multi-style videos."""

import argparse
import json
import logging
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def auto_annotate(video_dir: str, output_dir: str):
    """Generate JSON sidecar annotations for multi-style videos.

    Reads generation metadata (if available) or runs transition detection
    to identify transition frames and policies.

    Args:
        video_dir: directory containing multi-style MP4 files.
        output_dir: directory for JSON sidecars.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    logger.info("Found %d videos in %s", len(video_paths), video_dir)

    for video_path in video_paths:
        name = os.path.splitext(os.path.basename(video_path))[0]

        # Check for existing generation metadata
        meta_path = os.path.join(video_dir, f"{name}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            transitions = meta.get("transitions", [])
        else:
            # Run transition detection
            transitions = _detect_transitions(video_path)

        sidecar = {
            "video": os.path.basename(video_path),
            "transitions": transitions,
        }

        out_path = os.path.join(output_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        logger.info("  %s -> %d transitions", name, len(transitions))


def _detect_transitions(video_path: str) -> list[dict]:
    """Run transition detection on a video file.

    Args:
        video_path: path to MP4 file.

    Returns:
        list of {"frame": int, "policy": str} dicts.
    """
    try:
        import torch
        from quadstyle.utils.video_io import load_video
        from quadstyle.models.transition_detector import TransitionDetector
        from quadstyle.models.policy_classifier import PolicyClassifier
        import open_clip

        frames = load_video(video_path, max_frames=100)

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-14", pretrained="openai")
        clip_model.eval()

        detector = TransitionDetector(clip_model=clip_model)
        classifier = PolicyClassifier()

        clusters, delta_S = detector.detect(frames)
        transitions, policies, _ = classifier.classify(
            clusters, delta_S, N=frames.shape[0]
        )

        return [
            {"frame": t, "policy": p}
            for t, p in zip(transitions, policies)
        ]
    except Exception as e:
        logger.warning("Transition detection failed for %s: %s", video_path, e)
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare QuadStyle Dataset")
    parser.add_argument("--video_dir", type=str, required=True, help="Input video directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output annotation directory")
    parser.add_argument("--auto_annotate", action="store_true", help="Auto-generate annotations")
    args = parser.parse_args()

    if args.auto_annotate:
        auto_annotate(args.video_dir, args.output_dir)
    else:
        logger.info("Use --auto_annotate to generate JSON sidecars")


if __name__ == "__main__":
    main()
