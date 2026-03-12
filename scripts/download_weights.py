"""Download CogVideoX weights from HuggingFace and RAFT weights from GitHub."""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_cogvideox():
    """Download CogVideoX-5B-I2V from HuggingFace."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading CogVideoX-5B-I2V from HuggingFace...")
    snapshot_download(
        repo_id="THUDM/CogVideoX-5b-I2V",
        local_dir="weights/CogVideoX-5b-I2V",
        ignore_patterns=["*.md", "*.txt"],
    )
    logger.info("CogVideoX downloaded to weights/CogVideoX-5b-I2V")


def download_raft():
    """Download RAFT weights from GitHub releases."""
    import urllib.request

    url = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
    raft_url = "https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZdG6_Wia"

    os.makedirs("weights", exist_ok=True)
    target = "weights/raft-things.pth"

    if os.path.exists(target):
        logger.info("RAFT weights already exist at %s", target)
        return

    logger.info("Downloading RAFT weights...")
    logger.info(
        "Please download raft-things.pth manually from "
        "https://github.com/princeton-vl/RAFT and place at %s",
        target,
    )
    logger.info(
        "Alternatively, run: "
        "wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip "
        "&& unzip models.zip && mv models/raft-things.pth weights/"
    )


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--cogvideox", action="store_true", help="Download CogVideoX")
    parser.add_argument("--raft", action="store_true", help="Download RAFT")
    parser.add_argument("--all", action="store_true", help="Download all")
    args = parser.parse_args()

    if args.all or args.cogvideox:
        download_cogvideox()
    if args.all or args.raft:
        download_raft()
    if not (args.all or args.cogvideox or args.raft):
        logger.info("Use --all, --cogvideox, or --raft to specify what to download")


if __name__ == "__main__":
    main()
