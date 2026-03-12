

import numpy as np

import torch

from torch import Tensor


try:

    import cv2

except ImportError:  

    cv2 = None


def _fallback_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:

    gray_f = gray.astype(np.float32) / 255.0

    gy, gx = np.gradient(gray_f)

    mag = np.sqrt(gx * gx + gy * gy)

    threshold = max(low, high) / 255.0

    return (mag > threshold).astype(np.float32)


def extract_canny_batch(

    frames: Tensor,

    low: int = 50,

    high: int = 150,

) -> Tensor:

    B, N, C, H, W = frames.shape

    device = frames.device


    frames_np = (frames * 255).byte().cpu().numpy()

    canny_out = np.zeros((B, N, 1, H, W), dtype=np.float32)


    for b in range(B):

        for t in range(N):

            frame_rgb = frames_np[b, t].transpose(1, 2, 0)  

            if cv2 is not None:

                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

                edges = cv2.Canny(gray, low, high).astype(np.float32) / 255.0

            else:

                gray = frame_rgb.mean(axis=-1).astype(np.uint8)

                edges = _fallback_edges(gray, low, high)

            canny_out[b, t, 0] = edges


    return torch.from_numpy(canny_out).to(device)

