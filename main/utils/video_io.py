
import numpy as np
import torch
from torch import Tensor

def load_video(path: str, max_frames: int | None = None) -> Tensor:
    import imageio.v3 as iio

    frames = []
    for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if max_frames and i >= max_frames:
            break



        t = torch.from_numpy(frame.copy()).float() / 255.0
        t = t.permute(2, 0, 1)  
        frames.append(t)

    return torch.stack(frames)  


def save_video(frames: Tensor, path: str, fps: int = 8):
    import imageio.v3 as iio

    frames_np = (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    iio.imwrite(path, frames_np, fps=fps, plugin="pyav")

