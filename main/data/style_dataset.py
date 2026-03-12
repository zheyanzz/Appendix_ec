

import os

import glob


import torch

from torch.utils.data import Dataset

from torchvision import transforms


class StyleDataset(Dataset):


    def __init__(

        self,

        data_dirs: list[str],

        num_frames: int = 25,

        resolution: tuple[int, int] = (480, 720),

    ):

        self.num_frames = num_frames

        self.resolution = resolution


        self.paths = []

        for d in data_dirs:

            if not os.path.exists(d):

                continue

            self.paths.extend(sorted(

                glob.glob(os.path.join(d, "*.mp4")) +

                glob.glob(os.path.join(d, "*.jpg")) +

                glob.glob(os.path.join(d, "*.png"))

            ))


        self.transform = transforms.Compose([

            transforms.Resize(resolution),

            transforms.ToTensor(),

        ])


    def __len__(self):

        return max(len(self.paths), 1)


    def __getitem__(self, idx):

        if not self.paths:

            return torch.randn(self.num_frames, 3, *self.resolution)


        path = self.paths[idx % len(self.paths)]


        if path.endswith(".mp4"):

            return self._load_video(path)

        else:

            return self._load_image(path)


    def _load_video(self, path: str) -> torch.Tensor:

        from ..utils.video_io import load_video

        import torch.nn.functional as F


        frames = load_video(path, max_frames=self.num_frames)

        H, W = self.resolution

        frames = F.interpolate(frames, size=(H, W), mode="bilinear", align_corners=False)


        N = frames.shape[0]

        if N < self.num_frames:

            pad = self.num_frames - N

            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

        elif N > self.num_frames:

            frames = frames[:self.num_frames]


        return frames


    def _load_image(self, path: str) -> torch.Tensor:

        from PIL import Image


        img = Image.open(path).convert("RGB")

        t = self.transform(img)

        return t.unsqueeze(0).expand(self.num_frames, -1, -1, -1)

