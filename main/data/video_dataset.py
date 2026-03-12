

import os

import glob


import torch

from torch.utils.data import Dataset

from torchvision import transforms


class VideoDataset(Dataset):


    def __init__(

        self,

        data_dirs: list[str],

        num_frames: int = 25,

        resolution: tuple[int, int] = (480, 720),

        video_fraction: float = 1.0,

    ):

        self.num_frames = num_frames

        self.resolution = resolution  


        self.video_paths = []

        self.image_paths = []


        for d in data_dirs:

            if not os.path.exists(d):

                continue

            self.video_paths.extend(sorted(glob.glob(os.path.join(d, "*.mp4"))))

            self.image_paths.extend(sorted(

                glob.glob(os.path.join(d, "*.jpg")) +

                glob.glob(os.path.join(d, "*.png"))

            ))


        self.video_fraction = video_fraction

        self.all_items = []

        for p in self.video_paths:

            self.all_items.append(("video", p))

        for p in self.image_paths:

            self.all_items.append(("image", p))


        self.transform = transforms.Compose([

            transforms.Resize(resolution),

            transforms.ToTensor(),

        ])


    def __len__(self):

        return max(len(self.all_items), 1)


    def __getitem__(self, idx):

        if not self.all_items:

            

            return torch.randn(self.num_frames, 3, *self.resolution)


        item_type, path = self.all_items[idx % len(self.all_items)]


        if item_type == "video":

            return self._load_video(path)

        else:

            return self._load_image(path)


    def _load_video(self, path: str) -> torch.Tensor:

        from ..utils.video_io import load_video

        import torch.nn.functional as F


        frames = load_video(path, max_frames=self.num_frames)

        N = frames.shape[0]


        

        H, W = self.resolution

        frames = F.interpolate(frames, size=(H, W), mode="bilinear", align_corners=False)


        

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

