

import json
import os
import glob

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class MultiStyleDataset(Dataset):


    def __init__(

        self,

        data_dirs: list[str],

        num_frames: int = 25,

        resolution: tuple[int, int] = (480, 720),

    ):

        self.num_frames = num_frames

        self.resolution = resolution


        self.items = []

        for d in data_dirs:

            if not os.path.exists(d):

                continue

            for json_path in sorted(glob.glob(os.path.join(d, "*.json"))):

                with open(json_path) as f:

                    meta = json.load(f)

                video_path = meta.get("video", "")

                if not os.path.isabs(video_path):

                    video_path = os.path.join(d, video_path)

                if os.path.exists(video_path):

                    self.items.append({

                        "video_path": video_path,

                        "transitions": meta.get("transitions", []),

                    })


    def __len__(self):

        return max(len(self.items), 1)


    def __getitem__(self, idx):

        if not self.items:

            return {

                "frames": torch.randn(self.num_frames, 3, *self.resolution),

                "transitions": [],

                "gt_boundary_mask": torch.zeros(self.num_frames),

                "gt_policies": torch.zeros(0, dtype=torch.long),

            }


        item = self.items[idx % len(self.items)]

        frames = self._load_video(item["video_path"])

        transitions = item["transitions"]


        

        gt_boundary = torch.zeros(self.num_frames)

        gt_policies = []

        for tr in transitions:

            frame_idx = tr["frame"]

            if 1 <= frame_idx <= self.num_frames:

                gt_boundary[frame_idx - 1] = 1.0

                gt_policies.append(0 if tr["policy"] == "cut" else 1)


        return {

            "frames": frames,

            "transitions": transitions,

            "gt_boundary_mask": gt_boundary,

            "gt_policies": torch.tensor(gt_policies, dtype=torch.long) if gt_policies else torch.zeros(0, dtype=torch.long),

        }


    def _load_video(self, path: str) -> torch.Tensor:

        from ..utils.video_io import load_video


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

