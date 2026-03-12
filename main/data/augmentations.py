

import random


import torch

import torchvision.transforms.functional as TF

from torch import Tensor


def augment_style(

    frames: Tensor,

    p_flip: float = 0.5,

    p_jitter: float = 0.3,

    p_crop: float = 0.5,

    crop_scale: tuple[float, float] = (0.8, 1.0),

) -> Tensor:

    N, C, H, W = frames.shape


    

    if random.random() < p_crop:

        scale = random.uniform(crop_scale[0], crop_scale[1])

        crop_h = int(H * scale)

        crop_w = int(W * scale)

        top = random.randint(0, H - crop_h)

        left = random.randint(0, W - crop_w)

        frames = frames[:, :, top : top + crop_h, left : left + crop_w]

        frames = torch.nn.functional.interpolate(

            frames, size=(H, W), mode="bilinear", align_corners=False

        )


    

    if random.random() < p_flip:

        frames = torch.flip(frames, dims=[-1])


    

    if random.random() < p_jitter:

        brightness = random.uniform(0.8, 1.2)

        contrast = random.uniform(0.8, 1.2)

        saturation = random.uniform(0.8, 1.2)


        frames = frames.clamp(0, 1)

        frames = TF.adjust_brightness(frames, brightness)

        frames = TF.adjust_contrast(frames, contrast)

        frames = TF.adjust_saturation(frames, saturation)

        frames = frames.clamp(0, 1)


    return frames


def augment_content(

    frames: Tensor,

    p_flip: float = 0.5,

    p_crop: float = 0.3,

    crop_scale: tuple[float, float] = (0.85, 1.0),

) -> Tensor:

    N, C, H, W = frames.shape


    

    if random.random() < p_crop:

        scale = random.uniform(crop_scale[0], crop_scale[1])

        crop_h = int(H * scale)

        crop_w = int(W * scale)

        top = random.randint(0, H - crop_h)

        left = random.randint(0, W - crop_w)

        frames = frames[:, :, top : top + crop_h, left : left + crop_w]

        frames = torch.nn.functional.interpolate(

            frames, size=(H, W), mode="bilinear", align_corners=False

        )


    

    if random.random() < p_flip:

        frames = torch.flip(frames, dims=[-1])


    return frames

