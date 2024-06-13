# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode


def rvfm_image_transforms(output_size: int = 224) -> T.Transform:
    """Image transform used by RVFM.

    Args:
        output_size (int): output size of the image.

    Returns:
        T.Compose: the transform
    """
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(output_size, interpolation=InterpolationMode.BICUBIC),
        ]
    )


def vit_transforms(resize_size: int = 256, output_size: int = 224) -> T.Transform:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def r3m_transforms(resize_size: int = 256, output_size: int = 224) -> T.Transform:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            T.ToDtype(torch.float32, scale=False),
        ]
    )
