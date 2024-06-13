# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from numpy.typing import NDArray
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage


def totensor(arr: NDArray) -> torch.Tensor:
    """Convert ndarray to tensor."""
    return torch.from_numpy(arr)


oxe_image_transform = Compose(
    [ToImage(), ToDtype(torch.float32, scale=True), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)  # ImageNet statistics normalization
