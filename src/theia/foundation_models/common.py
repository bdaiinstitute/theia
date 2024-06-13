# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math

import torch

MODELS = [
    "facebook/dinov2-large",
    "facebook/sam-vit-huge",
    "google/vit-huge-patch14-224-in21k",
    "llava-hf/llava-1.5-7b-hf",
    "openai/clip-vit-large-patch14",
    "LiheYoung/depth-anything-large-hf",
]

# handy model feature size constants
# in the format of (latent_dim, width, height)
MODEL_FEATURE_SIZES = {
    "facebook/dinov2-large": (1024, 16, 16),
    "facebook/sam-vit-huge": (256, 64, 64),
    "google/vit-huge-patch14-224-in21k": (1280, 16, 16),
    "llava-hf/llava-1.5-7b-hf": (1024, 24, 24),
    "openai/clip-vit-large-patch14": (1024, 16, 16),
    "LiheYoung/depth-anything-large-hf": (32, 64, 64),
}


def get_model_feature_size(
    model_name: str, keep_spatial: bool = False, return_torch_size: bool = False
) -> tuple[int, ...] | torch.Size:
    """
    Get the size of queried model feature.

    Args:
        model_name (str): name of the model.
        keep_spatial (bool): whether to preserve spatial dim. Defaults to False.
        return_torch_size (bool): return torch.Size instead of python tuple. Defaults to False.

    Returns:
        tuple[int, ...] | torch.Size: the size of the feature.
    """
    size: tuple[int, ...] = MODEL_FEATURE_SIZES[model_name]

    if not keep_spatial:
        size = (size[0], math.prod(size[1:]))

    if return_torch_size:
        size = torch.Size(size)

    return size


def get_max_model_spatial_size(
    keep_spatial: bool = True,
    return_torch_size: bool = False,
    return_model_name: bool = False,
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """Get the maximal spatial dimensions from available models

    Args:
        keep_spatial (bool): whether to preserve spatial dim. Defaults to True.
        return_torch_size (bool): return torch.Size instead of python tuple. Defaults to False.
        return_model_name (bool): the name of the model with maximal size. Defaults to False.

    Returns:
        tuple[int, ...] | tuple[tuple[int, ...], str]: the maximal size and optional model name.
    """
    max_flatten_size = -1
    max_size: tuple[int, ...] = ()
    max_size_model_name: str = ""
    for model, size in MODEL_FEATURE_SIZES.items():
        flatten_size = math.prod(size[1:])
        if flatten_size > max_flatten_size:
            max_flatten_size = flatten_size
            max_size = size[1:]
            max_size_model_name = model

    if not keep_spatial:
        max_size = (max_flatten_size,)

    if return_torch_size:
        max_size = torch.Size(max_size)

    if return_model_name:
        return max_size, max_size_model_name
    else:
        return max_size
