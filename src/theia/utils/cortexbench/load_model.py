# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose


def load_model(
    model: nn.Module, transform: Compose, metadata: Any, **kwargs: Any
) -> tuple[nn.Module, torch.Size, Compose, Any]:
    """Helper function for loading model for cortexbench.

    Args:
        model (nn.Module): model.
        transform (torchvision.transforms.Compose): transform applied to input image.
        metadata (Any): any metadata embedded in the model.
        kwargs (Any): any parameters for loading the model. Including
            `checkpoint_path` for loading weights for rvfm.

    Returns:
        tuple[nn.Module, torch.Size, Compose, Any]: return model, size of the embedding, transform, and the metadata.
    """

    if kwargs.get("checkpoint_path"):
        model.load_pretrained_weights(kwargs["checkpoint_path"])

    with torch.inference_mode():
        zero_img = np.array(Image.new("RGB", (100, 100)))  # for getting the embedding shape
        transformed_img = transform(zero_img).unsqueeze(0)
        embedding_dim = model.forward_feature(transformed_img).size()[1:]  # [H*W, C]
        if len(embedding_dim) > 1:
            h = w = int(math.sqrt(embedding_dim[0]))
            embedding_dim = torch.Size((embedding_dim[1], h, w))  # [C, H, W]

    return model, embedding_dim, transform, metadata
