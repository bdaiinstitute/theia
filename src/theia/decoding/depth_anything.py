# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
import torch.nn as nn
from einops import rearrange
from theia.foundation_models.vision_models.depth_anything import DepthAnythingForDepthEstimation
from numpy.typing import NDArray
from torch.nn.functional import interpolate


def prepare_depth_decoder(model_name: str, device: int | str | torch.device = 0) -> tuple[nn.Module, int]:
    """Prepare a depth decoder using DepthAnythingForDepthEstimation.

    Args:
        model_name (str): name of the depth anything model.
        device (int | str | torch.device, optional): device to put the model on. Defaults to 0.

    Returns:
        tuple[nn.Module, int]: the decoder, and the patch size for depth anything model.
    """
    decoder_head = DepthAnythingForDepthEstimation.from_pretrained(model_name)
    patch_size = decoder_head.config.patch_size
    decoder_head = decoder_head.head
    decoder_head = decoder_head.to(device)
    return decoder_head, patch_size


def decode_depth_anything(features: torch.Tensor, decoder: nn.Module, device: int | str | torch.device = 0) -> NDArray:
    """Decode features to predicted depth using depth anything

    Args:
        features (torch.Tensor): features to be decoded, should be in shape [batch_size, num_tokens, latent_dim].
        decoder (nn.Module): depth anything decoder
        device (int | str | torch.device, optional): device to perform the decoding. Defaults to 0.

    Returns:
        NDArray: decoded depth in image format, represented by an NDArray in size [batch_size, height, width, channels]
            with value between [0, 1]. The depth values are min-max normalized to [0, 1] to generate images.
    """
    with torch.no_grad():
        P = int(features.size(1) ** 0.5)
        features = rearrange(features, "b (h w) c -> b c h w", h=P, w=P)
        features = interpolate(features, (224, 224))
        predicted_depths = []
        for feature in features:
            feature = feature.unsqueeze(0).to(device)

            predicted_depth = decoder.activation1(feature)
            predicted_depth = decoder.conv3(predicted_depth)
            predicted_depth = decoder.activation2(predicted_depth)
            predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)
            for i in range(len(predicted_depth)):
                min_depth, max_depth = predicted_depth[i].min(), predicted_depth[i].max()
                predicted_depth[i] = (predicted_depth[i] - min_depth) / (max_depth - min_depth)
            predicted_depths.append(predicted_depth.detach().cpu())
        predicted_depths = torch.cat(predicted_depths, dim=0)
    return predicted_depths.unsqueeze(-1).repeat((1, 1, 1, 3)).numpy()  # type: ignore [attr-defined]
