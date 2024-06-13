# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import torch


def handle_feature_output(
    x: torch.Tensor, feature_reduce_method: Optional[str] = None, num_discard_tokens: int = 0
) -> torch.Tensor:
    """Handle feature output from transformer.

    Args:
        x (torch.Tensor): input feature to be handled. shape is
            [B, 1+H*W+N, C] if including both CLS and register tokens.
            [B, 1+H*W, C] for standard model (N=0).
            [B, H*W, C] for model without CLS.
        feature_reduce_method (Optional[str]): method to select token. Options:
            - `mean_pooling`: average over spatial tokens (non CLS tokens), output shape = [B, C].
            - `max_pooling`: max over spatial tokens, output shape = [B, C].
            - `cls`: return CLS token only, output shape = [B, C].
            - `identity`: return the feature without touching it, output shape = input shape.
            - `None`: return spatial tokens, output shape = [B, H*W, C] (assuming input is [B, 1+H*W, C]).
            suppose raw feature is in shape [B, 1+H*W, C], `1` corresponds to CLS token.
        num_discard_tokens (int):
            number of tokens to be discarded. Assuming they are at the end of the sequence.
    Returns:
        torch.Tensor: selected feature tokens.
    """

    match feature_reduce_method:
        case "mean_pooling":
            return torch.mean(x[:, 1 : x.size(1) - num_discard_tokens], dim=1)  # [B, C]
        case "max_pooling":
            return torch.amax(x[:, 1 : x.size(1) - num_discard_tokens], dim=1)  # [B, C]
        case "cls":
            return x[:, 0]  # [B, C]
        case "identity":
            return x
        case None:
            return x[:, 1 : x.size(1) - num_discard_tokens]
        case _:
            raise NotImplementedError(f"feature_reduce_method {feature_reduce_method} it not implemented.")
