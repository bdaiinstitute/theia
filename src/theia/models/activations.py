# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch.nn as nn


def get_activation_fn(activation: str) -> nn.Module:
    """Return specified activation function.

    Args:
        activation (str): the name of the activation function.

    Returns:
        nn.Module: the activation function in nn.Module.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"{activation} is not defined in theia/models/activations.py:get_activation_fn()")
