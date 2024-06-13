# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Iterable

import torch.nn as nn


def param_groups_weight_decay(
    model: nn.Module, weight_decay: float = 1e-5, no_weight_decay_parameters: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Group parameters into sets with decay applied and no decay.

    Args:
        model (nn.Module): the model.
        weight_decay (float): weight decay. Defaults to 1e-5.
        no_weight_decay_parameters (Iterable[str]): parameters added to no weight decay
            in addition to defaults. Defaults to ().

    Returns:
        list[dict[str, Any]]: parameter groups with different weight decay values.
            Follow the format required by torch.optim.Optimizer.
    """
    no_weight_decay_parameters = set(no_weight_decay_parameters)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_parameters:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def param_groups_lr_weight_decay(
    model: nn.Module,
    backbone_lr: float = 1e-3,
    translator_lr: float = 1e-3,
    weight_decay: float = 1e-5,
    no_weight_decay_parameters: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Group parameters into set with decay applied and no decay.

    Args:
        model (nn.Module): the model.
        weight_decay (float): weight decay. Defaults to 1e-5.
        no_weight_decay_parameters (Iterable[str]): parameters added to no weight decay
            in addition to defaults. Defaults to ().

    Returns:
        list[dict[str, Any]]: parameter groups with different weight decay values.
            Follow the format required by torch.optim.Optimizer.
    """
    no_weight_decay_parameters = set(no_weight_decay_parameters)
    decay_backbone = []
    no_decay_backbone = []
    decay_translator = []
    no_decay_translator = []

    for name, param in model.module.backbone.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_parameters:
            no_decay_backbone.append(param)
        else:
            decay_backbone.append(param)

    for name, param in model.module.translator.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_parameters:
            no_decay_translator.append(param)
        else:
            decay_translator.append(param)

    return [
        {"params": no_decay_backbone, "weight_decay": 0.0, "lr": backbone_lr},
        {"params": decay_backbone, "weight_decay": weight_decay, "lr": backbone_lr},
        {"params": no_decay_translator, "weight_decay": 0.0, "lr": translator_lr},
        {"params": decay_translator, "weight_decay": weight_decay, "lr": translator_lr},
    ]
