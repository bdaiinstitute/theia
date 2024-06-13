# File modified. Modifications Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ConvBatchNormMLPDeterministicPolicy(nn.Module):
    def __init__(
        self,
        in_dim: tuple[int, ...],
        extra_dim: int,
        out_dim: int,
        max_a: Any = None,
        hidden_size: int = 256,
        nonlinearity: str = "relu",
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.extra_dim = extra_dim
        self.in_dim = in_dim
        self.neck = nn.Sequential(
            Rearrange("b (h w c) -> b c h w", h=14, w=14),
            nn.Conv2d(in_dim[0], 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),  # 7x7 -> 3x3
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),  # 3x3 -> 1x1
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Flatten(),
        )
        self.policy = nn.Sequential(
            nn.Linear(256 + extra_dim, hidden_size),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Linear(hidden_size, out_dim),
        )
        self.neck.to(device)
        self.policy.to(device)
        self.device = device

        self.init_state = copy.deepcopy(self.policy.state_dict())
        self.neck_init_state = copy.deepcopy(self.neck.state_dict())

        self.max_a = max_a
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        visual_state = state[..., : -self.extra_dim]
        feature = self.neck(visual_state)
        if self.extra_dim > 0:
            feature = torch.cat([feature, state[..., -self.extra_dim :]], dim=1)
        action = self.policy(feature)
        return action

    def reset(self) -> None:
        self.policy.load_state_dict(self.init_state)
        self.neck.load_state_dict(self.neck_init_state)

    def clip_action(self, a: torch.Tensor) -> torch.Tensor:
        if self.max_a is not None:
            a = torch.where(a > self.max_a, torch.tensor([self.max_a]).to(self.device), a)
            a = torch.where(a < -self.max_a, -torch.tensor([self.max_a]).to(self.device), a)
        return a

    def scale_to_range(self, a: torch.Tensor) -> torch.Tensor:
        """Does not do anything; just returns a"""
        return a


def construct_policy(
    type: str,
    task_state_type: str,
    train_ft_state_shape: int,
    pretrained_dim: tuple[int, ...],
    task_goal_type: str,
    out_dim: int,
    max_a: Any,
    device: str | int | torch.device,
    hidden_size: int = 256,
    nonlinearity: str = "relu",
    **kwargs: Any,
) -> ConvBatchNormMLPDeterministicPolicy:
    in_dim = pretrained_dim
    extra_dim = 0
    if task_state_type == "obj":
        extra_dim += 0
    elif task_state_type in ["ftpos_obj", "ftpos"]:
        extra_dim += train_ft_state_shape
    else:
        raise NameError("Invalid state_type")

    if task_goal_type == "goal_none":
        in_dim = pretrained_dim
    elif task_goal_type == "goal_cond":
        in_dim = (pretrained_dim[0] * 2, *pretrained_dim[1:])
    elif task_goal_type == "goal_o_pos":
        extra_dim += 3
    else:
        raise NameError("Invalid goal_type")

    if type == "ConvBatchNormMLP":
        policy = ConvBatchNormMLPDeterministicPolicy(
            in_dim=in_dim,
            extra_dim=extra_dim,
            out_dim=out_dim,
            max_a=max_a,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            device=device,
        )
    else:
        raise NotImplementedError(f"Policy network {type} is not supported.")
    return policy
