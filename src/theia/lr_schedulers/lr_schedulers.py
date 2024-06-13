# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, ConstantLR


def get_cos_lrs_with_linear_warm_up(
    optimizer: Optimizer,
    warm_up_steps: int = 2000,
    warm_up_lr_start_factor: float = 1e-2,
    warm_up_lr_end_factor: float = 1.0,
    cos_lrs_T_0: int = 5000,
) -> SequentialLR:
    """Get a cos annealing warm restarts lr scheduler with linear warm up at the beginning.

    Args:
        optimizer (Optimizer): original optimizer to be scheduled.
        warm_up_steps (int): number of warm up steps. Defaults to 2000.
        warm_up_lr_start_factor (float): start factor of the linear schedular. Defaults to 1e-2.
        warm_up_lr_end_factor (float): end factor of the linear scheduler. Defaults to 1.
        cos_lrs_T_0 (int): T_0 param of cos lrs. Number of steps per cycle. Defaults to 5000.

    Returns:
        SequentialLR: a sequential lrs that combines linear and cos to implement warm up.
    """
    linear_lrs = LinearLR(
        optimizer=optimizer,
        start_factor=warm_up_lr_start_factor,
        end_factor=warm_up_lr_end_factor,
        total_iters=warm_up_steps,
    )

    cos_lrs = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=cos_lrs_T_0, T_mult=1)

    lrs = SequentialLR(optimizer=optimizer, schedulers=[linear_lrs, cos_lrs], milestones=[warm_up_steps])

    return lrs


def get_constant_lrs_with_linear_warm_up(
    optimizer: Optimizer,
    warm_up_steps: int = 2000,
    warm_up_lr_start_factor: float = 1e-2,
    warm_up_lr_end_factor: float = 1.,
    **kwargs: Any
) -> SequentialLR:
    """Get a constant lr scheduler with linear warm up at the beginning.

    Args:
        optimizer (Optimizer): original optimizer to be scheduled.
        warm_up_steps (int): number of warm up steps. Defaults to 2000.
        warm_up_lr_start_factor (float): start factor of the linear schedular. Defaults to 1e-2.
        warm_up_lr_end_factor (float): end factor of the linear scheduler. Defaults to 1.

    Returns:
        SequentialLR: a sequential lrs that combines linear and constant lrs to implement warm up.
    """
    linear_lrs = LinearLR(
        optimizer = optimizer, 
        start_factor = warm_up_lr_start_factor, 
        end_factor = warm_up_lr_end_factor,
        total_iters = warm_up_steps
    )

    constant_lrs = ConstantLR(
        optimizer = optimizer,
        factor=1.0
    )

    lrs = SequentialLR(
        optimizer = optimizer,
        schedulers = [linear_lrs, constant_lrs],
        milestones = [warm_up_steps]
    )

    return lrs
