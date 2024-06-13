# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
import wandb


class SummaryType(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value
    Attributes:
        name (str): name of the meter.
        fmt (str): format string. Defaults to ':f'.
        summary_type (Enum): reduce method. Defaults to Summary.AVERAGE.

        val (float): last mean value over batch.
        avg (float): average value since meter creation.
        sum (float): sum of all the values = self.avg * self.count.
        count (int): number of values considered since meter creation.
    """

    def __init__(self, name: str, fmt: str = ":f", summary_type: SummaryType = SummaryType.AVERAGE) -> None:
        """Initialize an average meter."""
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        """Reset the meter."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter.

        Args:
            val (float): (mean) value over n samples.
            n (int): number of samples. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self) -> None:
        """Reduce meters across ranks."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=True)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """String representation of the meter."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        """Print the summary of the meter status."""
        fmtstr = ""
        match self.summary_type:
            case SummaryType.NONE:
                fmtstr = ""
            case SummaryType.AVERAGE:
                fmtstr = "{name} {avg:.3f}"
            case SummaryType.SUM:
                fmtstr = "{name} {sum:.3f}"
            case SummaryType.COUNT:
                fmtstr = "{name} {count:.3f}"
            case _:
                raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def create_meters(target_model_names: list[str]) -> dict[str, AverageMeter]:
    """Create meters for logging statistics, including individual meters for each target model.

    Args:
        target_model_names (list[str]): names of the target models.

    Returns:
        dict[str, AverageMeter]: meters created
    """
    meters = {}
    for loss in ["mse", "cos", "l1"]:
        meters[f"train_{loss}_loss"] = AverageMeter(f"train_{loss}_loss")
        meters[f"eval_{loss}_loss"] = AverageMeter(f"eval_{loss}_loss")

    for t in target_model_names:
        for loss in ["mse", "cos", "l1"]:
            for mode in ["train", "eval"]:
                meters[f"{mode}_{t}_{loss}_loss"] = AverageMeter(f"{mode}_{t}_{loss}_loss")

    return meters


def log_metrics(meters: dict[str, AverageMeter], **kwargs: Any) -> None:
    """log metrics to wandb.

    Args:
        meters (dict[str, AverageMeter]): _description_
    """
    metrics = {}

    mode = kwargs["mode"]
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 0

    if not kwargs.get("only_upload", False):
        # update meters
        meters[f"{mode}_mse_loss"].update(kwargs["mse_loss"].item(), n=batch_size)
        meters[f"{mode}_cos_loss"].update(kwargs["cos_loss"].item(), n=batch_size)
        meters[f"{mode}_l1_loss"].update(kwargs["l1_loss"].item(), n=batch_size)

        for t in kwargs["target_model_names"]:
            for loss in ["mse", "cos", "l1"]:
                meters[f"{mode}_{t}_{loss}_loss"].update(kwargs[f"{loss}_losses_per_model"][t], n=batch_size)

    # read out from meters or the raw for logging
    if kwargs["upload_wandb"]:
        if mode == "train":
            metrics["loss"] = kwargs["main_loss"].item()
            metrics["mse_loss"] = kwargs["mse_loss"].item()
            metrics["cos_loss"] = kwargs["cos_loss"].item()
            metrics["l1_loss"] = kwargs["l1_loss"].item()

        metrics[f"avg_{mode}_mse_loss"] = meters[f"{mode}_mse_loss"].avg
        metrics[f"avg_{mode}_cos_loss"] = meters[f"{mode}_cos_loss"].avg
        metrics[f"avg_{mode}_l1_loss"] = meters[f"{mode}_l1_loss"].avg
        for t in kwargs["target_model_names"]:
            for loss in ["mse", "cos", "l1"]:
                metrics[f"avg_{mode}_{t}_{loss}_loss"] = meters[f"{mode}_{t}_{loss}_loss"].avg

        if kwargs["device"] == 0:
            wandb.log(metrics)
