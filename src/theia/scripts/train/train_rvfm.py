# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Training script for theia, also called robot visual foundation model (RVFM) in
the code.
This training script uses hydra. To change configurations go for theia/configs.
"""

import math
import os.path as osp
import random
import warnings
from typing import Any, Callable

import hydra
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms.v2 import Compose
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from theia.models.rvfm import RobotVisionFM
from theia.optimizers.utils import param_groups_weight_decay
from theia.utils.logging import create_meters, log_metrics
from theia.utils.seed import seed_everything
from theia.foundation_models.common import MODEL_FEATURE_SIZES, get_model_feature_size
from theia.dataset.data_utils import get_frame_dataloader, get_frame_iterator, get_image_video_dataset
from theia.dataset.oxe.oxe_transforms import totensor


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def train(
    rvfm: nn.Module,
    target_model_names: list[str],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_dataset: Any,
    eval_dataset: Any,
    cfg: DictConfig,
    device: int = 0,
    train_epoch_steps: int = 0,
    eval_epoch_steps: int = 0,
    total_train_steps: int = 0,
    warmup_steps: int = 0,
) -> None:
    """Training and evaluation for robot visual foundation model (rvfm).

    Args:
        rvfm (nn.Module): model to train.
        target_model_names (list[str]): list of teacher model names.
        optimizer (torch.optim.Optimizer): optimizer.
        lr_scheduler (LRScheduler): learning rate scheduler.
        train_dataset (Any): train dataset.
        eval_dataset (Any): eval dataset.
        cfg (DictConfig): train config
        device (int, optional): device (of this process). Defaults to 0.
        train_epoch_steps (int, optional): steps per training epoch. Defaults to 0.
        eval_epoch_steps (int, optional): steps per eval epoch. Defaults to 0.
        total_train_steps (int, optional): total training steps. Defaults to 0.
        warmup_steps (int, optional): warmup steps. Defaults to 0.
    """
    epochs = cfg.training.epochs
    steps = 0
    # wrap the loaders so handle sync dataloaders easily
    for ep in range(epochs):

        train_loaders = get_frame_dataloader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            pin_memory=True,
            num_workers=cfg.training.num_workers,
            shuffle=cfg.dataset.shuffle,
            shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
            seed=cfg.seed + device * 100 + ep,  # either cfg.seed or cfg.seed + rank
        )
        eval_loaders = get_frame_dataloader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            pin_memory=True,
            num_workers=cfg.training.num_workers,
            shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
            seed=cfg.seed,  # either cfg.seed or cfg.seed + rank
        )
        train_iter = get_frame_iterator(train_loaders)

        metric_meters = create_meters(target_model_names)
        rvfm.train()
        train_tqdm = tqdm(range(train_epoch_steps), ncols=80) if device == 0 else range(train_epoch_steps)
        for _ in train_tqdm:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = get_frame_iterator(train_loaders)
                batch = next(train_iter)
            images_batch = batch["image"].to(device, non_blocking=True)
            if cfg.training.random_target_models > 0:
                batch_target_model_names = random.sample(target_model_names, 2)
            else:
                batch_target_model_names = target_model_names

            target_features_batch = {}
            for t in batch_target_model_names:
                base_name = t.replace("_cls", "")
                cls = True if "_cls" in t else False
                if cls:
                    target_features_batch[t] = batch[base_name]["cls"].to(device, non_blocking=True).float()
                else:
                    target_features_batch[t] = batch[base_name]["embedding"].to(device, non_blocking=True).float()

            pred = rvfm(images_batch)
            losses = rvfm.module.get_loss(pred, target_features_batch)

            if cfg.training.main_loss == "mse" or cfg.training.main_loss is None:
                main_loss = losses["mse_loss"]
            elif cfg.training.main_loss == "cos_l1":
                main_loss = 0.9 * losses["cos_loss"] + 0.1 * losses["l1_loss"]

            optimizer.zero_grad()
            main_loss.backward()
            if cfg.training.grad_clip:
                nn.utils.clip_grad_norm_(
                    rvfm.parameters(),
                    cfg.training.grad_clip_norm_warmup if steps < warmup_steps else cfg.training.grad_clip_norm,
                )
            optimizer.step()

            lr_scheduler.step()

            steps += 1
            batch_size = images_batch.size(0)

            log_metrics(
                metric_meters,
                target_model_names=target_model_names,
                device=device,
                batch_size=batch_size,
                mode="train",
                upload_wandb=True,
                main_loss=main_loss,
                **losses,
            )

            if cfg.training.freeze_translator:
                if steps == int(cfg.training.freeze_translator_start_steps_ratio * total_train_steps):
                    rvfm.module.freeze_translator()

            if steps % cfg.logging.save_ckpt_interval == 0 and device == 0:
                model_save_fn = f"{cfg.logging.run_identifier_prefix}_step{steps:08d}.pth"
                save_path = osp.join(cfg.logging.model_path, model_save_fn)
                torch.save(rvfm.module.state_dict(), save_path)

        dist.barrier()
        rvfm.eval()
        eval_iter = get_frame_iterator(eval_loaders)
        eval_tqdm = tqdm(range(eval_epoch_steps), ncols=80) if device == 0 else range(eval_epoch_steps)
        with torch.no_grad():
            for _ in eval_tqdm:
                batch = next(eval_iter)
                images_batch = batch["image"]
                target_features_batch = {}
                for t in target_model_names:
                    base_name = t.replace("_cls", "")
                    cls = True if "_cls" in t else False
                    if cls:
                        target_features_batch[t] = batch[base_name]["cls"].to(device, non_blocking=True).float()
                    else:
                        target_features_batch[t] = batch[base_name]["embedding"].to(device, non_blocking=True).float()

                pred = rvfm(images_batch)
                losses = rvfm.module.get_loss(pred, target_features_batch)
                if cfg.training.main_loss == "mse" or cfg.training.main_loss is None:
                    main_loss = losses["mse_loss"]
                elif cfg.training.main_loss == "cos_l1":
                    main_loss = 0.9 * losses["cos_loss"] + 0.1 * losses["l1_loss"]

                batch_size = images_batch.size(0)
                log_metrics(
                    metric_meters,
                    target_model_names=target_model_names,
                    device=device,
                    batch_size=batch_size,
                    mode="eval",
                    upload_wandb=False,
                    main_loss=main_loss,
                    **losses,
                )

        log_metrics(
            metric_meters,
            mode="eval",
            upload_wandb=True,
            only_upload=True,
            target_model_names=target_model_names,
            device=device,
        )

        if device == 0:
            model_save_fn = f"{cfg.logging.run_identifier_prefix}_step{steps:08d}.pth"
            save_path = osp.join(cfg.logging.model_path, model_save_fn)
            torch.save(rvfm.module.state_dict(), save_path)

        dist.barrier()


def ddp_setup() -> None:
    """Initialize stuff for DDP."""
    dist.init_process_group("nccl")


def ddp_cleanup() -> None:
    """Clean up stuff for DDP."""
    dist.destroy_process_group()


def ddp_main(cfg: DictConfig) -> None:
    """Entry point of DDP.

    Args:
        cfg (DictConfig): settings for training.
    """
    ddp_setup()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    target_model_names = (
        cfg.training.target_models.target_model_names
        if len(cfg.training.target_models.target_model_names) > 0
        else list(MODEL_FEATURE_SIZES.keys())
    )
    target_model_names = [t for t in target_model_names if "llava" not in t]  # llava is currently not supported
    target_feature_sizes = {t: get_model_feature_size(t, keep_spatial=True) for t in target_model_names}

    target_model_names_wocls = target_model_names[:]
    if hasattr(cfg.training, "distill_cls") and cfg.training.distill_cls == True:
        target_model_names_copy = target_model_names[:]
        for t in target_model_names:
            if "google/vit" in t or "facebook/dino" in t or "openai/clip" in t:
                target_feature_sizes[t+"_cls"] = get_model_feature_size(t, keep_spatial=True)[:1]
                target_model_names_copy.append(t+"_cls")

        target_model_names = target_model_names_copy

    rvfm = RobotVisionFM(
        translator=cfg.model.translator.type,
        translator_kwargs=cfg.model.translator.kwargs,
        target_feature_sizes=target_feature_sizes,
        target_loss_weights=cfg.training.target_models.target_model_weights,
        **cfg.model.backbone,
    )

    rvfm.to(rank)

    rvfm_ddp = DDP(rvfm, device_ids=[rank], find_unused_parameters=False)

    image_transform: Compose | Callable = totensor  # currently just ndarray to tensor

    train_dataset, train_dataset_expected_length = get_image_video_dataset(
        dataset_root=cfg.dataset.dataset_root,
        dataset_mix=cfg.dataset.dataset_mix,
        split="train",
        dataset_ratio=cfg.dataset.dataset_ratio,
        feature_models=target_model_names_wocls,
        image_transform=image_transform,
        feature_norm=cfg.dataset.feature_norm,
        rank=rank,
        world_size=world_size,
        shuffle=cfg.dataset.shuffle,
        seed=cfg.seed,
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
        num_workers=cfg.training.num_workers,
    )

    eval_dataset, eval_dataset_expected_length = get_image_video_dataset(
        dataset_root=cfg.dataset.dataset_root,
        dataset_mix=cfg.dataset.dataset_mix,
        split="val",
        dataset_ratio=0.1,
        feature_models=target_model_names_wocls,
        image_transform=image_transform,
        feature_norm=cfg.dataset.feature_norm,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        seed=cfg.seed,
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
        num_workers=cfg.training.num_workers,
    )

    train_epoch_steps = math.ceil(train_dataset_expected_length / cfg.training.batch_size / world_size)
    eval_epoch_steps = math.ceil(eval_dataset_expected_length / cfg.training.batch_size / world_size)
    total_train_steps = train_epoch_steps * cfg.training.epochs

    rvfm_param_groups = param_groups_weight_decay(rvfm_ddp, cfg.training.weight_decay)
    lr = cfg.training.base_lr * (
        (cfg.training.batch_size * world_size) / (cfg.training.base_batch_size * cfg.training.base_world_size)
    )
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, rvfm_param_groups, lr=lr)
    lr_scheduler = hydra.utils.instantiate(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        warm_up_steps=int(cfg.training.warm_up_steps_ratio * total_train_steps),
        cos_lrs_T_0=int(total_train_steps * (1 - cfg.training.warm_up_steps_ratio)),
    )

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))
        wandb.init(project=cfg.logging.project, name=cfg.logging.run_identifier_prefix, config=OmegaConf.to_object(cfg))

    train(
        rvfm_ddp,
        target_model_names,
        optimizer,
        lr_scheduler,
        train_dataset,
        eval_dataset,
        cfg=cfg,
        device=rank,
        train_epoch_steps=train_epoch_steps,
        eval_epoch_steps=eval_epoch_steps,
        total_train_steps=total_train_steps,
        warmup_steps=int(cfg.training.warm_up_steps_ratio * total_train_steps),
    )

    ddp_cleanup()


@hydra.main(version_base=None, config_path="../../configs", config_name="train_rvfm_imagenet")
def main(cfg: DictConfig) -> None:
    """Main. Dealing with arguments and call DDP."""

    backbone_fn = f"_{cfg.model.backbone.backbone.replace('/', '-')}"
    notes_fn = f"_{cfg.logging.notes}" if cfg.logging.notes else ""
    translator_fn = f"_{cfg.model.translator.type}"
    pretrained_fn = "_pretrained" if cfg.model.backbone.pretrained else ""
    dp_fn = f"_dp{cfg.dataset.dataset_ratio:.3f}"
    cfg.logging.run_identifier_prefix = f"rvfm{dp_fn}{backbone_fn}{translator_fn}{pretrained_fn}{notes_fn}"

    seed_everything(cfg.seed)

    ddp_main(cfg)


if __name__ == "__main__":
    main()
