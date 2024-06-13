# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Defines PyTorch datasets of dataloaders for multiple image, video, and OXE datasets.
Should use with webdataset >= 0.2.90. See https://github.com/webdataset/webdataset/pull/347"""

import glob
import json
import math
import os.path as osp
from collections import OrderedDict
from functools import partial
from io import BytesIO
from typing import Any, Callable, Generator, Iterator, Literal, Optional

import cv2
import numpy as np
import omegaconf
import torch
import webdataset as wds
from datasets.combine import DatasetType
from einops import rearrange
from numpy.typing import NDArray
from safetensors.torch import load as sft_load
from torch import default_generator
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate

from theia.foundation_models.common import MODELS
from theia.dataset.oxe.oxe_common import ALL_OXE_DATASETS
from theia.dataset.oxe.oxe_mixes import OXE_NAMED_MIXES

PACKED_FEATURES = [model_name for model_name in MODELS if "llava" not in model_name]


def normalize_ds_weights_by_ds_len(weights: list[float], lengths: list[int]) -> tuple[list[float], float | Literal[0]]:
    """Normalize dataset weights by dataset lengths (frames).

    Args:
        weights (list[float]): assigned weights.
        lengths (list[int]): lengths of datasets.

    Returns:
        tuple[list[float], int]: normalized weights, and sum of the expected lengths of datasets
    """
    expected_lengths = [weight * length for weight, length in zip(weights, lengths, strict=False)]
    sum_expected_lengths = sum(expected_lengths)
    if sum_expected_lengths == 0:
        raise ValueError("Sum of dataset length is 0.")
    normalized_weights = [length * 1.0 / sum_expected_lengths for length in expected_lengths]
    return normalized_weights, sum_expected_lengths


def get_vo_keys(dataset_name: str, image_views: Optional[list | str | dict[str, str | list[str]]] = None) -> list[str]:
    """Get visual observation keys of datasets (to be compatible with OXE).

    Args:
        dataset_name (str): name of the dataset.
        image_views (Optional[dict[str, str  |  list[str]]], optional): keys of selected views.
            Defaults to None.

    Returns:
        list[str]: keys to the views in the dataset.
    """
    default_visual_observation_keys = ALL_OXE_DATASETS[dataset_name]["visual_observation_keys"][:1]
    visual_observation_keys = []
    if image_views is None:
        visual_observation_keys = default_visual_observation_keys
    elif isinstance(image_views, list):
        visual_observation_keys = ALL_OXE_DATASETS[dataset_name]["visual_observation_keys"]
    elif isinstance(image_views, str):
        if image_views == "static":
            visual_observation_keys = [
                k
                for k in ALL_OXE_DATASETS[dataset_name]["visual_observation_keys"]
                if "wrist" not in k and "hand" not in k
            ]
        elif image_views == "wrist":
            visual_observation_keys = [
                k for k in ALL_OXE_DATASETS[dataset_name]["visual_observation_keys"] if "wrist" in k or "hand" in k
            ]
    if len(visual_observation_keys) == 0:
        visual_observation_keys = default_visual_observation_keys
    return visual_observation_keys


class RandomMix(IterableDataset):
    """A random interleave of multiple iterable datasets."""

    def __init__(
        self,
        datasets: list[IterableDataset],
        probs: list[float] | NDArray | None = None,
        stopping_strategy: str = "all_exhausted",
        seed: Optional[int | str] = 0,
    ) -> None:
        """Initialization of a random interleave dataset.

        Args:
            datasets (list[IterableDataset]): datasets to be interleaved.
            probs (list[float] | NDArray, optional): probability of each dataset. Defaults to None.
            stopping_strategy (str, optional): when to end the sampling for one epoch. Defaults to `all_exhausted`.
                `all_exhausted`: each sample in the dataset will be sampled at least once.
                `first_exhausted`: when the first dataset is ran out, this episode ends.
                See also https://huggingface.co/docs/datasets/en/stream#interleave for definitions.
            seed (Optional[int | str]): seed. Defaults to 0.
        """
        self.datasets = datasets
        if probs is None:
            self.probs = [1.0] * len(self.datasets)
        elif isinstance(probs, np.ndarray):
            self.probs = probs.tolist()
        else:
            self.probs = probs
        self.stopping_strategy = stopping_strategy
        self.seed = seed

    def __iter__(self) -> Generator:
        """Return an iterator over the sources."""
        sources = [iter(d) for d in self.datasets]
        probs = self.probs[:]
        seed_gen = torch.Generator()
        seed_gen.manual_seed(self.seed)
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        while len(sources) > 0:
            r = torch.rand(1, generator=seed_gen).item()
            i = np.searchsorted(cum, r)
            try:
                yield next(sources[i])
            except StopIteration:
                if self.stopping_strategy == "all_exhausted":
                    del sources[i]
                    del probs[i]
                    cum = (np.array(probs) / np.sum(probs)).cumsum()
                elif self.stopping_strategy == "first_exhausted":
                    break


def decode_sample(
    key: str, data: bytes, image_transform: Optional[Callable] = None, feature_transform: Optional[Callable] = None
) -> Any:
    """Decode a sample from bytes with optional image and feature transforms

    Args:
        key (str): key of an attribute (a column) of the sample.
        data (bytes): original data bytes.
        image_transform (Optional[Callable], optional): image transform. Defaults to None.
        feature_transform (Optional[Callable], optional): feature transform. Defaults to None.

    Returns:
        Any: decoded data.
    """
    if ".safetensors" in key:
        sft = sft_load(data)
        embedding = rearrange(sft["embedding"], "c h w -> (h w) c")
        if feature_transform is not None:
            embedding = feature_transform(embedding)
        if "cls_token" in sft:
            cls = sft["cls_token"]
            if feature_transform is not None:
                cls = feature_transform(cls)
                return {"embedding": embedding, "cls": cls}
        return {"embedding": embedding}
    elif key == ".image":
        image = np.load(BytesIO(data))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image_transform is not None:
            return image_transform(image)
        return image
    else:
        return data


def get_oxe_frame_dataset(
    dataset_root: str,
    dataset_mix: Optional[str | dict[str, float] | list] = "oxe_magic_soup",
    feature_models: Optional[list[str]] = None,
    split: str = "train",
    dataset_ratio: float = 1.0,
    image_views: Optional[dict[str, str | list[str]]] = None,
    image_transform: Optional[Callable[[Any], torch.Tensor]] = None,
    seed: Optional[int | str] = 0,
    shuffle: bool = False,
    world_size: int = 1,
) -> tuple[dict[str, DatasetType], float | Literal[0]]:
    """Get OXE datasets at frame level.

    Args:
        dataset_root (str): root dir of the datasets.
        dataset_mix (Optional[str  |  dict[str, float]  |  list], optional): how to mix the datasets.
            Defaults to "oxe_magic_soup".
        feature_models (Optional[list[str]], optional): models to load their features. Defaults to None.
        split (str, optional): split "train" or "val" or "test". Defaults to "train".
        dataset_ratio (float, optional): how much data use for the (combined) dataset. Defaults to 1.0.
        image_views (Optional[dict[str, str  |  list[str]]], optional): image views to select. Defaults to None.
        image_transform (Optional[Callable[[Any], torch.Tensor]], optional): image transform applied to samples.
            Defaults to None.
        seed (Optional[int  |  str], optional): seed. Defaults to 0.
        shuffle (bool, optional): shuffle or not. Defaults to False.
        world_size (int, optional): world size of DDP training. Defaults to 1.

    Returns:
        tuple[dict[str, DatasetType], int]: a dict of {dataset name: dataset class}.
    """
    # read dataset mix from any acceptable form
    if isinstance(dataset_mix, str) and dataset_mix in OXE_NAMED_MIXES:
        dataset_mix = OrderedDict({k: v for k, v in OXE_NAMED_MIXES[dataset_mix]})
    elif isinstance(dataset_mix, dict):
        dataset_mix = OrderedDict(**dataset_mix)
    elif isinstance(dataset_mix, list):
        dataset_mix = OrderedDict({d: 1.0 for d in dataset_mix})
    else:
        raise ValueError(f"dataset_mix of {dataset_mix}:{type(dataset_mix)} is not supported.")

    if split == "eval" or split == "val":
        dataset_mix = OrderedDict({d: 1.0 for d in dataset_mix})

    # note down the dataset weights
    dataset_weights: list[float] = []
    # get frame level length
    dataset_lens: list[int] = []

    all_feature_datasets: dict[str, DatasetType] = {}
    for dataset in dataset_mix:
        visual_observation_keys = get_vo_keys(dataset_name=dataset, image_views=image_views)

        if feature_models is None:
            feature_models = PACKED_FEATURES

        with open(osp.join(dataset_root, dataset, "splits.json"), "r") as splitf:
            dataset_len = json.load(splitf)[split]
        # if the length is 0, skip
        # this may happen for small datasets with very few shards
        if dataset_len == 0:
            continue

        for vo_key in visual_observation_keys:
            for model_name in feature_models:
                if model_name not in PACKED_FEATURES:
                    feature_set_name = model_name
                    path_pattern = osp.join(
                        dataset_root, dataset, vo_key + f"_{model_name.replace('/', '_')}", f"*-{split}*.tar"
                    )
                    rename_kw = {model_name: model_name.replace("/", "_") + ".safetensors"}  # replace v by k
                elif "packed" in all_feature_datasets:
                    continue
                else:
                    feature_set_name = "packed"
                    path_pattern = osp.join(dataset_root, dataset, vo_key, f"*-{split}*.tar")
                    rename_kw = {
                        name: name.replace("/", "_") + ".safetensors" for name in PACKED_FEATURES
                    }  # replace v by k
                rename_kw["image"] = "image"

                if feature_set_name not in all_feature_datasets:
                    all_feature_datasets[feature_set_name] = []

                shard_paths = sorted(glob.glob(path_pattern))
                num_shards = len(shard_paths)
                if num_shards < world_size * 8:
                    shard_paths *= math.ceil(world_size * 8 / num_shards)
                ds = (
                    wds.WebDataset(
                        shard_paths,
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker,
                        detshuffle=True,
                        shardshuffle=shuffle,
                        seed=seed,
                    )
                    .decode(partial(decode_sample, image_transform=image_transform))
                    .rename(keep=False, **rename_kw)
                )
                all_feature_datasets[feature_set_name].append(ds)

            dataset_weights.append(dataset_mix[dataset])
            dataset_lens.append(math.ceil(dataset_len * dataset_ratio))

    normalized_dataset_weights, sum_expected_lengths = normalize_ds_weights_by_ds_len(dataset_weights, dataset_lens)

    combined_feature_datasets: dict[str, Dataset] = {}
    for feature_set_name, fds in all_feature_datasets.items():
        ds = RandomMix(fds, probs=normalized_dataset_weights, stopping_strategy="all_exhausted")
        combined_feature_datasets[feature_set_name] = ds

    return combined_feature_datasets, sum_expected_lengths


def get_oxe_frame_dataloader(
    datasets: dict[str, DatasetType], batch_size: Optional[int] = None, shuffle_buffer_size: int = 1_000, **kwargs: Any
) -> dict[str, DataLoader]:
    """Get dataloaders of OXE datasets. Corresponding to `get_oxe_frame_dataset()`.

    Args:
        datasets (dict[str, DatasetType]): OXE datasets from `get_oxe_frame_dataset().
        batch_size (Optional[int], optional): batch size. Defaults to None.
        shuffle_buffer_size (int, optional): buffer for shuffle while streaming. Defaults to 1_000.

    Returns:
        dict[str, DataLoader]: dataloaders. a dict of {dataset name: dataloader}.
    """
    loaders = {
        k: (
            wds.WebLoader(datasets[k], batch_size=None, **kwargs)
            .shuffle(shuffle_buffer_size)  # shuffle after mix
            .batched(batch_size, collation_fn=default_collate)
        )
        for k in datasets
    }
    return loaders


def get_oxe_frame_iterator(
    data_loaders: dict[str, DataLoader],
) -> Iterator[dict[str, Any]]:
    """Get iterator from dataloders. Corresponding to `get_oxe_frame_dataloader()`.

    Args:
        data_loaders (dict[str, DataLoader]): dataloaders from `get_oxe_frame_dataloader()`.

    Yields:
        Iterator[dict[str, Any]]: data sample.
    """
    packed_loader = data_loaders.get("packed", None)
    # place packed_loader at the first
    if packed_loader is not None:
        loaders = [packed_loader, *[data_loaders[k] for k in data_loaders if k != "packed"]]
    else:
        loaders = list(data_loaders.values())

    # merge dicts
    for data in zip(*loaders, strict=False):
        # yield data
        for i in range(1, len(loaders)):
            for k in data[i]:
                if k not in data[0]:
                    data[0][k] = data[i][k]
        yield data[0]


def normalize_feature(
    x: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Normalize the feature given mean and std.

    Args:
        x (torch.Tensor): input features
        mean (Optional[torch.Tensor], optional): mean values. Defaults to None.
        std (Optional[torch.Tensor], optional): std values. Defaults to None.

    Returns:
        torch.Tensor: feature after normalization
    """
    return x if mean is None or std is None else (x - mean) / std


def load_feature_stats(
    dataset_root: str, feature_models: list[str]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load feature statictics (mean and variance).

    Args:
        dataset_root (str): root dir of the dataset (or where to hold the statistics).
        feature_models (list[str]): names of the models/features.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: means and variances. Keys are model names.
    """
    feature_means: dict[str, torch.Tensor] = {}
    feature_vars: dict[str, torch.Tensor] = {}
    for model in feature_models:
        model_name = model.replace("/", "_")
        feature_means[model] = torch.from_numpy(np.load(osp.join(dataset_root, f"imagenet_mean_{model_name}.npy"))).to(
            torch.bfloat16
        )
        feature_vars[model] = torch.from_numpy(np.load(osp.join(dataset_root, f"imagenet_var_{model_name}.npy"))).to(
            torch.bfloat16
        )
    return feature_means, feature_vars


def pad_shard_paths(shard_paths: list[str], num_shards: int, num_parts: int) -> list[str]:
    """Pad shard paths to be divided by number of partitions (ranks*nodes).

    Args:
        shard_paths (list[str]): pathes of dataset shards.
        num_shards (int): number of shards.
        num_parts (int): number of partitions.

    Returns:
        list[str]: shard paths padded.
    """
    final_shard_paths = shard_paths
    if num_shards % num_parts != 0:
        if num_shards < num_parts - num_shards:
            for _ in range(math.floor((num_parts - num_shards) / num_shards)):
                final_shard_paths += shard_paths[:]
            final_shard_paths += shard_paths[: num_parts - len(final_shard_paths)]
        else:
            final_shard_paths += shard_paths[: num_parts - len(final_shard_paths)]
    return final_shard_paths


def get_image_video_dataset(
    dataset_root: str,
    feature_models: list[str],
    dataset_mix: Optional[str | dict[str, float] | list] = None,
    split: str = "train",
    dataset_ratio: float = 1.0,
    image_transform: Optional[Callable[[Any], torch.Tensor]] = None,
    feature_norm: bool = False,
    seed: Optional[int | str] = 0,
    shuffle: bool = False,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[dict[str, DatasetType], float | Literal[0]]:
    """Get image and video datasets at frame level.

    Args:
        dataset_root (str): root dir of the datasets.
        feature_models (list[str]): models to load their features.
        dataset_mix (Optional[str  |  dict[str, float]  |  list], optional): how to mix the datasets.
        split (str, optional): split "train" or "val" or "test". Defaults to "train".
        dataset_ratio (float, optional): how much data use for the (combined) dataset. Defaults to 1.0.
        image_transform (Optional[Callable[[Any], torch.Tensor]], optional): image transform applied to samples.
            Defaults to None.
        feature_norm: (bool, optional): whether to normalize the feature. Defaults to False.
        seed (Optional[int  |  str], optional): seed. Defaults to 0.
        shuffle (bool, optional): shuffle or not. Defaults to False.
        world_size (int, optional): world size of DDP training. Defaults to 1.
        kwargs (Any): arguments to pass-through.

    Returns:
        tuple[dict[str, DatasetType], int]: a dict of {dataset name: dataset class}.
    """
    # read dataset mix from any acceptable form
    if isinstance(dataset_mix, str) and dataset_mix in OXE_NAMED_MIXES:
        dataset_mix = OrderedDict({k: v for k, v in OXE_NAMED_MIXES[dataset_mix]})
    elif isinstance(dataset_mix, dict):
        dataset_mix = OrderedDict(**dataset_mix)
    elif isinstance(dataset_mix, list) or isinstance(dataset_mix, omegaconf.listconfig.ListConfig):
        dataset_mix = OrderedDict({d: 1.0 for d in dataset_mix})
    else:
        raise ValueError(f"dataset_mix of {dataset_mix}:{type(dataset_mix)} is not supported.")

    if split == "eval" or split == "val":
        dataset_mix = OrderedDict({d: 1.0 for d in dataset_mix})

    # note down the dataset weights
    dataset_weights: list[float] = []
    # get frame level length
    dataset_lens: list[int] = []

    all_feature_datasets: dict[str, DatasetType] = {}

    if feature_norm:
        feature_means, feature_vars = load_feature_stats(dataset_root, feature_models)

    for d in dataset_mix:

        with open(osp.join(dataset_root, d, "splits.json"), "r") as splitf:
            dataset_len = json.load(splitf)[split]

        # if the length is 0, skip
        # this may happen for small datasets with very few shards
        if dataset_len == 0:
            continue

        path_pattern = osp.join(dataset_root, d, "images", f"*-{split}.tar")
        if "image" not in all_feature_datasets:
            all_feature_datasets["image"] = []
        shard_paths = sorted(glob.glob(path_pattern))
        num_shards = len(shard_paths)
        num_parts = world_size
        final_shard_paths = pad_shard_paths(shard_paths, num_shards, num_parts)
        ds = wds.WebDataset(
            final_shard_paths,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            detshuffle=True,
            shardshuffle=shuffle,
            seed=seed,
        ).decode(partial(decode_sample, image_transform=image_transform))
        all_feature_datasets["image"].append(ds)

        for model_name in feature_models:
            path_pattern = osp.join(dataset_root, d, f"{model_name.replace('/', '_')}", f"*-{split}.tar")
            rename_kw = {model_name: model_name.replace("/", "_").lower() + ".safetensors"}  # replace v by k

            if model_name not in all_feature_datasets:
                all_feature_datasets[model_name] = []

            shard_paths = sorted(glob.glob(path_pattern))
            num_shards = len(shard_paths)
            num_parts = world_size
            final_shard_paths = pad_shard_paths(shard_paths, num_shards, num_parts)
            if feature_norm:
                feature_transform = partial(
                    normalize_feature, mean=feature_means[model_name], std=feature_vars[model_name]
                )
            else:
                feature_transform = None
            ds = (
                wds.WebDataset(
                    final_shard_paths,
                    nodesplitter=wds.split_by_node,
                    workersplitter=wds.split_by_worker,
                    detshuffle=True,
                    shardshuffle=shuffle,
                    seed=seed,
                )
                .decode(partial(decode_sample, image_transform=image_transform, feature_transform=feature_transform))
                .rename(keep=False, **rename_kw)
            )
            all_feature_datasets[model_name].append(ds)

        dataset_weights.append(dataset_mix[d])
        dataset_lens.append(math.ceil(dataset_len * dataset_ratio))

    normalized_dataset_weights, sum_expected_lengths = normalize_ds_weights_by_ds_len(dataset_weights, dataset_lens)

    combined_feature_datasets: dict[str, Dataset] = {}
    for feature_set_name, fds in all_feature_datasets.items():
        ds = RandomMix(fds, probs=normalized_dataset_weights, stopping_strategy="all_exhausted", seed=seed)
        combined_feature_datasets[feature_set_name] = ds

    return combined_feature_datasets, sum_expected_lengths


def get_frame_dataloader(
    datasets: dict[str, DatasetType],
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1_000,
    seed: Optional[int] = 0,
    **kwargs: Any,
) -> dict[str, DataLoader]:
    """Get dataloaders of image and video datasets. Corresponding to `get_image_video_dataset()`.

    Args:
        datasets (dict[str, DatasetType]): image and video datasets from `get_image_video_dataset().
        batch_size (Optional[int], optional): batch size. Defaults to None.
        shuffle_buffer_size (int, optional): buffer for shuffle while streaming. Defaults to 1_000.

    Returns:
        dict[str, DataLoader]: dataloaders. a dict of {dataset name: dataloader}.
    """
    loaders = {}
    for k in datasets:
        loader = wds.WebLoader(datasets[k], batch_size=None, generator=default_generator, **kwargs)
        if shuffle:
            loader = loader.shuffle(shuffle_buffer_size, seed=seed)  # shuffle after mix
        loader = loader.batched(batch_size, collation_fn=default_collate)
        loaders[k] = loader
    return loaders


def get_frame_iterator(
    data_loaders: dict[str, DataLoader],
) -> Iterator[dict[str, Any]]:
    """Get iterator from image and video dataset dataloders. Corresponding to `get_frame_dataloader()`.

    Args:
        data_loaders (dict[str, DataLoader]): dataloaders from `get_frame_dataloader()`.

    Yields:
        Iterator[dict[str, Any]]: data sample.
    """
    packed_loader = data_loaders.get("packed", None)
    # place packed_loader at the first
    if packed_loader is not None:
        loaders = [packed_loader, *[data_loaders[k] for k in data_loaders if k != "packed"]]
    else:
        loaders = list(data_loaders.values())

    # merge dicts
    # this is to accommodate the old organization of datasets (each shard contains one or more columns,
    # and images are duplicated columns).
    # In new (current) dataset organization (columns are completely separated),
    # column keys are all different except some "built-in" keys added by webdataset,
    # but they are not related to any data, training, so on.
    # During transit from old to new, where two organizations exist at the same time,
    # this is to ignore extra "image" field in datasets loaded.
    for data in zip(*loaders, strict=False):
        # yield data
        for i in range(1, len(loaders)):
            for k in data[i]:
                if k not in data[0]:
                    data[0][k] = data[i][k]
        yield data[0]
