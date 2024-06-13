# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import os
import tarfile
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from safetensors.torch import load as sft_load

from theia.dataset import ALL_IMAGE_DATASETS, ALL_VIDEO_DATASETS
from theia.foundation_models.common import MODELS
from theia.preprocessing.feature_extraction_core import (
    get_feature_outputs,
    get_model,
)
from theia.utils.seed import seed_everything


def decode_oxe_sample(data: bytes, data_type: str) -> Any:
    """Decode the sample from bytes.

    Args:
        data (bytes): data to be decoded.
        data_type (str): the type of the data.
            Usually is part of the key (filename of the sample) in the webdataset.

    Returns:
        Any: decoded data or pass-through bytes without touch.
    """
    if ".safetensors" in data_type:
        sftensor = sft_load(data)
        return sftensor["embedding"]
    elif data_type == ".image":
        image = np.load(BytesIO(data))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # return torch.from_numpy(image)
        return image
    else:
        return data


def get_tar_sample(tarf: tarfile.TarFile, sample_index: int) -> bytes:
    """Get bytes of a sample with index `sample_index` in tarfile `tarf`.

    Args:
        tarf (tarfile.TarFile): tar file.
        sample_index (int): index of the sample

    Returns:
        bytes: content of the sample in bytes
    """
    tar_members = tarf.getmembers()
    tar_members = sorted(tar_members, key=lambda x: x.name)
    tar_mem = tar_members[sample_index]
    f = tarf.extractfile(tar_mem.name)
    if f:
        return f.read()
    else:
        raise IOError(f"failed to read tarfile {tarf}.")


def get_tar_sample_name(tarf: tarfile.TarFile, sample_index: int) -> str:
    """Get the name of the sample with index `sample_index` in the tarfile `tarf`.

    Args:
        tarf (tarfile.TarFile): tar file.
        sample_index (int): index of the sample

    Returns:
        str: name of the file
    """
    tar_members = tarf.getmembers()
    tar_members = sorted(tar_members, key=lambda x: x.name)
    tar_mem = tar_members[sample_index]
    return tar_mem.name


def check_feature(
    args: argparse.Namespace,
    dataset: str,
    modelnames_to_check: list[str],
    models: dict[str, Any],
    processors: dict[str, Any],
    shard_idx: int,
    sample_indices: list[int] | NDArray,
    split: str = "train",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, bool]:
    """Check feature consistency given a dataset, names of models to check,
        shard index and sample indices within that shard.

    Args:
        args (argparse.Namespace): arguments.
        dataset (str): name of the dataset
        modelnames_to_check (list[str]): names of the features (models) to check.
        models (dict[str, Any]): original models to produce features on the fly.
        processors (dict[str, Any]): original processor of the models.
        shard_idx (int): index of the shard.
        sample_indices (list[int] | NDArray): indices of samples to be checked.
        split (str, optional): name of the split of the dataset. Defaults to "train".
        dtype (torch.dtype, optional): dtype of the generated feature. Defaults to torch.bfloat16.

    Returns:
        dict[str, bool]: check result. The keys are model names. True means passing the check.
    """
    data_dir = os.path.join(args.dataset_root, dataset, "images")
    shard_filenames = sorted([filename for filename in os.listdir(data_dir) if f"{split}.tar" in filename])
    image_tar = tarfile.open(os.path.join(data_dir, shard_filenames[shard_idx]), "r")
    images = [
        decode_oxe_sample(get_tar_sample(image_tar, sample_index), data_type=".image")
        for sample_index in sample_indices
    ]
    for image, sample_index in zip(images, sample_indices, strict=False):
        if args.save_image:
            if not os.path.exists(args.image_save_dir):
                os.makedirs(args.image_save_dir)
            image = Image.fromarray(image)
            image.save(os.path.join(args.image_save_dir, f"image_{shard_idx}_{sample_index}.jpg"))
    image_names = [get_tar_sample_name(image_tar, sample_index).split(".")[0] for sample_index in sample_indices]

    model_check_pass = {m: False for m in modelnames_to_check}
    for model_name in modelnames_to_check:
        legit_model = model_name.replace("/", "_")
        data_dir = os.path.join(args.dataset_root, dataset, legit_model)
        shard_filenames = sorted([filename for filename in os.listdir(data_dir) if f"{split}.tar" in filename])
        feature_tar = tarfile.open(os.path.join(data_dir, shard_filenames[shard_idx]), "r")
        features = torch.stack(
            [
                decode_oxe_sample(get_tar_sample(feature_tar, sample_index), data_type=".safetensors")
                for sample_index in sample_indices
            ]
        )
        gt_features = get_feature_outputs(
            legit_model, models[legit_model], processors[legit_model], images, dtype=dtype
        )[legit_model]["embedding"]
        print(torch.sum(torch.abs(features - gt_features)), torch.max(torch.abs(features - gt_features)))
        model_check_pass[model_name] = torch.all((features - gt_features) == 0)
        if args.check_feature_name:
            names = [get_tar_sample_name(feature_tar, sample_index).split(".")[0] for sample_index in sample_indices]
            model_check_pass[model_name] = (
                all([imname == filename for imname, filename in zip(image_names, names, strict=False)])
                and model_check_pass[model_name]
            )
    return model_check_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--samples-per-shard", type=int, default=1000, help="number of samples per webdataset shard.")
    parser.add_argument("--check-feature-name", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--image-save-dir", type=str, default="./tmp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(0)

    all_datasets = {}
    all_datasets.update(ALL_IMAGE_DATASETS)
    all_datasets.update(ALL_VIDEO_DATASETS)

    with open(os.path.join(args.dataset_root, args.dataset, "splits.json"), "r") as f:
        dataset_len = json.load(f)[args.split]

    n_shards = dataset_len // args.samples_per_shard

    model_names = [model_name for model_name in MODELS if "llava" not in model_name]
    models, processors = {}, {}
    for model_name in model_names:
        legit_model_name = model_name.replace("/", "_")
        model, processor = get_model(model_name, device=0)
        models[legit_model_name] = model
        processors[legit_model_name] = processor

    shard_indices = np.random.permutation(n_shards)[:5]
    print(f"randomly check {args.dataset} shards {shard_indices}")
    model_check_pass: dict[str, list[bool]] = {model_name: [] for model_name in model_names}
    for shard_idx in shard_indices:
        sample_indices = np.random.permutation(1000)[:8]
        print(f"randomly check {args.dataset} shard {shard_idx} sample_indices {sample_indices}")
        check_result = check_feature(
            args, args.dataset, model_names, models, processors, shard_idx, sample_indices, split=args.split
        )
        for model_name in model_check_pass:
            model_check_pass[model_name].append(check_result[model_name])
    for model_name in model_check_pass:
        if not all(model_check_pass[model_name]):
            print(f"{args.dataset} {args.split} {model_name} check failed!!!")


if __name__ == "__main__":
    main()
