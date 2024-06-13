# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Calculate the channel-wise mean and var of extracted features on ImageNet dataset.
The resulting mean and var will be used in distillation process.
"""

import argparse
import glob
import os
from io import BytesIO

import numpy as np
import torch
import webdataset as wds
from einops import rearrange
from safetensors.torch import load as sft_load
from torch.utils.data import default_collate


def decode_dataset_sample(key: str, data: bytes) -> bytes | torch.Tensor:
    """
    Decode a feature / column in webdataset sample in bytes to its original format.

    Args:
        key (str): name of the feature / column.
        data (bytes): data in bytes.

    Returns:
        bytes | torch.Tensor: decoded feature.
    """
    if ".safetensors" in key:
        sft = sft_load(data)
        return rearrange(sft["embedding"], "c h w -> (h w) c")
    elif key == ".image":
        return torch.from_numpy(np.load(BytesIO(data)))
    else:
        return data


def main() -> None:
    """Entry point of this script for calculating mean and var."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    all_datasets = {}
    all_datasets.update({"imagenet": {"steps": 1_281_167}})
    ds_dir = args.dataset_path
    models = [m for m in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, m))]
    for model in models:
        print(model)
        if model == "images" or model == "image" or model == "images_val":
            continue
        if os.path.exists(f"{args.output_path}/imagenet_mean_{model}.npy"):
            continue
        model_mean: torch.Tensor = None
        model_var_sum: torch.Tensor = None
        n = 0
        ds = (
            wds.WebDataset(
                sorted(glob.glob(f"{ds_dir}/{model}/*.tar")),
                shardshuffle=False,
            )
            .decode(decode_dataset_sample)
            .batched(256, collation_fn=default_collate)
        )

        key = f"{model}.safetensors".lower()
        for batch_idx, batch in enumerate(ds):
            if model_mean is None:
                model_mean = torch.zeros((batch[key].size(-1)))
            new_n = np.prod(batch[key].size()[:2])
            batch_mean = batch[key].float().mean((0, 1))
            model_mean = (model_mean * n + batch_mean * new_n) / (n + new_n)
            n += new_n
            print(f"calc {model} mean {batch_idx*256:07d}\r", end="")

        model_mean_npy = model_mean.numpy()
        np.save(f"{args.output_path}/imagenet_mean_{model}.npy", model_mean_npy)

        # var
        for i, b in enumerate(ds):
            if model_var_sum is None:
                model_var_sum = torch.zeros((b[key].size(-1)))
            model_var_sum += ((b[key].float() - model_mean) ** 2).sum((0, 1))
            print(f"calc {model} var {i*256:07d}\r", end="")

        model_var = torch.sqrt(model_var_sum / (n - 1))
        np.save(f"{args.output_path}/imagenet_var_{model}.npy", model_var.numpy())


if __name__ == "__main__":
    main()
