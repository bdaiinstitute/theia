# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import math
import os
import tarfile
from collections import OrderedDict

from theia.dataset.oxe.oxe_common import ALL_OXE_DATASETS
from theia.dataset.video import ALL_VIDEO_DATASETS

DATASET_RATIOS = OrderedDict({"train": 0.8, "val": 0.05, "test": 0.15})

all_datasets = {}
all_datasets.update(ALL_OXE_DATASETS)
all_datasets.update(ALL_VIDEO_DATASETS)
# all_datasets.update(ALL_IMAGE_DATASETS) imagenet has its own splits, can be done seperately


def count_steps(tar_path: str) -> int:
    """Count how many samples are in the shard

    Args:
        tar_path (str): path to the shard
    """
    with tarfile.open(tar_path) as tarf:
        return len(list(set([x.name.split(".")[0] for x in tarf.getmembers()])))


def do_dataset_split(args: argparse.Namespace, dataset_name: str) -> None:
    """Split the dataset given a dataset name.
    The dataset will be split based on shards in the lexical order of their filenames.
    The first part goes to `training` set, the second part goes to `validation` set,
        and the last part goes to `test` set.

    Args:
        dataset_name (str): name of the dataset.
    """
    dataset_dir = os.path.join(args.dataset_root, dataset_name)
    split_json_file = os.path.join(dataset_dir, "splits.json")

    if os.path.exists(split_json_file):
        return

    # only apply to images
    # then feature extraction script will handle splits for features
    shard_dirs = [os.path.join(dataset_dir, "images")]
    for shard_dir in shard_dirs:
        shard_names = sorted(
            [filename for filename in os.listdir(shard_dir) if filename.endswith(".tar") and "-" in filename]
        )
        n_shards = len(shard_names)
        print(f"{dataset_name} total {n_shards} shards")

        cum_n_shards = 0
        split_steps_count = {}
        for _, split in enumerate(DATASET_RATIOS):
            ratio = DATASET_RATIOS[split]
            split_n_shards = math.ceil(n_shards * ratio)
            split_steps_count[split] = 0
            print(f"{dataset_name} {split} {split_n_shards} shards")

            for shard_idx in range(cum_n_shards, min(cum_n_shards + split_n_shards, n_shards)):
                original_path = os.path.join(shard_dir, shard_names[shard_idx])
                if shard_idx == n_shards - 1:
                    split_steps_count[split] += count_steps(original_path)
                else:
                    split_steps_count[split] += args.samples_per_shard
                split_shard_filename = shard_names[shard_idx].replace(".tar", f"-{split}.tar")
                split_shard_path = os.path.join(shard_dir, split_shard_filename)

                if not args.dry_run:
                    os.rename(original_path, split_shard_path)
            cum_n_shards += split_n_shards

        with open(os.path.join(dataset_dir, "splits.json"), "w") as f:
            json.dump(split_steps_count, f, indent=4)
        print(split_steps_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Number of samples per webdataset shard. Rarely changed. Replace with your actual setting.",
    )
    args = parser.parse_args()
    for dataset in all_datasets:
        if dataset in ALL_OXE_DATASETS:
            if "_sim" in dataset:
                continue
            if "uiuc_d3field" in dataset or "cmu_playing_with_food" in dataset or "robot_vqa" in dataset:
                continue
        do_dataset_split(args, dataset)


if __name__ == "__main__":
    main()
