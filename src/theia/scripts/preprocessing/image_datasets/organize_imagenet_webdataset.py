# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Organize imagefolder-like images (ImageNet) to webdataset format."""

import argparse
import glob
import os
import shutil
import tarfile
from io import BytesIO

import numpy as np
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize


def check_existing_shard(path: str) -> bool:
    """Check the integrity of the existing webdataset shard.

    Args:
        path (str): path to the webdataset shard.

    Returns:
        bool: True for complete shard.
            False for non-existing or broken shard.
    """
    try:
        tarf = tarfile.open(path)
        for _ in tarf.getmembers():
            pass
    except (ValueError, tarfile.ReadError, tarfile.CompressionError) as e:
        print(e)
        return False
    return True


def create_shard(
    args: argparse.Namespace,
    shard_idx: int,
    shard_path: str | None,
    remote_shard_path: str,
    frames: list[tuple[NDArray, str]],
) -> None:
    """Create a webdataset shard.

    Args:
        args (argparse.Namespace): arguments.
        shard_idx (int): index of this shard.
        shard_path (str): (local) path to save the shard.
        remote_shard_path (str): final destination (remote) to save the shard.
        frames (list[tuple[NDArray, str]]): images to save in this shard.
    """
    if check_existing_shard(remote_shard_path):
        print(f"creating {args.dataset} shard {shard_idx:06d} - check pass, skip\r", end="")
        return
    print(f"creating {args.dataset} shard {shard_idx:06d}\r", end="")
    if shard_path is None:
        shard_path = remote_shard_path
    with wds.TarWriter(shard_path) as tar_writer:
        for i, (image, basename) in enumerate(frames):
            image_out = BytesIO()
            np.save(image_out, image)
            sample = {"__key__": basename, "image": image_out.getvalue()}
            tar_writer.write(sample)
            if (i + 1) % 20 == 0:
                print(f"creating {args.dataset} shard {shard_idx:06d} - {(i+1) * 100 // len(frames):02d}%\r", end="")
    if shard_path != remote_shard_path:
        shutil.move(shard_path, remote_shard_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--imagenet-raw-path", type=str)
    parser.add_argument("--tmp-shard-path", type=str, default="None")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    args = parser.parse_args()

    match args.dataset:
        case "imagenet":
            IMAGE_DATASET_RAW_DIR = args.imagenet_raw_path
        case _:
            raise NotImplementedError(f"{args.dataset} is not supported")

    if args.tmp_shard_path == "None":
        TMP_SHARD_PATH = None
    else:
        TMP_SHARD_PATH = os.path.join(args.tmp_shard_path, args.dataset)
        if not os.path.exists(TMP_SHARD_PATH):
            os.makedirs(TMP_SHARD_PATH)

    OUTPUT_SHARD_PATH = os.path.join(args.output_path, args.dataset)
    if not os.path.exists(OUTPUT_SHARD_PATH):
        os.makedirs(OUTPUT_SHARD_PATH, exist_ok=True)

    if args.split == "train":
        image_paths = sorted(glob.glob(f"{IMAGE_DATASET_RAW_DIR}/{args.split}/*/*.JPEG"))
    else:
        image_paths = sorted(glob.glob(f"{IMAGE_DATASET_RAW_DIR}/{args.split}/*.JPEG"))

    transform = Compose([Resize((224, 224), antialias=True)])

    shard_idx = 0
    shard_buffer: list[tuple[NDArray, str]] = []
    for image_path in image_paths:
        basename = image_path.split("/")[-1].split(".")[0]
        image = np.array(transform(Image.open(image_path)))
        shard_buffer.append((image, basename))
        if len(shard_buffer) % 20 == 0:
            print(f"shard {shard_idx: 04d} frames {len(shard_buffer)}\r", end="")
        if len(shard_buffer) == args.samples_per_shard:
            shard_fn = f"{args.dataset}_{args.split}-{shard_idx:06d}-{args.split}.tar"
            local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
            remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
            create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)
            shard_buffer = []
            shard_idx += 1

    shard_fn = f"{args.dataset}_{args.split}-{shard_idx:06d}-{args.split}.tar"
    local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
    remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
    if len(shard_buffer) > 0:
        create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)


if __name__ == "__main__":
    main()
