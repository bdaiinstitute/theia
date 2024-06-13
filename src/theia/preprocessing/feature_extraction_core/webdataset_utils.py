# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import tarfile
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray


def check_existing_shard(path: str, keys: list[str]) -> tuple[int, dict]:
    """
    Check the integrity of a shard given path.

    Returns:
        tuple[int, dict]:
            code (int): 1 the file is ok, 0 not
            count_per_key (dict): if the file is ok, how many samples are generated per key
    """
    count_per_key = {k: 0 for k in keys}
    if os.path.exists(path):
        try:
            with tarfile.open(path, "r") as tarf:
                tar_members = tarf.getmembers()
                tar_members = sorted(tar_members, key=lambda x: x.name)
                for tar_mem in tar_members:
                    for k in keys:
                        if k in tar_mem.name:
                            count_per_key[k] += 1
            return 1, count_per_key
        except tarfile.TarError:
            return 0, count_per_key
    else:
        return 0, count_per_key


def read_shard(path: str) -> dict[str, bytes]:
    """Read a (half) processed tar shard and store file contents in bytes.

    The tar should be complete to read.

    Args:
        path (str): path to the tar file.

    Returns:
        dict[str, bytes]: tarfile content in a dictionary where key is the tarinfo.name of each member
    """
    samples = {}
    with tarfile.open(path, "r") as tarf:
        tar_members = tarf.getmembers()
        tar_members = sorted(tar_members, key=lambda x: x.name)
        for tar_mem in tar_members:
            f = tarf.extractfile(tar_mem.name)
            if f:
                samples[tar_mem.name] = f.read()
    return samples


def decode_image_npy_only(key: str, data: bytes) -> NDArray | bytes:
    if "image" in key:
        image = np.load(BytesIO(data))
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            return image
    else:
        return data
