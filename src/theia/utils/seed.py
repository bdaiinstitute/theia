# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import random
from typing import Any, Optional

import numpy as np
import torch

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[Any] = None, workers: bool = False) -> int:
    """Seed everything adopted from lightning_fabric.utilities.seed.seed_everything.

    Avoid using lightning only for seeding.

    Args:
        seed (Optional[Any]): seed, preferably an integer, or other stuff can be converted to an integer.

    Returns:
        int: the actual seed used. It should be the same as input seed in most of the cases.
    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = 0

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHON_SEED"] = str(seed)  # add python seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
