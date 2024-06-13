#!/bin/bash
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

torchrun --nproc_per_node=1 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 scripts/train/train_rvfm.py \
  +logging.note=sanitycheck +dataset.data_portion=0.001
