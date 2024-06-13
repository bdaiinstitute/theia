# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#! /bin/bash

dataset=$1
numgpus=$2

# modify models below
models=(facebook/dinov2-large google/vit-huge-patch14-224-in21k openai/clip-vit-large-patch14 LiheYoung/depth-anything-large-hf) # facebook/sam-vit-huge
for model in ${models[@]}
do
    (
        python feature_extraction.py --dataset $dataset --output-path /storage/nfs/datasets/jshang/ --model $model --split train --num-gpus $numgpus; \
        python feature_extraction.py --dataset $dataset --output-path /storage/nfs/datasets/jshang/ --model $model --split val --num-gpus $numgpus
    ) &
done
wait
