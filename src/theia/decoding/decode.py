# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from numpy.typing import NDArray
from PIL import Image
from sklearn.decomposition import PCA
from transformers import SamModel, SamProcessor
from transformers.pipelines import MaskGenerationPipeline

from theia.decoding.depth_anything import decode_depth_anything
from theia.decoding.dinov2 import decode_dinov2
from theia.decoding.sam import decode_sam
from theia.preprocessing.feature_extraction_core import (
    get_feature_outputs,
    get_model,
)


def denormalize_feature(
    x: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Denormalize the features using mean and std.

    Args:
        x (torch.Tensor): features to be denomalized.
        mean (Optional[torch.Tensor], optional): mean value of the features. Defaults to None
        std (Optional[torch.Tensor], optional): std value of the features. Defaults to None.

    Returns:
        torch.Tensor: denormalized features.
    """
    if mean is None and std is None:
        return x
    elif mean is None and std is not None:
        return x * std
    elif mean is not None and std is None:
        return x + mean
    return x * std + mean


def load_feature_stats(
    feature_models: list[str], stat_file_root: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load the statistics (mean and variance) of the features, per model.

    Args:
        feature_models (list[str]): names of the models. Note: there are `/` in the name.
        stat_file_root (str): directory that holds feature stat files.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: means and variance.
    """
    feature_means: dict[str, torch.Tensor] = {}
    feature_vars: dict[str, torch.Tensor] = {}
    for model in feature_models:
        model_name = model.replace("/", "_")
        feature_means[model] = torch.from_numpy(
            np.load(os.path.join(stat_file_root, f"imagenet_mean_{model_name}.npy"))
        )
        feature_vars[model] = torch.from_numpy(np.load(os.path.join(stat_file_root, f"imagenet_var_{model_name}.npy")))
    return feature_means, feature_vars


def decode_everything(
    theia_model: nn.Module,
    feature_means: dict[str, torch.Tensor],
    feature_vars: dict[str, torch.Tensor],
    images: list[Image.Image],
    mask_generator: MaskGenerationPipeline,
    sam_model: SamModel,
    depth_anything_decoder: nn.Module,
    pred_iou_thresh: float = 0.9,
    stability_score_thresh: float = 0.9,
    gt: bool = False,
    pca: Optional[PCA] = None,
    device: int | str | torch.device = 0,
) -> tuple[list[NDArray], Optional[list[NDArray]]]:
    """Decode features from given `theia_model` into different outputs corresponding to upstream models including
        DINOv2, Sam, and Depth-Anything.

    Args:
        theia_model (nn.Module): theia model.
        feature_means (dict[str, torch.Tensor]): means of the features for denormalization.
        feature_vars (dict[str, torch.Tensor]): variance of the features for denormalization.
        images (list[Image.Image]): input images.
        mask_generator (MaskGenerationPipeline): mask generation pipeline.
        sam_model (SamModel): sam model.
        depth_anything_decoder (nn.Module): depth anything decoder.
        pred_iou_thresh (float, optional): iou threshold for mask generation.
            See transformers.pipelines.MaskGenerationPipeline for more details. Defaults to 0.9.
        stability_score_thresh (float, optional): stability score threshold for mask generation.
            See transformers.pipelines.MaskGenerationPipeline for more details. Defaults to 0.9.
        gt (bool): whether to attach ground truth result in the visualization. Defaults to False.
        pca (Optional[PCA]): pca for DINOv2 decoding. If provided, will use this pca particular. Defaults to None.
        device (int | str | torch.device, optional): device for decoding. Defaults to 0.

    Returns:
        tuple[list[NDArray], Optional[list[NDArray]]]: decoding results from given model,
            and ground truth (if `gt=True`).
    """
    features: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for im in images:
            feature = theia_model([im])
            if len(features) == 0:
                features = {k: [] for k in feature}
            for k in feature:
                features[k].append(feature[k].detach().cpu())
    for k in features:
        features[k] = torch.cat(features[k], dim=0)
    for m in features:
        features[m] = denormalize_feature(features[m], feature_means[m], feature_vars[m])

    dino_model_name = "facebook/dinov2-large"
    sam_model_name = "facebook/sam-vit-huge"
    depth_anything_model_name = "LiheYoung/depth-anything-large-hf"

    pca = None
    # gt
    gt_decode_results = None
    if gt:
        def legit_model_name(model_name: str) -> str:
            return model_name.replace("/", "_")

        dino_model, dino_processor = get_model(dino_model_name, device=device)
        dino_gt_feature = []
        for im in images:
            dino_gt_feature.append(
                get_feature_outputs(
                    legit_model_name(dino_model_name), dino_model, dino_processor, [im], dtype=torch.float
                )[legit_model_name(dino_model_name)]["embedding"]
                .detach()
                .cpu()
            )
        dino_gt_feature = torch.cat(dino_gt_feature, dim=0)
        dino_gt_feature = rearrange(dino_gt_feature, "b c h w -> b (h w) c")
        dino_gt_dec, pca = decode_dinov2(dino_gt_feature, pca=pca)
        sam_processor = SamProcessor.from_pretrained(sam_model_name)
        sam_gt_feature = []
        for im in images:
            sam_inputs = sam_processor(images=[im], return_tensors="pt").to(device)
            with torch.no_grad():
                sam_gt_feature.append(sam_model.get_image_embeddings(sam_inputs["pixel_values"]).detach().cpu())
        sam_gt_feature = torch.cat(sam_gt_feature, dim=0)
        sam_gt_feature = rearrange(sam_gt_feature, "b c h w -> b (h w) c")
        sam_gt_dec = decode_sam(
            sam_gt_feature, images, mask_generator, pred_iou_thresh=0.9, stability_score_thresh=0.9, device=device
        )
        depth_anything_model, depth_anything_processor = get_model(depth_anything_model_name, device=device)
        depth_anything_gt_feature = []
        for im in images:
            depth_anything_gt_feature.append(
                get_feature_outputs(
                    legit_model_name(depth_anything_model_name),
                    depth_anything_model,
                    depth_anything_processor,
                    [im],
                    dtype=torch.float,
                )[legit_model_name(depth_anything_model_name)]["embedding"]
                .detach()
                .cpu()
            )
        depth_anything_gt_feature = torch.cat(depth_anything_gt_feature, dim=0)
        depth_anything_gt_feature = rearrange(depth_anything_gt_feature, "b c h w -> b (h w) c")
        depth_gt_dec = decode_depth_anything(depth_anything_gt_feature, depth_anything_decoder, device=device)

        gt_decode_results = [
            np.hstack([np.array(images[i]).astype(np.float32) / 255.0, dino_gt_dec[i], sam_gt_dec[i], depth_gt_dec[i]])
            for i in range(len(images))
        ]

    dino_dec, _ = decode_dinov2(features[dino_model_name], pca=pca)
        
    try:
        sam_dec = decode_sam(
            features[sam_model_name],
            images,
            mask_generator,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            device=device,
        )
    except IndexError:
        sam_dec = np.zeros_like(dino_dec)
    depth_dec = decode_depth_anything(features[depth_anything_model_name], depth_anything_decoder, device=device)

    theia_decode_results = [
        np.hstack([np.array(images[i]).astype(np.float32) / 255.0, dino_dec[i], sam_dec[i], depth_dec[i]])
        for i in range(len(images))
    ]

    return theia_decode_results, gt_decode_results
