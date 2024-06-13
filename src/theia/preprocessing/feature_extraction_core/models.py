# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn.functional import interpolate
from theia.foundation_models import (
    get_clip_feature,
    get_clip_model,
    get_depth_anything_feature,
    get_depth_anything_model,
    get_dinov2_feature,
    get_dinov2_model,
    get_llava_vision_model,
    get_llava_visual_feature,
    get_sam_feature,
    get_sam_model,
    get_vit_feature,
    get_vit_model,
)


def get_model(model_name: str, device: int | str | torch.device = "cpu") -> tuple[nn.Module, Any]:
    if "google/vit" in model_name:
        model, processor = get_vit_model(model_name, device=device)
    elif "facebook/sam" in model_name:
        model, processor = get_sam_model(model_name, device=device, with_upscaled=False)
    elif "openai/clip" in model_name:
        model, processor = get_clip_model(model_name, device=device)
    elif "facebook/dinov2" in model_name:
        model, processor = get_dinov2_model(model_name, device=device)
    elif "llava" in model_name:
        model, processor = get_llava_vision_model(model_name, device=device)
    elif "depth-anything" in model_name:
        model, processor = get_depth_anything_model(model_name, device=device, selected_feature="head")
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    return model, processor


def get_models(
    model_names: list[str], device: int | str | torch.device = "cpu"
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    models: dict[str, nn.Module] = {}
    processors: dict[str, Any] = {}
    for model_name in model_names:
        model, processor = get_model(model_name, device)
        models[model_name.replace("/", "_")] = model
        processors[model_name.replace("/", "_")] = processor
    return models, processors


def get_feature_outputs(
    model_name: str, model: nn.Module, processor: Any, batch_images: list[NDArray], dtype: torch.dtype = torch.bfloat16
) -> dict[str, dict[str, torch.Tensor]]:
    features: dict[str, dict[str, torch.Tensor]] = {model_name: {}}
    if "google_vit" in model_name:
        cls_token, feature = get_vit_feature(model, processor, batch_images)
        features[model_name] = {
            "cls_token": cls_token.detach().cpu().to(dtype).contiguous(),
            "embedding": feature.detach().cpu().to(dtype).contiguous()
        }
    elif "facebook_sam" in model_name:
        feature, upscaled_feature = get_sam_feature(model, processor, batch_images)
        features[model_name] = {"embedding": feature.detach().cpu().to(dtype).contiguous()}
        features[model_name + "_32"] = {
            "embedding": interpolate(feature, (32, 32)).detach().cpu().to(dtype).contiguous()
        }

        if upscaled_feature:
            features[model_name]["upscaled_embedding"] = upscaled_feature.detach().cpu().to(dtype).contiguous()
    elif "openai_clip" in model_name:
        cls_token, visual_tokens, pooled_cls_token = get_clip_feature(model, processor, batch_images)
        features[model_name] = {
            "embedding": visual_tokens.detach().cpu().to(dtype).contiguous(),
            "cls_token": cls_token.detach().cpu().to(dtype).contiguous(),
            "pooled_cls_token": pooled_cls_token.detach().cpu().to(dtype).contiguous(),
        }
    elif "facebook_dinov2" in model_name:
        cls_token, visual_tokens, pooled_cls_token = get_dinov2_feature(model, processor, batch_images)
        features[model_name] = {
            "embedding": visual_tokens.detach().cpu().to(dtype).contiguous(),
            "cls_token": cls_token.detach().cpu().to(dtype).contiguous(),
            "pooled_cls_token": pooled_cls_token.detach().cpu().to(dtype).contiguous(),
        }
    elif "llava" in model_name:
        feature = get_llava_visual_feature(model, processor, batch_images)
        features[model_name] = {"embedding": feature.detach().cpu().to(dtype).contiguous()}
    elif "depth-anything" in model_name:
        feature = get_depth_anything_feature(model, processor, batch_images)
        features[model_name] = {"embedding": interpolate(feature, (64, 64)).detach().cpu().to(dtype).contiguous()}
    else:
        raise NotImplementedError(f"model {model_name} is not supported")

    return features
