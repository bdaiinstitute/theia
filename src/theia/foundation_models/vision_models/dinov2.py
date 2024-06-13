# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, Dinov2Model


def get_dinov2_feature(
    model: Dinov2Model, processor: AutoImageProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get DINOv2 features.

    Args:
        model (Dinov2Model): DINOv2 model.
        processor (AutoImageProcessor): DINOv2 input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (
            cls_token:      last layer embedding from cls token # (1, 1, 1024) if dinov2-large,
            visual_tokens:  last layer embeddings from image # (1, 1024, 16, 16) BCHW if dinov2-large,
            pooled_cls_token: last layer embedding from cls + layernorm # (1, 1, 1024) if dinov2-large
        )
    """
    inputs = processor(images, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    cls_token = outputs.last_hidden_state[:, :1]  # (1, 1, 1024) if vit-large
    visual_tokens = outputs.last_hidden_state[:, 1:]  # (1, 256, 1024) if vit-large
    pooled_cls_token = outputs.pooler_output.unsqueeze(1)  # (1, 1, 1024) if vit-large
    batch_size, num_patches, num_channels = visual_tokens.size()
    visual_tokens = visual_tokens.transpose(1, 2)
    visual_tokens = visual_tokens.reshape(
        batch_size, num_channels, int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
    )  # (1, 1024, 16, 16) BCHW for vit-huge
    return cls_token, visual_tokens, pooled_cls_token


def get_dinov2_model(
    model_name: str = "facebook/dinov2-large", device: str | torch.device = "cuda"
) -> tuple[Dinov2Model, AutoImageProcessor]:
    """Get DINOv2 model and its input processor.

    Args:
        model_name (str, optional): name of DINOv2 model. Defaults to "facebook/dinov2-large".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[Dinov2Model, AutoImageProcessor]: DINOv2 model and the corresponding input processor
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    return model, processor


def print_feature_size(model_name: str = "facebook/dinov2-large") -> None:
    """Print the sizes of features from DINOv2.

    Args:
        model_name (str, optional): the name of DINOv2. Defaults to "facebook/dinov2-large".
    """
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = [np.array(image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_dinov2_model(model_name=model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_dinov2_feature(model, processor, image)
    print(cls_token.size(), visual_tokens.size(), pooled_cls_token.size())
    # (1, 1, 1024), (1, 1024, 16, 16), (1, 1, 1024) for dinov2-large
