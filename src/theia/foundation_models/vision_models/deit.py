# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def get_deit_feature(
    model: AutoModel, processor: AutoImageProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> torch.Tensor:
    """Get feature from DeiT model.

    Args:
        model (AutoModel): DeiT model.
        processor (AutoImageProcessor): DeiT input processor.
        images (list[np.ndarray]): images to be encoded.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        torch.Tensor: feature from last layer, (1, 768, 14, 14) BCHW deit-base
    """
    inputs = processor(images, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state[:, 1:]
    batch_size, num_patches, num_channels = last_hidden_state.size()
    last_hidden_state = last_hidden_state.transpose(1, 2)
    last_hidden_state = last_hidden_state.reshape(
        batch_size, num_channels, int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
    )
    return last_hidden_state  # (1, 768, 14, 14) BCHW for deit-base


def get_deit_model(
    model_name: str = "facebook/deit-tiny-patch16-224", device: str | torch.device = "cuda"
) -> tuple[AutoModel, AutoImageProcessor]:
    """Get DeiT model and its corresponding input processor.

    Args:
        model_name (str, optional): the name of DeiT model. Defaults to "facebook/deit-tiny-patch16-224".
        device (str | torch.device, optional): device to put model on. Defaults to "cuda".

    Returns:
        tuple[DeiTModel, AutoImageProcessor]: DeiT model and its processor.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, processor


def print_feature_size(model_name: str = "facebook/deit-tiny-patch16-224") -> None:
    """Print the size of the feature from ViT.

    Args:
        model_name (str, optional): the name of ViT model. Defaults to "facebook/deit-tiny-patch16-224".
    """
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = np.array(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_deit_model(model_name=model_name, device=device)
    feature = get_deit_feature(model, processor, image)
    print(feature.size())
    # (1, 768, 14, 14) BCHW for deit-base
