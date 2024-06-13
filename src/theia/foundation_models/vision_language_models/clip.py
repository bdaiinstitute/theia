# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModel


def get_clip_feature(
    model: CLIPVisionModel, processor: AutoProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get features from the visual encoder of CLIP.

    Args:
        model (CLIPVisionModel): CLIP model.
        processor (AutoProcessor): CLIP input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: features from clip (
            cls_token:      last layer embedding from cls token         # (1, 1, 1024) if vit-large,
            visual_tokens:  last layer embeddings from image            # (1, 1024, 16, 16) BCHW if vit-large,
            pooled_cls_token: last layer embedding from cls + layernorm # (1, 1, 1024) if vit-large
        )
    """
    inputs = processor(images=images, return_tensors="pt").to(model.device)
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


def get_clip_model(
    model_name: str = "openai/clip-vit-large-patch14", device: str | torch.device = "cuda"
) -> tuple[CLIPVisionModel, AutoProcessor]:
    """Get CLIP model and its input processor.

    Args:
        model_name (str, optional): name of CLIP model. Defaults to "openai/clip-vit-large-patch14".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[CLIPVisionModel, AutoProcessor]: CLIP model and the correponding input processor.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device)
    return model, processor


def print_feature_size(model_name: str = "openai/clip-vit-large-patch14") -> None:
    """Print the sizes of features from CLIP.

    Args:
        model_name (str, optional): the name of CLIP model. Defaults to "openai/clip-vit-large-patch14".
    """
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_clip_model(model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_clip_feature(model, processor, image)

    print(model_name, cls_token.size(), visual_tokens.size(), pooled_cls_token.size())


if __name__ == "__main__":
    print_feature_size()
