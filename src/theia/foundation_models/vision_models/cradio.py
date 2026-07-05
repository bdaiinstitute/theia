# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""C-RADIO teacher for Theia distillation.

C-RADIO (NVIDIA, https://github.com/NVlabs/RADIO) is an agglomerative vision
foundation model (multi-teacher distillation). The "C-RADIO" weights are
released under the NVIDIA Open Model License, which permits commercial use, so
they are usable as a default Theia teacher.

The public C-RADIO checkpoints expose a HuggingFace `AutoModel` with
`trust_remote_code=True`. A forward pass returns a `(summary, spatial_features)`
pair:
    summary          : (B, C_s)      global/CLS-like token
    spatial_features : (B, N, C)     per-patch tokens, N = (H/patch) * (W/patch)

C-RADIO applies its own ImageNet input conditioning internally, so inputs are
passed as float pixel values in [0, 1].
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

# C-RADIO ViT backbones use patch size 16, so the input resolution must be a
# multiple of 16. 224 matches the other Theia teachers (224/16 = 14 -> 14x14 =
# 196 spatial tokens); for C-RADIOv3-H this yields (1280, 14, 14) features.
CRADIO_INPUT_RESOLUTION = 224


def get_cradio_feature(
    model: torch.nn.Module,
    processor: None,
    images: list[np.ndarray],
    requires_grad: bool = False,
    resolution: int = CRADIO_INPUT_RESOLUTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get C-RADIO features.

    Args:
        model (torch.nn.Module): C-RADIO model (HF AutoModel, trust_remote_code).
        processor (None): unused; C-RADIO conditions its own inputs. Kept for a
            uniform teacher signature with the other foundation models.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8, HWC.
        requires_grad (bool): maintains gradient. Defaults to False.
        resolution (int): square resolution to resize inputs to. Must be a
            multiple of the backbone patch size (16).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (
            summary_token:  (B, 1, C_s) global token (CLS-like),
            visual_tokens:  (B, C, H, W) BCHW spatial features
        )
    """
    # stack -> (B, H, W, C) uint8 -> (B, C, H, W) float in [0, 1]
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(img)) for img in images]).to(model.device)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)

    if requires_grad:
        output = model(x)
    else:
        with torch.no_grad():
            output = model(x)

    # RADIO HF checkpoints return a `RadioOutput` (fields: summary, features);
    # some versions return a plain tuple. Handle both.
    if isinstance(output, (tuple, list)):
        summary, spatial_features = output[0], output[1]
    else:
        summary, spatial_features = output.summary, output.features

    batch_size, num_patches, num_channels = spatial_features.size()
    hw = int(np.sqrt(num_patches))
    visual_tokens = spatial_features.transpose(1, 2).reshape(batch_size, num_channels, hw, hw)  # BCHW
    summary_token = summary.unsqueeze(1)  # (B, 1, C_s) to mirror the cls_token layout of other teachers
    return summary_token, visual_tokens


def get_cradio_model(
    model_name: str = "nvidia/C-RADIOv3-H", device: str | torch.device = "cuda"
) -> tuple[torch.nn.Module, None]:
    """Get C-RADIO model. No separate processor (input conditioning is internal).

    Args:
        model_name (str, optional): name of the C-RADIO checkpoint. Defaults to
            "nvidia/C-RADIOv3-H".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[torch.nn.Module, None]: C-RADIO model and a `None` processor placeholder.
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return model, None


def print_feature_size(model_name: str = "nvidia/C-RADIOv3-H") -> None:
    """Print the sizes of features from C-RADIO.

    Handy to confirm the feature dimensions registered in
    `theia.foundation_models.common.MODEL_FEATURE_SIZES` for a given checkpoint.

    Args:
        model_name (str, optional): the name of the C-RADIO checkpoint.
    """
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw).convert("RGB"))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_cradio_model(model_name, device=device)
    summary_token, visual_tokens = get_cradio_feature(model, processor, image)
    # For C-RADIOv3-H at 224: summary (1, 1, 3840), visual (1, 1280, 14, 14).
    print(model_name, "summary:", tuple(summary_token.size()), "visual:", tuple(visual_tokens.size()))
    print("-> set MODEL_FEATURE_SIZES['%s'] = (%d, %d, %d)" % (
        model_name, visual_tokens.size(1), visual_tokens.size(2), visual_tokens.size(3)
    ))


if __name__ == "__main__":
    print_feature_size()
