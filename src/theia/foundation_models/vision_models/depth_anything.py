# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
# File modified.

# -----------------------------------------------------------------------
# Copyright 2024 TikTok and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Depth Anything model."""
import copy
from typing import Any, Optional

import numpy.typing as npt
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoImageProcessor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import DepthEstimatorOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoBackbone
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DepthAnythingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthAnythingModel`].
    It is used to instantiate an DepthAnything model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DepthAnything
    [LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`dict[str, Any] | PretrainedConfig`, *optional*):
            The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
            leverage the [`AutoBackbone`] API.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights
            from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        patch_size (`int`, *optional*, defaults to 14):
            The size of the patches to extract from the backbone features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        reassemble_hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the reassemble layers.
        reassemble_factors (`tuple[int | float, ...]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`tuple[int]`, *optional*, defaults to `[48, 96, 192, 384]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 64):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the depth estimation head.
        head_hidden_size (`int`, *optional*, defaults to 32):
            The number of output channels in the second convolution of the depth estimation head.
    ```"""

    model_type = "depth_anything"

    def __init__(
        self,
        backbone_config: dict[str, Any] | PretrainedConfig = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        patch_size: int = 14,
        initializer_range: float = 0.02,
        reassemble_hidden_size: int = 384,
        reassemble_factors: tuple[int | float, ...] = (4, 2, 1, 0.5),
        neck_hidden_sizes: tuple[int, ...] = (48, 96, 192, 384),
        fusion_hidden_size: int = 64,
        head_in_index: int = -1,
        head_hidden_size: int = 32,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Dinov2` backbone.")
            backbone_config = CONFIG_MAPPING["dinov2"](
                image_size=518,
                hidden_size=384,
                num_attention_heads=6,
                out_indices=[9, 10, 11, 12],
                apply_layernorm=True,
                reshape_hidden_states=False,
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output


class DepthAnythingReassembleLayer(nn.Module):
    def __init__(self, config: DepthAnythingConfig, channels: int, factor: int | float):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=config.reassemble_hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    # Copied from transformers.models.dpt.modeling_dpt.DPTReassembleLayer.forward
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)

        return hidden_state


class DepthAnythingReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Take the patch embeddings and reshape them to image-like feature representations.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        for channels, factor in zip(config.neck_hidden_sizes, config.reassemble_factors, strict=False):
            self.layers.append(DepthAnythingReassembleLayer(config, channels=channels, factor=factor))

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height: Optional[int] = None, patch_width: Optional[int] = None
    ) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            # reshape to (batch_size, num_channels, height, width)
            hidden_state = hidden_state[:, 1:]
            batch_size, _, num_channels = hidden_state.shape
            hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
            hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
            hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


class DepthAnythingPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        return hidden_state + residual


class DepthAnythingFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        self.residual_layer1 = DepthAnythingPreActResidualLayer(config)
        self.residual_layer2 = DepthAnythingPreActResidualLayer(config)

    def forward(
        self, hidden_state: torch.Tensor, residual: Optional[torch.Tensor] = None, size: Optional[int] = None
    ) -> torch.Tensor:
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class DepthAnythingFeatureFusionStage(nn.Module):
    # Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage.__init__ with DPT->DepthAnything
    def __init__(self, config: DepthAnythingConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DepthAnythingFeatureFusionLayer(config))

    def forward(self, hidden_states: torch.Tensor, size: Optional[int] = None) -> list[torch.Tensor]:
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # first layer only uses the last hidden_state
        size = hidden_states[1].shape[2:]
        fused_hidden_state = self.layers[0](hidden_states[0], size=size)
        fused_hidden_states.append(fused_hidden_state)

        # looping from the last layer to the second
        for idx, (hidden_state, layer) in enumerate(zip(hidden_states[1:], self.layers[1:], strict=False)):
            size = hidden_states[1:][idx + 1].shape[2:] if idx != (len(hidden_states[1:]) - 1) else None

            fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size)

            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


# Copied from transformers.models.dpt.modeling_dpt.DPTPreTrainedModel with DPT->DepthAnything,dpt->depth_anything
class DepthAnythingPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DepthAnythingConfig
    base_model_prefix = "depth_anything"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DepthAnythingNeck(nn.Module):
    """
    DepthAnythingNeck. A neck is a module that is normally used between the backbone and the head.
    It takes a list of tensors as input and produces another list of tensors as output.
    For DepthAnything, it includes 2 stages:

    * DepthAnythingReassembleStage
    * DepthAnythingFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()
        self.config = config

        self.reassemble_stage = DepthAnythingReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # fusion
        self.fusion_stage = DepthAnythingFeatureFusionStage(config)

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height: Optional[int] = None, patch_width: Optional[int] = None
    ) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)`
            or `(batch_size, hidden_size, height, width)`): List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output


class DepthAnythingDepthEstimationHead(nn.Module):
    """
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the DPT paper's
    supplementary material).
    """

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()

        self.head_in_index = config.head_in_index
        self.patch_size = config.patch_size

        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features // 2, config.head_hidden_size, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU()
        self.conv3 = nn.Conv2d(config.head_hidden_size, 1, kernel_size=1, stride=1, padding=0)
        self.activation2 = nn.ReLU()

    def forward(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        hidden_states = hidden_states[self.head_in_index]

        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(patch_height * self.patch_size), int(patch_width * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)

        return predicted_depth


class DepthAnythingForDepthEstimation(DepthAnythingPreTrainedModel):
    def __init__(self, config: DepthAnythingConfig):
        super().__init__(config)

        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.neck = DepthAnythingNeck(config)
        self.head = DepthAnythingDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, ...] | DepthEstimatorOutput:
        r"""
        Forward pass for Depth Anything.

        Args:
            pixel_values (torch.FloatTensor): input images.
            labels (Optional[torch.LongTensor]: labels for loss. Defaults to None.
            output_attentions (Optional[bool]): whether to return attentions. Defaults to None.
            output_hidden_states (Optional[bool]): whether to return hidden_states. Defaults to None.
            return_dict (Optional[bool]): whether to return dict. Defaults to None.

        Returns:
            Tuple[torch.Tensor] | DepthEstimatorOutput: forward output

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states, patch_height, patch_width)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output  # noqa

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class DepthAnythingNeckFeature(DepthAnythingForDepthEstimation):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for Depth Anything with only neck feature returned.

        Args:
            pixel_values (torch.FloatTensor): input images.
            labels (Optional[torch.LongTensor]: labels for loss. Defaults to None.
            output_attentions (Optional[bool]): whether to return attentions. Defaults to None.
            output_hidden_states (Optional[bool]): whether to return hidden_states. Defaults to None.
            return_dict (Optional[bool]): whether to return dict. Defaults to None.

        Returns:
            torch.Tensor: neck feature.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        return hidden_states


class DepthAnythingHeadFeature(DepthAnythingForDepthEstimation):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for Depth Anything with only last layer (head) feature returned.

        Args:
            pixel_values (torch.FloatTensor): input images.
            labels (Optional[torch.LongTensor]: labels for loss. Defaults to None.
            output_attentions (Optional[bool]): whether to return attentions. Defaults to None.
            output_hidden_states (Optional[bool]): whether to return hidden_states. Defaults to None.
            return_dict (Optional[bool]): whether to return dict. Defaults to None.

        Returns:
            torch.Tensor: last layer (head) feature
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        hidden_states = hidden_states[-1]

        head_feature = self.head.conv1(hidden_states)
        head_feature = nn.functional.interpolate(
            head_feature,
            (int(patch_height * patch_size), int(patch_width * patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        head_feature = self.head.conv2(head_feature)

        return head_feature


def get_depth_anything_feature(
    model: DepthAnythingForDepthEstimation,
    processor: AutoImageProcessor,
    images: list[npt.NDArray],
    requires_grad: Optional[bool] = False,
) -> torch.Tensor | list[torch.Tensor]:
    """Get feature (after neck) from depth anything model.

    Args:
        model (DepthAnythingNeckFeature): Depth Anything model.
        processor (AutoImageProcessor): Depth Anything processor.
        images (list[npt.NDArray]): images to extract feature.
        requires_grad (Optional[bool], optional): whether to keep gradient. Defaults to False.

    Returns:
        torch.Tensor: feature from depth anything model.
    """
    inputs = processor(images, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
            # if neck
            # [torch.Size([1, D, 37, 49]), torch.Size([1, D, 74, 98]),
            # torch.Size([1, D, 148, 196]), torch.Size([1, D, 296, 392])]
            # D = 64, 128, 256 for small, base, large
            # if head
            # torch.Size([1, 32, 518, 686])
    return outputs


def get_depth_anything_model(
    model_name: Optional[str] = "LiheYoung/depth-anything-large-hf",
    device: Optional[str | torch.device] = "cuda",
    selected_feature: Optional[str] = "neck",
) -> tuple[DepthAnythingForDepthEstimation, AutoImageProcessor]:
    """Get depth anything model.

    Args:
        model_name (Optional[str]): name of the model. Defaults to "LiheYoung/depth-anything-large-hf".
        device (Optional[str | torch.device]): device to put model on. Defaults to "cuda".

    Returns:
        Tuple[DepthAnythingForDepthEstimation, AutoImageProcessor]: Depth Anything model and the processor.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    if selected_feature == "neck":
        model = DepthAnythingNeckFeature.from_pretrained(model_name).to(device)
    elif selected_feature == "head":
        model = DepthAnythingHeadFeature.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"{selected_feature} is not supported for Depth Anything")
    return model, processor


def print_feature_size(
    model_name: Optional[str] = "LiheYoung/depth-anything-large-hf", selected_feature: Optional[str] = "neck"
) -> None:
    """Print the size of the feature from Depth Anything.

    Args:
        model_name (Optional[str]): the name of Depth Anything model.
            Defaults to "LiheYoung/depth-anything-large-hf".
    """
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [Image.open(requests.get(url, stream=True).raw)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_depth_anything_model(model_name=model_name, device=device, selected_feature=selected_feature)

    with torch.no_grad():
        embedding = get_depth_anything_feature(model, processor, image)

    print([x.size() for x in embedding] if isinstance(embedding, list) else embedding.size())
