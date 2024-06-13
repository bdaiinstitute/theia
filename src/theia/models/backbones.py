# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTModel


# Modified from huggingface transformers ViTEmbeddings
# Original Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
class ViTEmbeddingsNoCLS(ViTEmbeddings):
    """ViT Embedding Module without CLS token."""

    def __init__(self, config: AutoConfig, use_mask_token: bool = False):
        """Initialization.

        Args:
            config (AutoConfig): config for ViT.
            use_mask_token (bool, optional): whether to use mask token. Defaults to False.
        """
        super(ViTEmbeddingsNoCLS, self).__init__(config, use_mask_token=use_mask_token)
        self.cls_token = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings[:, 1:]

        embeddings = self.dropout(embeddings)

        return embeddings


# modified from huggingface transformers ViTModel
class ViTModelNoCLS(ViTModel):
    """ViT Model without CLS token."""

    def __init__(self, config: AutoConfig, add_pooling_layer: bool = True, use_mask_token: bool = False) -> None:
        super(ViTModelNoCLS, self).__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = ViTEmbeddingsNoCLS(config, use_mask_token=use_mask_token)
        self.no_cls = True

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


# modified from huggingface transformers ViTEmbeddings
class ViTEmbeddingsReg(ViTEmbeddings):
    """
    ViT Embedding Module with register tokens. https://openreview.net/forum?id=2dnO3LLiJ1
    """

    def __init__(self, config: AutoConfig, use_mask_token: bool = False, num_reg_tokens: int = 7):
        super(ViTEmbeddingsReg, self).__init__(config, use_mask_token=use_mask_token)
        self.reg_token = nn.Parameter(torch.randn(1, num_reg_tokens, config.hidden_size))
        self.num_reg_tokens = num_reg_tokens
        self.reg_pos_embed = nn.Parameter(torch.randn(1, num_reg_tokens, config.hidden_size))

        self.reg_pos_embed.data = nn.init.trunc_normal_(
            self.reg_pos_embed.data.to(torch.float32),
            mean=0.0,
            std=self.config.initializer_range,
        ).to(self.reg_pos_embed.dtype)

        self.reg_token.data = nn.init.trunc_normal_(
            self.reg_token.data.to(torch.float32),
            mean=0.0,
            std=self.config.initializer_range,
        ).to(self.reg_token.dtype)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1 - self.num_reg_tokens
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        reg_pos_embed = self.reg_pos_embed
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed, reg_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        reg_tokens = self.reg_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings, reg_tokens), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + torch.cat([self.position_embeddings, self.reg_pos_embed], dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


# modified from huggingface transformers ViTModel
class ViTModelReg(ViTModel):
    """ViT Model with register tokens. https://openreview.net/forum?id=2dnO3LLiJ1"""

    def __init__(
        self, config: AutoConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, num_reg_tokens: int = 7
    ):
        super(ViTModelReg, self).__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = ViTEmbeddingsReg(config, use_mask_token=use_mask_token, num_reg_tokens=num_reg_tokens)
        self.num_reg_tokens = num_reg_tokens

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


class DeiT(nn.Module):
    """DeiT model.

    Paper: Training data-efficient image transformers & distillation through attention
        https://arxiv.org/abs/2012.12877
    Huggingface Reference: https://huggingface.co/docs/transformers/en/model_doc/deit

    Attributes:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size
        model = AutoModel.from_pretrained(model_name)
        if pretrained:
            self.model = model
        else:
            deit_config = model.config
            self.model = AutoModel.from_config(deit_config)
            del model

        self.model.pooler = nn.Identity()

        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (224, 224)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)[:, 1:]  # for getting feature size, discard cls token
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: Optional[bool] = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

        Returns:
            torch.Tensor: model output.
        """
        input = self.processor(
            x, return_tensors="pt", do_resize=do_resize, do_rescale=do_rescale, do_normalize=do_normalize
        ).to(self.model.device)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state


class DeiTNoCLS(nn.Module):
    """Modified DeiT model without CLS token."""

    def __init__(
        self, model_name: str = "nocls-facebook/deit-small-patch16-224", pretrained: bool = False, image_size: int = 224
    ):
        super().__init__()
        self.image_size = image_size
        pretrained_model_name = model_name.replace("nocls-", "")
        deit_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.model = ViTModelNoCLS(deit_config)
        if pretrained:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in self.model.state_dict()}
            self.load_state_dict(pretrained_dict, strict=False)
            del pretrained_model, pretrained_dict

        self.model.pooler = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.no_cls = True

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (self.image_size, self.image_size)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: Optional[bool] = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

            - argument for forward
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.

        Returns:
            torch.Tensor: model output.
        """
        input = self.processor(
            x, return_tensors="pt", do_resize=do_resize, do_rescale=do_rescale, do_normalize=do_normalize
        ).to(self.model.device)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state


class DeiTReg(nn.Module):
    """Modified DeiT model with register tokens."""

    def __init__(
        self,
        model_name: str = "reg-facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
        num_reg_tokens: int = 7,
    ):
        super().__init__()
        self.image_size = image_size
        pretrained_model_name = model_name.replace("reg-", "")
        deit_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.model = ViTModelReg(deit_config, num_reg_tokens=num_reg_tokens)
        if pretrained:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in self.model.state_dict()}
            self.load_state_dict(pretrained_dict, strict=False)
            del pretrained_model, pretrained_dict

        self.model.pooler = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.num_reg_tokens = num_reg_tokens

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (self.image_size, self.image_size)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)[:, 1 : -self.num_reg_tokens]
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: Optional[bool] = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

        Returns:
            torch.Tensor: model output.
        """
        input = self.processor(
            x, return_tensors="pt", do_resize=do_resize, do_rescale=do_rescale, do_normalize=do_normalize
        ).to(self.model.device)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state


def build_backbone(model_name: str, pretrained: bool = False, image_size: int = 224, **kwargs: Any) -> nn.Module:
    """Build the backbone visual encoder of robot vision foundation model.

    Args:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights. Defaults to False.
        image_size (int): size of the image. Assume a square image. Defaults to 224
        kwargs (Any): any kwargs specific to some models. For example,
            `num_reg_tokens` for `DeiTReg` when `"reg"` in `model_name`

    Returns:
        nn.Module: backbone network.
    """
    if "reg" in model_name:
        return DeiTReg(model_name=model_name, pretrained=pretrained, image_size=image_size, **kwargs)
    elif "nocls" in model_name:
        return DeiTNoCLS(model_name=model_name, pretrained=pretrained, image_size=image_size, **kwargs)
    elif "deit" in model_name:
        return DeiT(model_name=model_name, pretrained=pretrained, image_size=image_size)
    else:
        raise NotImplementedError(f"Requested {model_name} is not implemented.")
