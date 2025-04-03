# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from theia.models.backbones import build_backbone
from theia.models.feature_translators import build_feature_translator
from theia.models.utils import handle_feature_output


class RobotVisionFM(nn.Module):
    """Robot Vision Foundation Model (temporary name).

    Attributes:
        backbone (str | nn.Module): backbone network. Defaults to "deit-small-patch16-224".
        pretrained (bool): whether to use pretrained weights. Default to False.
        translator (str | nn.Module): feature translator module. Defaults to "conv".
        target_feature_sizes (Optional[dict[str, torch.Size | tuple[int, ...]]]):
            a dict to hold target feature sizes.
        translator_kwargs (Optional[dict[str, Any]]): other keyword arguments to the translator.
        target_loss_weights (Optional[dict[str, float]]):
            weights to balance loss from different target models. If not specified, use even weights.
        checkpoint_path: (Optional[str]): filename of pretrained weights to load.
        feature_reduce_method: (Optional[str]): how to reduce the feature in downstream applications.
    """

    def __init__(
        self,
        backbone: str | nn.Module = "facebook/deit-small-patch16-224",
        pretrained: bool = False,
        translator: str | nn.Module = "lconv",
        target_feature_sizes: Optional[dict[str, torch.Size | tuple[int, ...]]] = None,
        translator_kwargs: Optional[dict[str, Any]] = None,
        target_loss_weights: Optional[dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
        feature_reduce_method: Optional[str] = None,
        image_size: int = 224,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.target_feature_sizes = target_feature_sizes
        self.preprocessor = None
        self.pretrained = pretrained

        # backbone
        self.image_size = image_size
        self.backbone: nn.Module = build_backbone(backbone, pretrained, image_size=image_size, **kwargs)
        self.final_spatial = None
        if hasattr(self.backbone, "final_spatial"):
            self.final_spatial = self.backbone.final_spatial

        # handle output feature (feature reduce)
        self.feature_reduce_method = feature_reduce_method
        self.no_cls = hasattr(self.backbone, "no_cls")
        self.num_reg_tokens = self.backbone.num_reg_tokens if hasattr(self.backbone, "num_reg_tokens") else 0

        # translator
        backbone_feature_size = self.backbone.get_feature_size(keep_spatial=True)
        if self.target_feature_sizes:
            translator_kwargs = {} if translator_kwargs is None else OmegaConf.to_container(translator_kwargs)
            translator_kwargs["backbone_feature_size"] = backbone_feature_size
            translator_kwargs["target_feature_sizes"] = target_feature_sizes
            self.translator = build_feature_translator(translator, **translator_kwargs)

        # loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.cos_target = torch.ones((1), dtype=torch.int, requires_grad=False)
        self.target_loss_weights = target_loss_weights

    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights.

        Args:
            checkpoint_path (str): path to checkpoint / weight.
        """
        if checkpoint_path:
            weights_dict = torch.load(checkpoint_path, map_location="cpu")
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in weights_dict.items() if k in self.state_dict()}
            self.load_state_dict(pretrained_dict, strict=False)

    def freeze_translator(self) -> None:
        """Freeze the feature translator."""
        for param in self.translator.parameters():
            param.requires_grad = False

    def forward_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward RVFM feature only (before translators).

        Args:
            x (torch.Tensor): input image. By default it accepts images 
                in shape [B, H, W, C] or [B, C, H, W], pixel range [0,255], torch.uint8.
            kwargs (Any): kwargs including mainly those for huggingface preprocessor:
                `do_resize` (bool) defaults to True.
                `interpolate_pos_encoding` (Optional[bool]) defaults to None.
                `do_rescale` (bool) defaults to True.
                `do_normalize` (bool) defaults to True.

        Returns:
            torch.Tensor: RVFM feature.
        """
        feature = self.backbone(x, **kwargs)
        # [B, 1+H*W+N, C] if including both CLS and register tokens.
        # [B, 1+H*W, C] for standard model (N=0).
        # [B, H*W, C] for model without CLS.
        return handle_feature_output(feature, feature_reduce_method=self.feature_reduce_method, num_discard_tokens=self.num_reg_tokens)

    def forward(self, x: torch.Tensor, target_model_names: Optional[list[str]] = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass of Robot Vision Foundation Model.

        Args:
            x (torch.Tensor): input image. By default it accepts images 
                in shape [B, H, W, C] or [B, C, H, W], pixel range [0,255], torch.uint8.
            target_model_names (Optional[list[str]]): names of the target foundation models.
            kwargs (Any): kwargs including mainly those for huggingface preprocessor:
                `do_resize` (bool) defaults to True.
                `interpolate_pos_encoding` (Optional[bool]) defaults to None.
                `do_rescale` (bool) defaults to True.
                `do_normalize` (bool) defaults to True.

        Returns:
            dict[str, torch.Tensor]: features that match to each foundation model.
                Each feature is in [B, (H*W), C] or [B, C].
        """
        x = self.backbone(x, **kwargs)
        if self.num_reg_tokens > 0:
            x = x[:, :-self.num_reg_tokens]  # [B, (1)+H*W, C]
        features = self.translator(x, target_model_names, backbone_no_cls=self.no_cls)  # each is [B, H*W, C] or [B, C]
        return features

    def get_loss(self, pred_features: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Get loss terms given predictions and targets.

        Args:
            pred_features (dict[str, torch.Tensor]): predictions.
            y (dict[str, torch.Tensor]): targets.

        Returns:
            tuple[Any, ...]: loss terms
        """
        mse_loss_avg, cos_loss_avg, l1_loss_avg = 0, 0, 0
        mse_losses_per_model = {}
        cos_losses_per_model = {}
        l1_losses_per_model = {}

        for t in pred_features:
            pred = pred_features[t]
            target = y[t]

            # mse loss
            mse_loss = self.mse_loss(pred, target)
            weight = self.target_loss_weights if self.target_loss_weights else 1.0 / len(pred_features)

            # l1 loss
            l1_loss = self.l1_loss(pred, target)

            # cos loss
            pred_norm = F.normalize(pred.flatten(start_dim=1), dim=1, p=2)
            target_norm = F.normalize(target.flatten(start_dim=1), dim=1, p=2)
            target = self.cos_target.repeat(pred.size(0)).to(pred.device)
            cos_loss = self.cos_loss(pred_norm, target_norm, target)

            mse_loss_avg += mse_loss * weight
            cos_loss_avg += cos_loss / len(pred_features)  # balance cos by default for meaningful eval
            l1_loss_avg += l1_loss * weight

            mse_losses_per_model[t] = mse_loss.item()
            cos_losses_per_model[t] = cos_loss.item()
            l1_losses_per_model[t] = l1_loss.item()

        return {
            "mse_loss": mse_loss_avg,
            "cos_loss": cos_loss_avg,
            "l1_loss": l1_loss_avg,
            "mse_losses_per_model": mse_losses_per_model,
            "cos_losses_per_model": cos_losses_per_model,
            "l1_losses_per_model": l1_losses_per_model,
        }
