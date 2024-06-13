# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from theia.models.adapter_heads import ConvAdapterHead, LightConvAdapterHead, MLPAdapterHead, LinearAdapterHead


class FeatureTranslator(nn.Module):
    """Base class for the feature translator.

    The flow is backbone_adapter -> translator_stem -> translator_heads.

    Attributes:
        backbone_feature_size (torch.Size): the size of features of the backbone.
        target_feature_sizes (dict[str, torch.Size | tuple[int, ...]]): the sizes of features of target models.
        translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        target_model_names (list[str]): convenient attribute to hold all the names of the target models.

        backbone_adapter (nn.Module): the adapter to map channel dim of backbone to the translator hidden dim.
        translator_stem (nn.Module):  the shared stem for all target models.
        translator_heads (nn.ModuleDict): specific heads for different target models.
    """

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
    ) -> None:
        """Initalization function for FeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size | tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        """
        super().__init__()
        self.backbone_feature_size = backbone_feature_size  # (C, H, W)
        self.target_feature_sizes = target_feature_sizes  # [(C, H, W)]
        self.translator_hidden_size = translator_hidden_size  # C
        self.target_model_names = list(target_feature_sizes.keys())
        self.legit_target_model_name_map: dict[str, str] = {t: t.replace(".", "_") for t in self.target_model_names}
        self.translator_heads: nn.ModuleDict = None

        self.backbone_adapter = nn.Sequential(
            nn.LayerNorm(self.backbone_feature_size[0]),  # do a pre-norm
            nn.Linear(
                self.backbone_feature_size[0],  # C in [C,H,W]
                self.translator_hidden_size,
            ),
        )
        self.translator_stem: nn.Module = nn.Identity()
        self.build_translator_heads()

    def build_translator_heads(self) -> None:
        """Build translator heads to match the dimension of each target feature set.

        Example:
            translator_heads: dict[str, nn.Module] = ...
            self.translator_heads = nn.ModuleDict(translator_heads)
        """
        raise NotImplementedError("build_translator_heads() should be overridden")

    def forward(
        self, x: torch.Tensor, target_model_names: Optional[list[str]] = None, backbone_no_cls: bool = False
    ) -> torch.Tensor:
        """Forward pass for a base feature translator.

        Args:
            x (torch.Tensor): input features from the backbone. [B, (1)+H*W, C].
                (1) means optional CLS token. If `backbone_no_cls==True`, then [B, H*W, C].
            target_model_names (Optional[list[str]]): names of the target models.
            backbone_no_cls (bool): indicate backbone has cls token or not.
                Can use it to customize whether to drop cls.

        Returns:
            dict[str, torch.Tensor]: predicted features for target models.
        """
        # x: [B, (1)+H*W, C]
        x = self.backbone_adapter(x)  
        x = self.translator_stem(x) 
        target_model_names = target_model_names if target_model_names is not None else self.target_model_names
        features = {t: self.translator_heads[self.legit_target_model_name_map[t]](x, backbone_no_cls=backbone_no_cls) for t in target_model_names}
        return features


class MLPFeatureTranslator(FeatureTranslator):
    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
        translator_n_layer: int = 3,
    ) -> None:
        """Initalization function for MLPFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 2048.
            translator_n_layer (int): number of MLP layers. Defaults to 3.
        """
        self.translator_n_layer = translator_n_layer

        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build MLP translator heads to match the dimension of each target feature set."""
        translator_heads = {}
        source_size = (self.translator_hidden_size, *self.backbone_feature_size[1:])
        for target_model, target_size in self.target_feature_sizes.items():
            head = MLPAdapterHead(source_size=source_size, target_size=target_size, num_layer=self.translator_n_layer)
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class ConvFeatureTranslator(FeatureTranslator):
    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
    ) -> None:
        """Initalization function for ConvFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 2048.
        """
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build translator heads to match the dimension of each target feature set.

        Returns:
            nn.ModuleDict: the translator heads.
        """
        translator_heads = {}
        source_size = (self.translator_hidden_size, *self.backbone_feature_size[1:])
        for target_model, target_size in self.target_feature_sizes.items():
            head = ConvAdapterHead(source_size=source_size, target_size=target_size)
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class LightConvFeatureTranslator(FeatureTranslator):
    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
        hidden_size_factor: int | float = 1.0,
    ) -> None:
        """Initalization function for LightConvFeatureTranslator.
            It's for a smaller translator compared to ConvFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 1024.
            hidden_size_factor: the size of hidden dim of feature translator
                as a factor of input feature hidden dim. Defaults to 1.0
        """
        self.hidden_size_factor = hidden_size_factor
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )
        self.backbone_adapter = nn.Identity()

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build translator heads to match the dimension of each target feature set.

        Returns:
            nn.ModuleDict: the translator heads.
        """
        translator_heads = {}
        for target_model, target_size in self.target_feature_sizes.items():
            if "_cls" in target_model:
                head = LinearAdapterHead(
                    source_size=self.backbone_feature_size,
                    target_size=target_size
                )
            else:
                head = LightConvAdapterHead(
                    source_size=self.backbone_feature_size, 
                    target_size=target_size, 
                    hidden_size_factor=self.hidden_size_factor
                )
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class TransformerFreatureTranslator(FeatureTranslator):
    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, int]],
        translator_hidden_size: int = 1024,
        translator_n_layers: int = 2,
        translator_n_heads: int = 8,
        translator_activation: str = "gelu",
    ) -> None:
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

        self.translator_stem = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=translator_hidden_size,
                nhead=translator_n_heads,
                dim_feedforward=translator_hidden_size * 2,
                activation=translator_activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=translator_n_layers,
        )

        self.decode_tokens = nn.Parameter(
            torch.randn((1, math.prod(self.backbone_feature_size[1:]), translator_hidden_size))
        )

        self.target_model_emb = nn.ParameterDict(
            {
                self.legit_target_model_name_map[t]: torch.randn(1, 1, translator_hidden_size)
                for t in self.target_model_names
            }
        )

    def build_translator_heads(self) -> None:
        """Build Transformer translator heads to match the dimension of each target feature set."""
        translator_heads = {}
        for target_model, target_size in self.target_feature_sizes.items():
            head = MLPAdapterHead(
                source_size=(self.translator_hidden_size, *self.backbone_feature_size[1:]),
                target_size=target_size,
                num_layer=2,
            )
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)

    def forward(
        self, x: torch.Tensor, target_model_names: Optional[list[str]] = None, backbone_no_cls: bool = False
    ) -> torch.Tensor:
        """Forward pass for a simple linear translator.

        Args:
            x (torch.Tensor): input features from the backbone.
            target_model_names (Optional[str]): names of the target models.
            backbone_no_cls (bool): indicate backbone has cls token or not.
                Can use it to customize whether to drop cls.

        Returns:
            dict[str, torch.Tensor]: predicted features for target models.
        """
        if not backbone_no_cls:
            x = x[:, 1:]
        x = self.backbone_adapter(x)
        features = {}
        target_model_names = target_model_names if target_model_names is not None else self.target_model_names
        for t in target_model_names:
            feature = self.translator_stem(
                torch.cat(
                    [
                        self.decode_tokens.repeat(x.size(0), 1, 1),
                        self.target_model_emb[self.legit_target_model_name_map[t]].repeat(x.size(0), 1, 1),
                    ],
                    dim=1,
                ),
                memory=x,
            )[:, 1:, ...]
            features[t] = self.translator_heads[self.legit_target_model_name_map[t]](feature)
        return features


def build_feature_translator(translator_type: str, **kwargs: Any) -> FeatureTranslator:
    """Handy function to build feature translators given the type

    Args:
        translator_type (str): the type of the translator,
            one in `"mlp"`, `"conv"`, `"lconv"`, `"transformer"` (or `"trans"`).
            At the moment we are actively using `"lconv"`.

    Returns:
        FeatureTranslator: the corresponding FeatureTranslator
    """
    if translator_type == "mlp":
        return MLPFeatureTranslator(**kwargs)
    elif translator_type == "conv":
        return ConvFeatureTranslator(**kwargs)
    elif translator_type == "lconv":
        return LightConvFeatureTranslator(**kwargs)
    elif translator_type == "transformer" or translator_type == "trans":
        return TransformerFreatureTranslator(**kwargs)
    else:
        raise NotImplementedError(f"Requested {translator_type} is not implemented yet.")
