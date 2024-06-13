# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import torch
import torch.nn as nn
from theia.foundation_models import get_clip_model, get_deit_model, get_dinov2_model, get_sam_model, get_vit_model
from transformers import AutoImageProcessor, AutoModel

from theia.models.utils import handle_feature_output


class VFMEncoder(nn.Module):
    """Wrapper class of an individual VFM Encoder for feature extraction.

    Attrs:
        model_name (str): name of the model.
        feature_reduce_method (str): how to select the output feature token and shape.
        processor (AutoProcessor): input pre-processor.
    """

    def __init__(self, model_name: str, feature_reduce_method: Optional[str] = None, **kwargs: Any):
        """Instanciate a (off-the-shelf) VFM encoder.

        Args:
            model_name (str): name of the encoder
            feature_reduce_method (Optional[str]): how to select the output feature token and shape. Defaults to None.
            **kwargs (Any): anything not needed got pass-through
        """
        super().__init__()
        self.model_name = model_name
        if "google/vit" in model_name:
            model, processor = get_vit_model(model_name, device="cpu")
        elif "facebook/dino" in model_name:
            model, processor = get_dinov2_model(model_name, device="cpu")
        elif "facebook/sam" in model_name:
            model, processor = get_sam_model(model_name, device="cpu")
        elif "openai/clip" in model_name:
            model, processor = get_clip_model(model_name, device="cpu")
        elif "facebook/deit" in model_name:
            model, processor = get_deit_model(model_name, device="cpu")
        elif "nvidia" in model_name:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            processor = AutoImageProcessor.from_pretrained(model_name)
        elif "mvp" in model_name:
            import mvp

            model_name_mvp = model_name.replace("mvp-", "")
            model = mvp.load(model_name_mvp)
            processor = None
        elif "vip" in model_name:
            from vip import load_vip

            model = load_vip()
            processor = None
        elif "r3m" in model_name:
            from r3m import load_r3m

            model_name_r3m = model_name.replace("r3m-", "")
            model = load_r3m(model_name_r3m)
            processor = None
        else:
            raise NotImplementedError(f"{model_name} is not supported in theia.models.vfm.VFM")

        self.model = model
        self.processor = processor
        self.feature_reduce_method = feature_reduce_method
        if "image_size" in kwargs:
            self.image_size = kwargs["image_size"]
        if "final_spatial" in kwargs:
            self.final_spatial = kwargs["final_spatial"]

    def get_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return the feature from the VFM.

        Args:
            x (torch.Tensor): input image.
            kwargs: any arguments pass-through (mainly for processor currently).
                For example, `do_rescale`, `do_resize`, `interpolate_pos_encoding`
                    to control image preprocessing pipeline.

        Returns:
            torch.Tensor: feature.
        """
        if (
            "google/vit" in self.model_name
            or "facebook/dinov2" in self.model_name
            or "facebook/deit" in self.model_name
        ):
            inputs = self.processor(x, return_tensors="pt", **kwargs).to(self.model.device)
            feature = self.model(**inputs).last_hidden_state
        elif "openai/clip" in self.model_name:
            inputs = self.processor(images=x, return_tensors="pt", **kwargs).to(self.model.device)
            feature = self.model(**inputs).last_hidden_state
        elif "facebook/sam" in self.model_name:
            inputs = self.processor(x, return_tensors="pt", **kwargs).to(self.model.device)
            feature = self.model(**inputs).image_embeddings
        elif "nvidia" in self.model_name:
            inputs = (
                self.processor(images=x, return_tensors="pt", **kwargs)
                .pixel_values.to(torch.bfloat16)
                .to(self.model.device)
            )
            summary, feature = self.model(inputs)
            if self.feature_reduce_method == "cls_identity":
                feature = summary.to(torch.float32)
            else:
                feature = feature.to(torch.float32)
        elif "mvp" in self.model_name:
            feature = self.model(x)
        elif "vip" in self.model_name:
            feature = self.model(x)
        elif "r3m" in self.model_name:
            feature = self.model(x)
        return feature

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward method, including getting the feature and handle the output token / shape.

        Args:
            x (torch.Tensor): input image.

        Returns:
            torch.Tensor: output feature with token or shape handled.
        """
        feature = self.get_feature(x, **kwargs)  # [B, 1+H*W, C]
        return handle_feature_output(feature, self.feature_reduce_method)

    def forward_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Alias of forward() to accommandate some downstream usage.

        Args:
            x (torch.Tensor): input image.

        Returns:
            torch.Tensor: output feature with token or shape handled.
        """
        return self.forward(x, **kwargs)


class ConcatVFMEncoder(nn.Module):
    """Wrapper class that combines features from multiple VFM Encoders. The combination is channel-wise concatenation.

    Attrs:
        model_names (list[str]): names of the models.
        feature_reduce_method (Optional[str]): how to select the output feature token and shape.
        model (nn.ModuleDict): a dict to hold different VFM encoders.
    """

    def __init__(self, model_names: list[str], feature_reduce_method: Optional[str] = None, **kwargs: Any):
        """Instanciate a (off-the-shelf) VFM encoder.

        Args:
            model_names (list[str]): name of the encoder
            feature_reduce_method (str, optional): how to select the output feature token and shape. Defaults to None.
            **kwargs (Any): anything not needed got pass-through
        """
        super().__init__()
        self.model_names = model_names
        self.model = {}
        for model_name in model_names:
            model = VFMEncoder(model_name, feature_reduce_method=feature_reduce_method, **kwargs)
            self.model[model_name] = model

        self.model = nn.ModuleDict(self.model)
        self.feature_reduce_method = feature_reduce_method

    def get_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Get different features from VFMs.

        Args:
            x (torch.Tensor): input image.

        Returns:
            torch.Tensor: features concatenated at channel dimension.
        """
        features = []
        for model_name in self.model_names:
            features.append(self.model[model_name](x, **kwargs))
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward method, including getting the feature and handle the output token / shape.

        Args:
            x (torch.Tensor): input image.

        Returns:
            torch.Tensor: output feature with token or shape handled.
        """
        feature = self.get_feature(x, **kwargs)  # [B, 1+H*W, C]
        return handle_feature_output(feature, self.feature_reduce_method)

    def forward_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Alias of forward() to accommandate some downstream usage.

        Args:
            x (torch.Tensor): input image.

        Returns:
            torch.Tensor: output feature with token or shape handled.
        """
        return self.forward(x, **kwargs)
