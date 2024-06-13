# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SamConfig, SamModel, SamProcessor
from transformers.models.sam.modeling_sam import SamMaskDecoder, SamMaskDecoderConfig
from transformers.utils import ModelOutput


class SamMaskDecoderWithFeature(SamMaskDecoder):
    """Mask decoder with upscaled feature exposed. Borrowed from transformers."""

    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__(config)

    # borrowd from huggingface transformer
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> Any:
        """Predict masks given image and prompt embeddings."""
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs: tuple[Any, ...] = (masks, iou_pred)

        if output_attentions:
            outputs = (*outputs, attentions)
        else:
            outputs = (*outputs, None)

        outputs = (*outputs, upscaled_embedding.reshape(batch_size * point_batch_size, num_channels, height, width))
        return outputs


@dataclass
class SamImageSegmentationWithFeatureOutput(ModelOutput):
    """Sam segmentation output plus features."""

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    image_embeddings: Optional[tuple[torch.FloatTensor]] = None
    upscaled_image_embeddings: Optional[tuple[torch.FloatTensor]] = None


class SamModelWithFeature(SamModel):
    """SAM model with feature exposed. Borrowed from transformers."""

    def __init__(self, config: SamConfig):
        super().__init__(config)
        self.mask_decoder = SamMaskDecoderWithFeature(config.mask_decoder_config)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Optional[dict[str, Any]],
    ) -> tuple | SamImageSegmentationWithFeatureOutput:
        """Sam forward pass with feature returned"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`,"
                " `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]  # type: ignore [union-attr]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:  # type: ignore [union-attr]
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),  # type: ignore [union-attr]
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions, upscaled_image_embeddings = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output: tuple[Any, ...] = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = (*output, vision_hidden_states)
            if output_attentions:
                output = (*output, vision_attentions, mask_decoder_attentions)

            output = (*output,)
            return output

        return SamImageSegmentationWithFeatureOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            image_embeddings=image_embeddings,
            upscaled_image_embeddings=upscaled_image_embeddings,
        )


class SamModelVisionFeature(SamModel):
    """Sam with only feature from the vision backbone. Borrowed from transformers."""

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Optional[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        """Sam forward pass that only goes through vision backbone and returns visual feature."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`,"
                " `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]  # type: ignore [union-attr]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        return SamImageSegmentationWithFeatureOutput(
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            image_embeddings=image_embeddings,
        )


def get_sam_feature(
    model: SamModel, processor: SamProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get features from SAM.

    Args:
        model (SamModel): SAM model.
        processor (SamProcessor): SAM input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (
            image_embeddings: feature from SAM visual encoder # (1, 256, 64, 64) if BCHW vit-huge
            upscaled_image_embeddings: features from mask decoder # (1, 32, 256, 256)
        )
    """
    inputs = processor(images, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    return (outputs.image_embeddings, outputs.upscaled_image_embeddings)


def get_sam_model(
    model_name: str = "facebook/sam-vit-huge", device: str | torch.device = "cuda", with_upscaled: bool = False
) -> tuple[SamModelWithFeature, SamProcessor]:
    """Get sam model and its input processor.

    Args:
        model_name (str, optional): name of SAM model. Defaults to "facebook/sam-vit-huge".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".
        with_upscaled (bool, optional): if return upscaled features. Defaults to False.

    Returns:
        tuple[SamModelWithFeature, SamProcessor]: SAM and its corresponding input processor
    """
    if with_upscaled:
        model = SamModelWithFeature.from_pretrained(model_name).to(device)
    else:
        model = SamModelVisionFeature.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(model_name)
    return model, processor


def print_feature_size(model_name: str = "facebook/sam-vit-huge") -> None:
    """Print the size of features from sam.

    Args:
        model_name (str, optional): the name of SAM model. Defaults to "facebook/sam-vit-huge".
    """
    import requests

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    image_array = [np.array(raw_image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_sam_model(model_name=model_name, device=device)
    image_embeddings, upscaled_embeddings = get_sam_feature(model, processor, image_array)

    print(image_embeddings.size(), upscaled_embeddings.size() if upscaled_embeddings is not None else None)
    # (1, 256, 64, 64) and (1, 32, 256, 256) for vit-huge
