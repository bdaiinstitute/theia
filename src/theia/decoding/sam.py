# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Generator, Optional

import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from PIL import Image
from transformers import SamModel, SamProcessor
from transformers.image_utils import load_image
from transformers.pipelines import MaskGenerationPipeline


class MaskGenerationPipelineWithEmbeddings(MaskGenerationPipeline):
    """
    The wrapper class for huggingface transformers.pipelines.MaskGenerationPipeline
        that can decode from intermediate SAM embeddings.
    """

    def _sanitize_parameters(self, **kwargs: Any) -> tuple[dict[str, Any], ...]:
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_params = {}
        # preprocess args
        if "embeddings" in kwargs:  # inject embeddings here
            preprocess_kwargs["embeddings"] = kwargs["embeddings"]
        if "points_per_batch" in kwargs:
            preprocess_kwargs["points_per_batch"] = kwargs["points_per_batch"]
        if "points_per_crop" in kwargs:
            preprocess_kwargs["points_per_crop"] = kwargs["points_per_crop"]
        if "crops_n_layers" in kwargs:
            preprocess_kwargs["crops_n_layers"] = kwargs["crops_n_layers"]
        if "crop_overlap_ratio" in kwargs:
            preprocess_kwargs["crop_overlap_ratio"] = kwargs["crop_overlap_ratio"]
        if "crop_n_points_downscale_factor" in kwargs:
            preprocess_kwargs["crop_n_points_downscale_factor"] = kwargs["crop_n_points_downscale_factor"]
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]
        # postprocess args
        if "pred_iou_thresh" in kwargs:
            forward_params["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
        if "stability_score_offset" in kwargs:
            forward_params["stability_score_offset"] = kwargs["stability_score_offset"]
        if "mask_threshold" in kwargs:
            forward_params["mask_threshold"] = kwargs["mask_threshold"]
        if "stability_score_thresh" in kwargs:
            forward_params["stability_score_thresh"] = kwargs["stability_score_thresh"]
        if "crops_nms_thresh" in kwargs:
            postprocess_kwargs["crops_nms_thresh"] = kwargs["crops_nms_thresh"]
        if "output_rle_mask" in kwargs:
            postprocess_kwargs["output_rle_mask"] = kwargs["output_rle_mask"]
        if "output_bboxes_mask" in kwargs:
            postprocess_kwargs["output_bboxes_mask"] = kwargs["output_bboxes_mask"]
        return preprocess_kwargs, forward_params, postprocess_kwargs

    def preprocess(
        self,
        image: list[Image.Image],
        points_per_batch: int = 64,
        crops_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        points_per_crop: int = 32,
        crop_n_points_downscale_factor: int = 1,
        timeout: Optional[float] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Generator[Any, Any, Any]:
        image = load_image(image, timeout=timeout)
        target_size = self.image_processor.size["longest_edge"]
        crop_boxes, grid_points, cropped_images, input_labels = self.image_processor.generate_crop_boxes(
            image, target_size, crops_n_layers, crop_overlap_ratio, points_per_crop, crop_n_points_downscale_factor
        )
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt")

        with self.device_placement():
            if self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    if embeddings is None:
                        image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
                    else:
                        model_inputs.pop("pixel_values")
                        image_embeddings = embeddings
                    model_inputs["image_embeddings"] = image_embeddings

        n_points = grid_points.shape[1]
        points_per_batch = points_per_batch if points_per_batch is not None else n_points

        if points_per_batch <= 0:
            raise ValueError(
                "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
                "To return all points at once, set points_per_batch to None"
            )

        for i in range(0, n_points, points_per_batch):
            batched_points = grid_points[:, i : i + points_per_batch, :, :]
            labels = input_labels[:, i : i + points_per_batch]
            is_last = i == n_points - points_per_batch
            yield {
                "input_points": batched_points,
                "input_labels": labels,
                "input_boxes": crop_boxes,
                "is_last": is_last,
                **model_inputs,
            }


def draw_mask(mask: NDArray, random_color: bool = False) -> NDArray:
    """Draw the mask on an image.

    Args:
        mask (NDArray): mask in shape [height, width].
        random_color (bool): if using a random color. Defaults to False.

    Returns:
        NDArray: NDArray format of the image.
    """
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def decode_sam(
    features: torch.Tensor,
    images: list[Image.Image],
    mask_generator: Any,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.5,
    stability_score_thresh: float = 0.6,
    random_color: bool = True,
    device: int | str | torch.device = 0,
) -> NDArray:
    """Decode features using SAM (auto-prompting) mask generation pipeline.

    Args:
        features (torch.Tensor): features to be decoded, should be in shape [batch_size, num_tokens, latent_dim].
        images (list[Image.Image]): images corresponding to these features.
        mask_generator (Any): mask generation pipeline.
        points_per_batch (int): points per batch for auto-prompting. Defaults to 64.
            See transformers.pipelines.MaskGenerationPipeline for more details. Same below.
        pred_iou_thresh (float): iou threshold. Defaults to 0.5.
        stability_score_thresh (float): stability threshold. Defaults to 0.6.
        random_color (bool): if using a random color. Defaults to True.
        device (int | str | torch.device): device to perform the decoding. Defaults to 0.

    Returns:
        NDArray: decoded masks rendered in image format, represented by an NDArray in size
            [batch_size, height, width, channels] with value between [0, 1].
    """
    masks_rgbs = []
    num_patches = int(features.size(1) ** 0.5)
    features = rearrange(features, "b (h w) c -> b c h w", h=num_patches, w=num_patches)
    with torch.no_grad():
        for im, feature in zip(images, features, strict=False):
            predicted_ouputs = mask_generator(
                im,
                points_per_batch=points_per_batch,
                embeddings=feature.unsqueeze(0).to(device),
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            predicted_masks = predicted_ouputs["masks"]
            masks_rgb = np.zeros((224, 224, 3), dtype=np.float32)
            for mask in predicted_masks:
                masks_rgb += draw_mask(mask, random_color=random_color)
            # masks_rgb = cv2.cvtColor(masks_rgb, cv2.COLOR_RGBA2RGB)
            masks_rgbs.append(masks_rgb)
    return np.stack(masks_rgbs)


def prepare_mask_generator(device: int | str | torch.device = 0) -> MaskGenerationPipeline:
    """Prepare a mask generation pipeline on device `device`.

    Args:
        device (int | str | torch.device): device to perform mask generation. Defaults to 0.

    Returns:
        MaskGenerationPipeline: mask generator.
    """
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model.eval()
    mask_generator = MaskGenerationPipelineWithEmbeddings(
        task="mask_generation", model=sam_model, image_processor=processor.image_processor, device=device
    )
    return mask_generator, sam_model
