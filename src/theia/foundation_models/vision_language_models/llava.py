# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast


@dataclass
class LlavaVisualFeatureOutput(LlavaCausalLMOutputWithPast):
    """Visual feature output for LLaVA.

    Args:
        visual_embeddings (Optional[torch.FloatTensor]): feature from visual encoder.
    """

    visual_embeddings: Optional[torch.FloatTensor] = None


class LlavaVisualFeature(LlavaForConditionalGeneration):
    """LLaVA model with only visual feature returned. Borrowed from transformers."""

    # TODO: reduce VRAM use of language model part, because only vocabulary is used, not the whole model
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | LlavaVisualFeatureOutput:
        """LLaVA visual encoder forward pass, from transformers package.

        Returns:
            tuple | LlavaVisualFeatureOutput: feature from visual encoder.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        image_features = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    image_features = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    image_features = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )
        return LlavaVisualFeatureOutput(visual_embeddings=image_features)


def get_llava_visual_feature(
    model: LlavaVisualFeature, processor: AutoProcessor, images: list[np.array], requires_grad: bool = False
) -> torch.FloatTensor:
    """Get the feature from the visual encoder of LLaVA.

    Args:
        model (LlavaVisualFeature): LLaVA model
        processor (AutoProcessor): LLaVA input processor
        images (list[np.array]): images to be encoded, in RGB, uint8
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        torch.FloatTensor: LLaVA feature. (1, 1024, 24, 24) if using llava-7b
    """
    inputs = processor(text=["placeholder"], images=images, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    batch_size, num_patches, num_channels = outputs.visual_embeddings.size()
    visual_tokens = outputs.visual_embeddings.transpose(1, 2).reshape(
        batch_size, num_channels, int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
    )
    return visual_tokens  # (1, 1024, 24, 24) if llava-7b


def get_llava_vision_model(
    model_name: str = "llava-hf/llava-1.5-7b-hf", device: str | torch.device = "cuda"
) -> tuple[LlavaVisualFeature, AutoProcessor]:
    """Get LLaVA model and its input processor.

    Args:
        model_name (str, optional): name of LLaVA model. Defaults to "llava-hf/llava-1.5-7b-hf".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[LlavaVisualFeature, AutoProcessor]: LLaVA model and the corresponding input processor.
    """
    model = LlavaVisualFeature.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def print_feature_size(model_name: str = "llava-hf/llava-1.5-7b-hf") -> None:
    """Print the size of the feature from LLaVA.

    Args:
        model_name (str, optional): the name of LLaVA model. Defaults to "llava-hf/llava-1.5-7b-hf".
    """
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = [np.array(image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_llava_vision_model(model_name=model_name, device=device)
    feature = get_llava_visual_feature(model, processor, image)
    print(model_name, feature.size())
    # (1, 1024, 24, 24) if llava-7b


if __name__ == "__main__":
    print_feature_size()
