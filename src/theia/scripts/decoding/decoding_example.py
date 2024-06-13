# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Example script to decode features from theia model to corresponding visual task output,
    including DINOv2 visualization, SAM segmentation masks, and Depth Anything predicted depths.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import transformers

from PIL import Image
from theia.foundation_models.common import get_model_feature_size
from theia.decoding import decode_everything, load_feature_stats, prepare_depth_decoder, prepare_mask_generator
from theia.models.rvfm import RobotVisionFM
from theia.utils.seed import seed_everything
from torchvision.io import read_video, write_video

transformers.logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="facebook/deit-tiny-patch16-224", help="name of the backbone")
    parser.add_argument("--checkpoint-path", type=str, help="path to the model weights")
    parser.add_argument("--feature-stat-dir", type=str, help="the directory to find feature stats")
    parser.add_argument("--media-to-vis-path", type=str, help="the location of source video / image for visualization")
    parser.add_argument(
        "--vis-output-dir", type=str, default="./vis_output/", help="output dir to save visualization result"
    )
    args = parser.parse_args()
    seed_everything(0)
    device = 0

    target_model_names = [
        "google/vit-huge-patch14-224-in21k",
        "facebook/dinov2-large",
        "openai/clip-vit-large-patch14",
        "facebook/sam-vit-huge",
        "LiheYoung/depth-anything-large-hf",
    ]
    target_feature_sizes = {t: get_model_feature_size(t, keep_spatial=True) for t in target_model_names}
    theia_model = RobotVisionFM(
        translator="lconv", target_feature_sizes=target_feature_sizes, backbone=args.backbone, pretrained=False
    )

    theia_model.load_pretrained_weights(args.checkpoint_path)
    theia_model = theia_model.to(device)
    feature_means, feature_vars = load_feature_stats(target_model_names, stat_file_root=args.feature_stat_dir)

    mask_generator, sam_model = prepare_mask_generator(device)
    depth_anything_model_name = "LiheYoung/depth-anything-large-hf"
    depth_anything_decoder, _ = prepare_depth_decoder(depth_anything_model_name, device)

    if args.media_to_vis_path.lower().endswith((".mp4")):
        video, _, _ = read_video(args.media_to_vis_path, pts_unit="sec", output_format="THWC")
        video = video.numpy()
        images = [Image.fromarray(cv2.resize(im, (224, 224))) for im in video]
    elif args.media_to_vis_path.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        images = [Image.open(args.media_to_vis_path).resize((224, 224))]

    theia_decode_results, gt_decode_results = decode_everything(
        theia_model=theia_model,
        feature_means=feature_means,
        feature_vars=feature_vars,
        images=images,
        mask_generator=mask_generator,
        sam_model=sam_model,
        depth_anything_decoder=depth_anything_decoder,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.7,
        gt=True,
        device=device,
    )


    if not os.path.exists(args.vis_output_dir):
        os.makedirs(args.vis_output_dir)
    if len(images) > 1:
        vis_output_save_fn = (
            f"{args.media_to_vis_path.split('/')[-1].split('.')[0]}_{args.checkpoint_path.split('/')[-1].replace('.pth', '')}.mp4"
        )
        vis_video = np.stack(
            [np.vstack([tr, gtr]) for tr, gtr in zip(theia_decode_results, gt_decode_results, strict=False)]
        )
        vis_video = torch.from_numpy(vis_video * 255.0).to(torch.uint8)
        
        vis_save_path = os.path.join(args.vis_output_dir, vis_output_save_fn)
        write_video(vis_save_path, vis_video, fps=10)
    else:
        vis_output_save_fn = (
            f"{args.media_to_vis_path.split('/')[-1].split('.')[0]}_{args.checkpoint_path.split('/')[-1].replace('.pth', '')}.png"
        )
        vis_image = np.stack(
            [np.vstack([tr, gtr]) for tr, gtr in zip(theia_decode_results, gt_decode_results, strict=False)]
        )
        vis_image = Image.fromarray((vis_image * 255.0).astype(np.uint8)[0])
        vis_save_path = os.path.join(args.vis_output_dir, vis_output_save_fn)
        vis_image.save(vis_save_path)


if __name__ == "__main__":
    main()
