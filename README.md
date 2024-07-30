<h1 align="center">Theia: Distilling Diverse Vision Foundation Models for Robot Learning</h1>

<h4 align="center">
    <a href="https://www3.cs.stonybrook.edu/~jishang" target="_blank">Jinghuan Shang</a><sup>1,2</sup>, <a href="https://sites.google.com/view/karlschmeckpeper" target="_blank">Karl Schmeckpeper</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=_UnlC7IAAAAJ&hl=en" target="_blank">Brandon B. May</a><sup>1</sup>, Maria Vittoria Minniti<sup>1</sup>, <a href="http://kelestemur.com" target="_blank">Tarik Kelestemur</a><sup>1</sup>, <a href="https://davidjosephwatkins.com" target="_blank">David Watkins</a><sup>1</sup>, Laura Herlant<sup>1</sup>
</h4>
<p align="center">
<sup>1</sup><a href="http://theaiinstitute.com/" target="_blank">The AI Institute</a>
<sup>2</sup><a href="https://www3.cs.stonybrook.edu/~cvl/" target="_blank">Stony Brook University</a>
</p>
<p align="center">
<a href="" target="_blank"></a>Project Website (Coming Soon), <a href="https://arxiv.org/abs/2407.20179" target="_blank">Paper (arXiv)</a>, <a href="https://huggingface.co/collections/theaiinstitute/theia-66a7a6ae80a707547c358cce" target="_blank">Models</a>
</p>

<p align="center">
<!-- <img src="doc/theia_overview.gif"> -->
<img src="doc/theia_overview.gif" height="300px">
</p>

## Quick Start: Use Pre-trained Theia Models
Through huggingface:
```
import transformers
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", trust_remote_code=True)
fake_input = torch.zeros((1, 224 ,224, 3), dtype=torch.uint8)

theia_feature = model.forward_feature(fake_input)
# Theia / intermediate feature, mainly used for robot learning.
# To change different feature reduction methods, pass `feature_reduction_method` argument in AutoModel.from_pretrained() method

predicted_features = model(fake_input)
# predicted_features is dict[str, torch.Tensor] where each kv pair is target model name and predicted feature
# they are predicted features that tries to match teacher model features.
```

`theia-<size>-patch16-224-cdiv` are used for main evaluations in the paper.

## Installation
Make sure you have Python >= 3.10. Create any virtual Python environment you like or use the [Dockerfile](./Dockerfile). Then
```
pip install -e .
```

## Data Preparation

### Datasets
The datasets should be organized in webdataset format.

1. Prepare images from ImageNet

First download and [prepare](https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a) ImageNet.
```
cd src/theia/scripts/preprocessing/image_datasets
python organize_imagenet_webdataset.py --dataset <dataset_name> --imagenet-raw-path <path_to_raw_images> --output-path <root_dir_to_hold_datasets>
```
For any other image dataset you want to use, you can simply dump all of them in a folder (any subfolder also works), and modify how you can get their paths in `organize_imagenet_webdataset.py` (variable `image_paths`).

2. (Optional) Prepare frames from video datasets

```
cd src/theia/scripts/preprocessing/video_datasets
python subsampling_videos.py --dataset <dataset_name> --dataset-path <path_to_raw_videos> --output-path <root_dir_to_hold_datasets> [--subsampling-rate] [--samples-per-shard]
```

### Feature Extraction
```
cd src/theia/scripts/preprocessing
python feature_extraction.py --dataset <dataset_name> --output-path <root_dir_to_hold_datasets> --model <model_name> --split <train or val (or test)> [--num-gpus]
```

You can also refer to the integrated script `src/theia/scripts/preprocessing/iv_feature_extraction.py` that launches feature extraction for multiple models at the same time.

During training we will need mean and variance for each teacher model to normalize teacher features. You can extract them using `src/theia/scripts/preprocessing/calc_feature_mean.py` or use the stats we provide in `feature_stats`.

### Expected Dataset Format
More details about dataset format are available at [dataset_format](doc/dataset_format). Please use this to verify or toubleshoot your data.

## Training
```
cd src/theia/scripts

# train theia tiny using training configuration trian_rvfm_imagenet
# with teacher models CLIP, DINOv2, and ViT
torchrun --nproc_per_node=8 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:11111 train_rvfm.py --config-name=train_rvfm_imagenet logging.notes=imagenet_cdiv training/target_models=cdiv dataset.dataset_ratio=1.0 model.backbone.backbone=facebook/deit-tiny-patch16-224 logging.save_ckpt_interval=50000 dataset.dataset_root=<root_dir_to_hold_datasets>
```

To change output paths and wandb logging configs, override or modify `src/theia/configs/logging/default.yaml`.

To use different teacher models, override `training/target_models=<teacher model config>`. Available configs are under `src/theia/configs/training/target_models`

To change different datasets, override `dataset=<dataset config>`. Available configs are under `src/theia/configs/dataset`.




## Decode Theia-representation to VFM outputs

You can decode Theia-predicted VFM representations to their outputs. For DINOv2 we apply the PCA vsiualization, for SAM we use decoder to generate segmentation masks (but with SAM's pipeline of prompting), and for Depth-Anything we use the deocder head to do depth prediction. Below are example outputs. Theia model should be trained on those teachers during distillation. To use any models available online, you can find models with `cddsv` in its name, indicating that it is trained on all teachers.

![](doc/more_decoding_visualization.png)

Try out our online demo(work in-progress) or [notebook example](src/theia/example/decode_to_vfms.ipynb), or you can get outputs from local checkpoints by
```
cd src/theia/scripts/decoding
python decoding_example.py --backbone <backbone_name> --checkpoint-path <path to theia model checkpoint> --feature-stat-dir <where feature mean and std are placed> --media-to-vis-path <path to the video or image to decode>
```


## References
[Webdataset](https://github.com/webdataset/webdataset), [transformers](https://github.com/huggingface/transformers), [safetensors](https://huggingface.co/docs/safetensors/en/index), [DINOv2](https://github.com/facebookresearch/dinov2), [CLIP](https://github.com/openai/CLIP), [ViT](https://github.com/google-research/vision_transformer), [SAM](https://github.com/facebookresearch/segment-anything), [RADIO](https://github.com/NVlabs/RADIO)

## Citation
If you use Theia in your research, please use the following BibTeX entry:
```bibtex
@article{shang2024theia,
  author    = {Shang, Jinghuan and Schmeckpeper, Karl and May, Brandon B. and Minniti, Maria Vittoria and Kelestemur, Tarik and Watkins, David and Herlant, Laura},
  title     = {Theia: Distilling Diverse Vision Foundation Models for Robot Learning},
  journal   = {arXiv},
  year      = {2024},
}
```
