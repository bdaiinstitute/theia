## Expected Dataset Format

### Folder Structure
Dataset files are expected to be organized into the structure below.
```
<root_dir_to_hold_datasets>
├── imagenet
│   ├── images
│   ├───── imagenet_train-000000-train.tar
│   ├───── imagenet_train-000000-train.tar
│   ├───── ...
│   ├───── imagenet_val-000000-val.tar
│   ├───── ...
│   ├── LiheYoung_depth-anything-large-hf
│   ├── facebook_dinov2-large
│   ├── facebook_sam-vit-huge
│   ├── google_vit-huge-patch14-224-in21k
│   ├── openai_clip-vit-large-patch14
│   ├── <name of any addition model you want to have>
│   ├── splits.json
├── <any additional datasets>                                      
├── ...                                    
```


### Dataset shard
Each tar file is a "shard" in [webdataset](https://github.com/webdataset/webdataset) that contains the same number of samples of that column (i.e. feature or input image), defaults to 1000, except the last shard. You can also use `tarfile` library to inspect these shards.

For raw images, a shard contains
```
n01775062_3267.image  
n01775062_327.image
<name_corresponding_to_image_sample>.image
...
```
Images are in `numpy` arrays with `np.uint8`

For features, a shard contains
```
n01775062_3267.google_vit-huge-patch14-224-in21k.safetensors   
n01775062_327.google_vit-huge-patch14-224-in21k.safetensors
<name_corresponding_to_image_sample>.<model_name>.safetensors
...
```

Each `safetensors` is a huggingface [safetensor object](https://huggingface.co/docs/safetensors/en/index) containing features of that single image sample. There could be multiple keys within that safetensor. Usually, we use the format
```
{
    "embedding": spatial features in shape of [C, H, W],
    "cls_token": [CLS] token in shape of [C],
    ...
}
```

The file `splits.json` under each dataset defines number of samples in each split. For example, the content of `splits.json` for ImageNet is:
```
{
    "train": 1281167,
    "val": 50000,
    "test": 0
}
```

In our dataset pipeline file (src/theia/dataset/data_utils.py), `decode_sample()` is the method to decode single sample (an element in tar shard) to actual data. 