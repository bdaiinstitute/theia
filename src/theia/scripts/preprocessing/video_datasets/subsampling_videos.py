# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import os
import shutil
import tarfile
from io import BytesIO

import numpy as np
import torch
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image
from torchvision.io import VideoReader, read_video
from torchvision.transforms import Compose, Resize, ToPILImage

# torchvision.set_video_backend("video_reader")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument(
    "--dataset-path",
    type=str,
    help="please provide the dataset path directly contains videos (.mp4, .webm) or frames (.tar for epic_kitchen)",
)
parser.add_argument("--output-path", type=str, help="will create a subfolder within this output path")
parser.add_argument("--subsampling-rate", type=int, default=-1)
parser.add_argument("--samples-per-shard", type=int, default=1000)
args = parser.parse_args()


if args.dataset == "ego4d":
    # default sampling rate for ego4d
    SUBSAMPLING_RATE = 150 if args.subsampling_rate > 0 else args.subsampling_rate
    video_ext = ".mp4"
elif args.dataset == "ssv2":
    # default sampling rate for ego4d
    SUBSAMPLING_RATE = 32 if args.subsampling_rate > 0 else args.subsampling_rate
    video_ext = ".webm"
elif args.dataset == "epic_kitchen":
    # default sampling rate for ego4d
    SUBSAMPLING_RATE = 60 if args.subsampling_rate > 0 else args.subsampling_rate
    video_ext = ".tar"
else:
    raise NotImplementedError(f"{args.dataset} is not supported.")

print(f"subsampling {args.dataset} by 1/{SUBSAMPLING_RATE}")

RAW_VIDEO_PATH = args.dataset_path
TMP_SAMPLED_FRAMES_PATH = f"/storage/nvme/tmp_video_subsampling/{args.dataset}_1in{SUBSAMPLING_RATE}_images"
SAMPLED_FRAMES_PATH = os.path.join(args.output_path, f"{args.dataset}_1in{SUBSAMPLING_RATE}_images")
os.makedirs(SAMPLED_FRAMES_PATH, exist_ok=True)
os.makedirs(TMP_SAMPLED_FRAMES_PATH, exist_ok=True)

SAMPLES_PER_SHARD = args.samples_per_shard

video_fns = sorted([fn for fn in os.listdir(RAW_VIDEO_PATH) if video_ext in fn])

transform = Compose([Resize((224, 224), antialias=True), ToPILImage()])


def check_existing_shard(path: str) -> bool:
    """
    Check the integrity of a shard given path.

    Returns:
        bool: True if the shard exists and is complete.
    """
    if os.path.exists(path):
        try:
            tarf = tarfile.open(path)
            for _ in tarf.getmembers():
                pass
        except tarfile.TarError:
            return False
    else:
        return False
    return True


def create_shard(shard_idx: int, frames: list[tuple[NDArray, str]]) -> None:
    """Create a shard given index and frame list.

    Args:
        shard_idx (int): index of this shard. Used to determine file paths.
        frames (list[tuple[NDArray, str]]): frames to write to this shard.
    """
    shard_fn = f"{args.dataset}_1in{SUBSAMPLING_RATE}-{shard_idx:06d}.tar"
    local_shard_path = os.path.join(TMP_SAMPLED_FRAMES_PATH, shard_fn)
    remote_shard_path = os.path.join(SAMPLED_FRAMES_PATH, shard_fn)
    if check_existing_shard(remote_shard_path):
        print(f"creating {args.dataset} shard {shard_idx:06d} - check pass, skip\r", end="")
        return
    print(f"creating {args.dataset} shard {shard_idx:06d}\r", end="")
    with wds.TarWriter(local_shard_path) as tar_writer:
        for i, (image, basename) in enumerate(frames):
            image_out = BytesIO()
            np.save(image_out, image)
            sample = {"__key__": basename, "image": image_out.getvalue()}
            tar_writer.write(sample)
            if (i + 1) % 20 == 0:
                print(
                    f"creating {args.dataset} shard {shard_idx:06d} - {int((i+1) / len(frames) * 100):02d}%\r", end=""
                )

    # move from local to remote
    shutil.move(local_shard_path, remote_shard_path)


shard_idx = 0
shard_buffer: list[tuple[NDArray, str]] = []
cum_video_len = 0
for vfn in video_fns:
    if args.dataset == "ego4d":
        print(vfn)
    video_path = os.path.join(RAW_VIDEO_PATH, vfn)
    if video_ext == ".mp4":  # for ego4d
        video = VideoReader(video_path, stream="video", num_threads=32)
        metadata = video.get_metadata()
        fps = metadata["video"]["fps"][0]
        duration = metadata["video"]["duration"][0]
        fi = 0
        while fi < (duration * fps):
            frame = next(video.seek(fi / fps))
            basename = f"{vfn.replace(video_ext, '')}_{fi:06d}"
            image = np.array(transform(frame["data"]))
            # print (image.dtype, image.shape)
            shard_buffer.append((image, basename))
            if len(shard_buffer) % 20 == 0:
                print(f"shard {shard_idx: 04d} frames {len(shard_buffer)}\r", end="")
            if len(shard_buffer) == SAMPLES_PER_SHARD:
                create_shard(shard_idx, shard_buffer)
                shard_buffer = []
                shard_idx += 1
            fi += SUBSAMPLING_RATE

    elif video_ext == ".webm":  # for ssv2
        video, _, info = read_video(video_path, output_format="TCHW")
        video_len = video.size(0)  # for webm, only fps is available; 12 fps for ssv2
        for fi in range(video_len):
            if (fi + cum_video_len) % SUBSAMPLING_RATE == 0:
                frame = video[fi]
                basename = f"{vfn.replace(video_ext, '')}_{fi:06d}"
                image = np.array(transform(frame))
                shard_buffer.append((image, basename))
                if len(shard_buffer) % 20 == 0:
                    print(f"shard {shard_idx: 04d} frames {len(shard_buffer)} - file progress {vfn} - {fi}\r", end="")
                if len(shard_buffer) == SAMPLES_PER_SHARD:
                    create_shard(shard_idx, shard_buffer)
                    shard_buffer = []
                    shard_idx += 1
        cum_video_len += video_len

    elif video_ext == ".tar":  # for epic_kitchen
        tar = tarfile.open(video_path)
        frame_fns = sorted([tinfo.name for tinfo in tar.getmembers() if ".jpg" in tinfo.name])
        video_len = len(frame_fns)
        for fi in range(video_len):
            if (fi + cum_video_len) % SUBSAMPLING_RATE == 0:
                frame_tarf = tar.extractfile(frame_fns[fi])
                if frame_tarf:
                    frame_bytes = frame_tarf.read()
                else:
                    continue
                image = np.array(
                    transform(torch.from_numpy(np.array(Image.open(BytesIO(frame_bytes)))).permute(-1, 0, 1))
                )
                basename = f"{vfn.replace(video_ext, '')}_{fi:06d}"
                shard_buffer.append((image, basename))
                if len(shard_buffer) % 20 == 0:
                    print(f"shard {shard_idx: 04d} frames {len(shard_buffer)} - file progress {vfn} - {fi}\r", end="")
                if len(shard_buffer) == SAMPLES_PER_SHARD:
                    create_shard(shard_idx, shard_buffer)
                    shard_buffer = []
                    shard_idx += 1
        cum_video_len += video_len

# create a shard for final remainings
if len(shard_buffer) > 0:
    create_shard(shard_idx, shard_buffer)
    shard_buffer = []
    shard_idx += 1
