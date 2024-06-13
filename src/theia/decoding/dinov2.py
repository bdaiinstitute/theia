# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


def decode_dinov2(
    features: NDArray, threshold: int | float = -100, interpolation: bool = False, pca: Optional[PCA] = None
) -> tuple[NDArray, PCA]:
    """
    Decode the input `features` in DINOv2 style using PCA.

    Args:
        features (NDArray): features to be decoded, should be in shape [batch_size, num_tokens, latent_dim].
        threshold (int | float): threshold of foreground-background split in PCA visualization.
            Defaults to -100 (all patches are included).
        interpolation (bool): whether interpolate the 16x16 pca map to the original image size.
        pca (Optional[PCA]): if provided, use the provided PCA. This is to keep visualizations stable across samples.

    Returns:
        tuple[NDArray, PCA]: the rendered image of this visualization, in NDArray in size
            [batch_size, height, width, channels] with value ranges [0, 1], and the PCA used in this visualization.
    """
    features = features.numpy()
    batch_size, spatial_size, latent_dim = features.shape
    h = w = int(spatial_size**0.5)

    features = features.reshape(-1, latent_dim)

    if pca is None:
        pca = PCA(n_components=3)
        pca.fit(features)

    pca_features = pca.transform(features)

    # segment using the first component
    bg_mask = pca_features[:, 0] < threshold
    fg_mask = ~bg_mask

    # PCA for only foreground patches
    # pca.fit(features[fg_mask])
    pca_features_fg = pca.transform(features[fg_mask])
    for i in range(3):
        pca_features_fg[:, i] = minmax_scale(pca_features_fg[:, i])

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[bg_mask] = 0
    pca_features_rgb[fg_mask] = pca_features_fg

    pca_features_rgb = pca_features_rgb.reshape(batch_size, h, w, 3)
    if not interpolation:
        H = W = 224
        scale = H // h
        interpolated_pca_features = np.zeros((batch_size, H, W, 3), dtype=pca_features_rgb.dtype)
        for i in range(len(pca_features_rgb)):
            for j in range(h):
                for k in range(w):
                    interpolated_pca_features[i, scale * j : scale * (j + 1), scale * k : scale * (k + 1)] = (
                        pca_features_rgb[i, j, k]
                    )
        pca_features_rgb = interpolated_pca_features
    else:
        pca_features_rgb = np.stack([cv2.resize(p, (224, 224)) for p in pca_features_rgb])
    return pca_features_rgb, pca
