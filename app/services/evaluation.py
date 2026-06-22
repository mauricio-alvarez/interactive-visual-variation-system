from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L").resize((256, 256)))


def pairwise_diversity(image_paths: list[Path]) -> dict[str, float]:
    """Estimate diversity with inverse SSIM across generated variations."""

    if len(image_paths) < 2:
        return {"mean_inverse_ssim": 0.0}

    distances = []
    for left, right in combinations(image_paths, 2):
        score = structural_similarity(_load_gray(left), _load_gray(right), data_range=255)
        distances.append(1.0 - float(score))

    return {
        "mean_inverse_ssim": float(np.mean(distances)),
        "min_inverse_ssim": float(np.min(distances)),
        "max_inverse_ssim": float(np.max(distances)),
    }

