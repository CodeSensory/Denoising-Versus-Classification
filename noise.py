import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

import configs


def file_seed(relative_path: str) -> int:
    digest = hashlib.sha256(f"42:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def apply_noise(image01: np.ndarray, relative_path: str) -> np.ndarray:
    rng = np.random.default_rng(file_seed(relative_path))
    lam = np.clip(image01, 0, None) * configs.POISSON_SCALE
    noisy = rng.poisson(lam).astype(np.float32) / np.float32(configs.POISSON_SCALE)
    gaussian = rng.normal(0, configs.GAUSSIAN_NOISE_STD, size=image01.shape)
    noisy = noisy + gaussian.astype(np.float32)
    return np.clip(noisy, 0, 1).astype(np.float32)


def apply_gaussian_filter(image01: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(
        image01.astype(np.float32),
        sigma=configs.GAUSSIAN_SIGMA,
        truncate=configs.GAUSSIAN_TRUNCATE,
    )
    return np.clip(smoothed, 0, 1).astype(np.float32)


def assert_noise_repeatable(image01: np.ndarray, relative_path: str) -> None:
    first = apply_noise(image01, relative_path)
    second = apply_noise(image01, relative_path)
    if np.max(np.abs(first - second)) != 0:
        raise RuntimeError("같은 파일의 노이즈가 두 번 다르게 나왔다")
