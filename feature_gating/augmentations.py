# wx:Dynamic gate v3

"""Mild, geometry-preserving image augmentations for Dynamic Gate V1.

The augmentation pipeline intentionally avoids flips, crops, translations,
cutout, and other operations that may change robot-control geometry.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class MildAugmentationConfig:
    brightness_min: float = 0.90
    brightness_max: float = 1.10
    contrast_min: float = 0.90
    contrast_max: float = 1.10
    gamma_min: float = 0.90
    gamma_max: float = 1.10
    gaussian_noise_std_max: float = 3.0
    blur_radius_max: float = 0.60
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 100
    brightness_probability: float = 0.80
    contrast_probability: float = 0.80
    gamma_probability: float = 0.50
    gaussian_noise_probability: float = 0.50
    blur_probability: float = 0.30
    jpeg_probability: float = 0.30

    def validate(self) -> None:
        if not (0.0 < self.brightness_min <= self.brightness_max):
            raise ValueError("Invalid brightness range.")
        if not (0.0 < self.contrast_min <= self.contrast_max):
            raise ValueError("Invalid contrast range.")
        if not (0.0 < self.gamma_min <= self.gamma_max):
            raise ValueError("Invalid gamma range.")
        if self.gaussian_noise_std_max < 0.0:
            raise ValueError("gaussian_noise_std_max must be non-negative.")
        if self.blur_radius_max < 0.0:
            raise ValueError("blur_radius_max must be non-negative.")
        if not (1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100):
            raise ValueError("Invalid JPEG quality range.")

        probabilities = (
            self.brightness_probability,
            self.contrast_probability,
            self.gamma_probability,
            self.gaussian_noise_probability,
            self.blur_probability,
            self.jpeg_probability,
        )
        if any(not (0.0 <= value <= 1.0) for value in probabilities):
            raise ValueError("Augmentation probabilities must be in [0, 1].")


def stable_augmentation_seed(
    sample_id: str,
    *,
    base_seed: int,
    augmentation_step: int,
    view_name: str = "augmented",
) -> int:
    """Return a process-stable augmentation seed for one query image."""
    token = (
        f"{int(base_seed)}|{int(augmentation_step)}|"
        f"{str(view_name)}|{str(sample_id)}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {array.dtype}.")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected image shape [H, W, 3], got {array.shape}."
        )
    return array


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    corrected = np.power(np.clip(array, 0.0, 1.0), gamma)
    corrected = np.clip(np.round(corrected * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _apply_gaussian_noise(
    image: Image.Image,
    *,
    standard_deviation: float,
    numpy_rng: np.random.Generator,
) -> Image.Image:
    if standard_deviation <= 0.0:
        return image

    array = np.asarray(image, dtype=np.float32)
    noise = numpy_rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=array.shape,
    )
    noisy = np.clip(np.round(array + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


class MildVisualAugmenter:
    """Deterministic-per-sample mild appearance augmentation.

    Training should vary ``augmentation_step`` across optimization steps.
    Validation should keep it fixed, typically at zero.
    """

    def __init__(
        self,
        config: MildAugmentationConfig | None = None,
        *,
        base_seed: int = 0,
    ) -> None:
        self.config = config or MildAugmentationConfig()
        self.config.validate()
        self.base_seed = int(base_seed)

    def augment_one(
        self,
        image: np.ndarray,
        sample_id: str,
        *,
        augmentation_step: int,
        view_name: str = "augmented",
    ) -> np.ndarray:
        source = _ensure_uint8_rgb(image)
        seed = stable_augmentation_seed(
            sample_id,
            base_seed=self.base_seed,
            augmentation_step=augmentation_step,
            view_name=view_name,
        )
        python_rng = random.Random(seed)
        numpy_rng = np.random.default_rng(seed)

        output = Image.fromarray(source, mode="RGB")
        config = self.config

        if python_rng.random() < config.brightness_probability:
            factor = python_rng.uniform(
                config.brightness_min,
                config.brightness_max,
            )
            output = ImageEnhance.Brightness(output).enhance(factor)

        if python_rng.random() < config.contrast_probability:
            factor = python_rng.uniform(
                config.contrast_min,
                config.contrast_max,
            )
            output = ImageEnhance.Contrast(output).enhance(factor)

        if python_rng.random() < config.gamma_probability:
            gamma = python_rng.uniform(config.gamma_min, config.gamma_max)
            output = _apply_gamma(output, gamma)

        if python_rng.random() < config.gaussian_noise_probability:
            standard_deviation = python_rng.uniform(
                0.0,
                config.gaussian_noise_std_max,
            )
            output = _apply_gaussian_noise(
                output,
                standard_deviation=standard_deviation,
                numpy_rng=numpy_rng,
            )

        if python_rng.random() < config.blur_probability:
            radius = python_rng.uniform(0.0, config.blur_radius_max)
            if radius > 0.0:
                output = output.filter(ImageFilter.GaussianBlur(radius=radius))

        if python_rng.random() < config.jpeg_probability:
            quality = python_rng.randint(
                config.jpeg_quality_min,
                config.jpeg_quality_max,
            )
            output = _apply_jpeg(output, quality)

        result = np.asarray(output, dtype=np.uint8).copy()
        if result.shape != source.shape:
            raise RuntimeError(
                "Appearance augmentation changed image geometry: "
                f"before={source.shape}, after={result.shape}."
            )
        return result

    def augment_batch(
        self,
        images: Sequence[np.ndarray],
        sample_ids: Sequence[str],
        *,
        augmentation_step: int,
        view_name: str = "augmented",
    ) -> List[np.ndarray]:
        if len(images) != len(sample_ids):
            raise ValueError(
                f"images/sample_ids length mismatch: "
                f"{len(images)} vs {len(sample_ids)}."
            )
        return [
            self.augment_one(
                image,
                sample_id,
                augmentation_step=augmentation_step,
                view_name=view_name,
            )
            for image, sample_id in zip(images, sample_ids)
        ]
