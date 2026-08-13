# wx:Dynamic gate v3.5

"""Global channel-redundancy cache and mask-weighted redundancy loss.

This module supports a true channel-level Dynamic Gate by treating every
projected visual feature dimension as one gate group (num_groups=feature_dim).
A cache of globally redundant channel pairs is built once from frozen PiZero
projector features. During gate training, the loss penalizes co-selecting those
redundant dimensions while normalizing by the average co-selection over all
channel pairs.
"""

from __future__ import annotations

import json
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from .teacher_cache import normalize_path, sha256_file


CHANNEL_REDUNDANCY_CACHE_VERSION = "channel_redundancy_pairs_v1"


@dataclass
class ChannelRedundancyLossOutput:
    """Scalar redundancy loss and diagnostics for one mask batch."""

    loss: torch.Tensor
    redundant_pair_coselection_mean: torch.Tensor
    weighted_redundant_pair_coselection_mean: torch.Tensor
    all_pair_coselection_mean: torch.Tensor
    effective_num_channels: torch.Tensor


def save_channel_redundancy_cache(
    cache_dir: str,
    *,
    channel_i: np.ndarray,
    channel_j: np.ndarray,
    weights: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> str:
    """Save one global redundant-channel-pair cache."""
    import os
    import shutil
    import uuid

    cache_dir = normalize_path(cache_dir)
    if osp.exists(cache_dir):
        if not overwrite:
            raise FileExistsError(
                f"Channel redundancy cache already exists: {cache_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(cache_dir)

    channel_i = np.asarray(channel_i, dtype=np.int64)
    channel_j = np.asarray(channel_j, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    channel_mean = np.asarray(channel_mean, dtype=np.float32)
    channel_std = np.asarray(channel_std, dtype=np.float32)

    if channel_i.ndim != 1 or channel_j.ndim != 1 or weights.ndim != 1:
        raise ValueError("channel_i, channel_j, and weights must be 1-D arrays.")
    if not (len(channel_i) == len(channel_j) == len(weights)):
        raise ValueError("Redundancy-pair arrays have inconsistent lengths.")
    if len(channel_i) == 0:
        raise ValueError("At least one redundant channel pair is required.")
    if np.any(channel_i < 0) or np.any(channel_j < 0):
        raise ValueError("Channel indices must be non-negative.")
    if np.any(channel_i >= channel_j):
        raise ValueError("Every cached pair must satisfy channel_i < channel_j.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Redundancy weights must be finite and positive.")
    if channel_mean.ndim != 1 or channel_std.ndim != 1:
        raise ValueError("channel_mean and channel_std must be 1-D arrays.")
    if channel_mean.shape != channel_std.shape:
        raise ValueError("channel_mean/channel_std shape mismatch.")

    feature_dim = int(channel_mean.shape[0])
    if int(channel_j.max()) >= feature_dim:
        raise ValueError(
            f"Pair index exceeds feature_dim={feature_dim}: max={int(channel_j.max())}."
        )

    pair_keys = channel_i.astype(np.int64) * feature_dim + channel_j.astype(np.int64)
    if len(np.unique(pair_keys)) != len(pair_keys):
        raise ValueError("Redundancy cache contains duplicate channel pairs.")

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "cache_version": CHANNEL_REDUNDANCY_CACHE_VERSION,
            "feature_dim": feature_dim,
            "num_pairs": int(len(channel_i)),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
        }
    )

    parent = osp.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)
    temporary_dir = f"{cache_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    os.makedirs(temporary_dir, exist_ok=False)

    try:
        np.save(osp.join(temporary_dir, "channel_i.npy"), channel_i)
        np.save(osp.join(temporary_dir, "channel_j.npy"), channel_j)
        np.save(osp.join(temporary_dir, "weights.npy"), weights)
        np.save(osp.join(temporary_dir, "channel_mean.npy"), channel_mean)
        np.save(osp.join(temporary_dir, "channel_std.npy"), channel_std)
        with open(osp.join(temporary_dir, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump(final_metadata, file, ensure_ascii=False, indent=2)
        os.replace(temporary_dir, cache_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return cache_dir


class ChannelRedundancyCache:
    """Read globally redundant channel pairs and compute their gate loss."""

    def __init__(self, cache_dir: str, *, mmap: bool = True) -> None:
        self.cache_dir = normalize_path(cache_dir)
        metadata_path = osp.join(self.cache_dir, "metadata.json")
        channel_i_path = osp.join(self.cache_dir, "channel_i.npy")
        channel_j_path = osp.join(self.cache_dir, "channel_j.npy")
        weights_path = osp.join(self.cache_dir, "weights.npy")
        channel_mean_path = osp.join(self.cache_dir, "channel_mean.npy")
        channel_std_path = osp.join(self.cache_dir, "channel_std.npy")

        for path in (
            metadata_path,
            channel_i_path,
            channel_j_path,
            weights_path,
            channel_mean_path,
            channel_std_path,
        ):
            if not osp.isfile(path):
                raise FileNotFoundError(path)

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata: Dict[str, Any] = json.load(file)

        version = self.metadata.get("cache_version")
        if version != CHANNEL_REDUNDANCY_CACHE_VERSION:
            raise ValueError(
                f"Unsupported redundancy cache version={version!r}; "
                f"expected {CHANNEL_REDUNDANCY_CACHE_VERSION!r}."
            )

        mmap_mode = "r" if mmap else None
        self.channel_i = np.load(channel_i_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.channel_j = np.load(channel_j_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.weights = np.load(weights_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.channel_mean = np.load(
            channel_mean_path,
            allow_pickle=False,
            mmap_mode=mmap_mode,
        )
        self.channel_std = np.load(
            channel_std_path,
            allow_pickle=False,
            mmap_mode=mmap_mode,
        )

        if not (len(self.channel_i) == len(self.channel_j) == len(self.weights)):
            raise ValueError("Corrupt redundancy cache: pair-array length mismatch.")
        if self.channel_mean.shape != self.channel_std.shape:
            raise ValueError("Corrupt redundancy cache: channel-stat shape mismatch.")

        self.feature_dim = int(self.metadata["feature_dim"])
        if self.channel_mean.shape != (self.feature_dim,):
            raise ValueError(
                "Corrupt redundancy cache: feature_dim does not match channel stats."
            )

        self._tensor_cache: Dict[Tuple[str, Optional[int]], Tuple[torch.Tensor, ...]] = {}

    def __len__(self) -> int:
        return len(self.channel_i)

    def validate(
        self,
        *,
        manifest_path: Optional[str] = None,
        policy_setup: Optional[str] = None,
        split: Optional[str] = None,
        base_checkpoint: Optional[str] = None,
        flow_sampling: Optional[str] = None,
        feature_dim: Optional[int] = None,
    ) -> None:
        """Validate that the cache matches the current frozen policy/data setup."""
        if manifest_path is not None:
            expected_hash = sha256_file(normalize_path(manifest_path))
            actual_hash = str(self.metadata.get("manifest_sha256", ""))
            if actual_hash != expected_hash:
                raise ValueError(
                    "Redundancy cache manifest fingerprint mismatch: "
                    f"cache={actual_hash}, expected={expected_hash}."
                )

        expected_pairs = {
            "policy_setup": policy_setup,
            "split": split,
            "flow_sampling": flow_sampling,
        }
        for key, expected in expected_pairs.items():
            if expected is None:
                continue
            actual = self.metadata.get(key)
            if str(actual) != str(expected):
                raise ValueError(
                    f"Redundancy cache {key} mismatch: "
                    f"cache={actual!r}, expected={expected!r}."
                )

        if base_checkpoint is not None:
            expected = normalize_path(base_checkpoint)
            actual = normalize_path(str(self.metadata.get("base_checkpoint", "")))
            if actual != expected:
                raise ValueError(
                    "Redundancy cache base checkpoint mismatch: "
                    f"cache={actual!r}, expected={expected!r}."
                )

        if feature_dim is not None and int(feature_dim) != self.feature_dim:
            raise ValueError(
                f"Redundancy cache feature_dim mismatch: "
                f"cache={self.feature_dim}, expected={int(feature_dim)}."
            )

    def _get_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        key = (device.type, device.index)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        tensors = (
            torch.as_tensor(np.asarray(self.channel_i), dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(self.channel_j), dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(self.weights), dtype=torch.float32, device=device),
        )
        self._tensor_cache[key] = tensors
        return tensors

    def compute_loss(
        self,
        channel_mask: torch.Tensor,
        *,
        eps: float = 1e-8,
    ) -> ChannelRedundancyLossOutput:
        """
        Penalize co-selection of globally redundant feature dimensions.

        The numerator measures mean weighted co-selection over cached redundant
        pairs. The denominator is the mean co-selection over *all* unordered
        channel pairs. This makes the objective insensitive to uniform scaling
        of the whole mask, while the separate budget loss fixes the desired
        average keep ratio.
        """
        if channel_mask.ndim != 2:
            raise ValueError(
                f"channel_mask must have shape [B,D], got {tuple(channel_mask.shape)}."
            )
        if channel_mask.shape[-1] != self.feature_dim:
            raise ValueError(
                "Dimension-level redundancy requires one mask value per channel: "
                f"mask D={channel_mask.shape[-1]}, cache D={self.feature_dim}."
            )
        if self.feature_dim < 2:
            raise ValueError("feature_dim must be at least 2.")

        mask = channel_mask.float()
        pair_i, pair_j, pair_weight = self._get_tensors(mask.device)

        pair_product = mask.index_select(1, pair_i) * mask.index_select(1, pair_j)
        weighted_pair_product = pair_product * pair_weight.unsqueeze(0)

        redundant_pair_mean = pair_product.mean(dim=1)
        weighted_redundant_pair_mean = weighted_pair_product.mean(dim=1)

        mask_sum = mask.sum(dim=1)
        mask_square_sum = mask.square().sum(dim=1)
        num_all_unordered_pairs = float(self.feature_dim * (self.feature_dim - 1) / 2)
        all_pair_sum = 0.5 * (mask_sum.square() - mask_square_sum)
        all_pair_mean = all_pair_sum / num_all_unordered_pairs

        loss_per_sample = weighted_redundant_pair_mean / all_pair_mean.clamp_min(eps)

        effective_num_channels = (
            mask_sum.square() / mask_square_sum.clamp_min(eps)
        )

        return ChannelRedundancyLossOutput(
            loss=loss_per_sample.mean(),
            redundant_pair_coselection_mean=redundant_pair_mean.mean(),
            weighted_redundant_pair_coselection_mean=(
                weighted_redundant_pair_mean.mean()
            ),
            all_pair_coselection_mean=all_pair_mean.mean(),
            effective_num_channels=effective_num_channels.mean(),
        )
