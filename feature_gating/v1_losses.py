# wx:Dynamic gate v3

"""Losses and diagnostics for label-free Dynamic Gate V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class V1LossWeights:
    sufficiency: float = 1.0
    action_invariance: float = 1.0
    mask_invariance: float = 0.1
    budget: float = 10.0


@dataclass
class V1LossOutput:
    total: torch.Tensor
    sufficiency: torch.Tensor
    action_invariance: torch.Tensor
    mask_invariance: torch.Tensor
    budget: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def _require_same_shape(first: torch.Tensor, second: torch.Tensor, name: str) -> None:
    if first.shape != second.shape:
        raise ValueError(
            f"{name} shape mismatch: {tuple(first.shape)} vs {tuple(second.shape)}."
        )


def compute_v1_losses(
    *,
    teacher_action: torch.Tensor,
    gated_original_action: torch.Tensor,
    gated_augmented_action: torch.Tensor,
    original_group_mask: torch.Tensor,
    augmented_group_mask: torch.Tensor,
    target_keep_ratio: float,
    weights: V1LossWeights,
    smooth_l1_beta: float = 1.0,
) -> V1LossOutput:
    """Compute the four label-free V1 objectives using element-wise means."""
    _require_same_shape(
        gated_original_action,
        teacher_action,
        "teacher/original action",
    )
    _require_same_shape(
        gated_original_action,
        gated_augmented_action,
        "original/augmented action",
    )
    _require_same_shape(
        original_group_mask,
        augmented_group_mask,
        "original/augmented group mask",
    )

    if original_group_mask.ndim != 2:
        raise ValueError(
            "group masks must have shape [B, G], got "
            f"{tuple(original_group_mask.shape)}."
        )
    if not (0.0 < target_keep_ratio < 1.0):
        raise ValueError("target_keep_ratio must be in (0, 1).")

    teacher = teacher_action.detach().float()
    original_action = gated_original_action.float()
    augmented_action = gated_augmented_action.float()
    original_mask = original_group_mask.float()
    augmented_mask = augmented_group_mask.float()

    sufficiency = F.smooth_l1_loss(
        original_action,
        teacher,
        reduction="mean",
        beta=float(smooth_l1_beta),
    )
    action_invariance = F.smooth_l1_loss(
        original_action,
        augmented_action,
        reduction="mean",
        beta=float(smooth_l1_beta),
    )
    mask_invariance = torch.mean(torch.abs(original_mask - augmented_mask))

    original_sample_mean = original_mask.mean(dim=-1)
    augmented_sample_mean = augmented_mask.mean(dim=-1)
    target = torch.as_tensor(
        float(target_keep_ratio),
        device=original_mask.device,
        dtype=original_mask.dtype,
    )
    budget = 0.5 * (
        torch.mean((original_sample_mean - target) ** 2)
        + torch.mean((augmented_sample_mean - target) ** 2)
    )

    total = (
        float(weights.sufficiency) * sufficiency
        + float(weights.action_invariance) * action_invariance
        + float(weights.mask_invariance) * mask_invariance
        + float(weights.budget) * budget
    )

    combined_masks = torch.cat([original_mask, augmented_mask], dim=0)
    within_sample_group_std = combined_masks.std(dim=-1, unbiased=False).mean()
    if combined_masks.shape[0] > 1:
        across_sample_mask_std = combined_masks.std(dim=0, unbiased=False).mean()
    else:
        across_sample_mask_std = torch.zeros_like(within_sample_group_std)

    metrics = {
        "mask_mean": combined_masks.mean(),
        "mask_min": combined_masks.min(),
        "mask_max": combined_masks.max(),
        "within_sample_group_std": within_sample_group_std,
        "across_sample_mask_std": across_sample_mask_std,
        "original_mask_mean": original_mask.mean(),
        "augmented_mask_mean": augmented_mask.mean(),
        "weighted_sufficiency": float(weights.sufficiency) * sufficiency,
        "weighted_action_invariance": (
            float(weights.action_invariance) * action_invariance
        ),
        "weighted_mask_invariance": float(weights.mask_invariance) * mask_invariance,
        "weighted_budget": float(weights.budget) * budget,
    }

    return V1LossOutput(
        total=total,
        sufficiency=sufficiency,
        action_invariance=action_invariance,
        mask_invariance=mask_invariance,
        budget=budget,
        metrics=metrics,
    )
