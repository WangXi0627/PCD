# wx:Dynamic gate v3.5

"""Training step for channel-level Dynamic Gate with redundancy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .augmentations import MildVisualAugmenter
from .channel_redundancy import ChannelRedundancyCache
from .fixed_noise_teacher import (
    build_initial_action_noise,
    compute_gated_action,
    unwrap_pizero_model,
)
from .teacher_cache import TeacherActionCache
from .training_step import prepare_pizero_batch_inputs


@dataclass(frozen=True)
class RedundancyLossWeights:
    sufficiency: float = 1.0
    redundancy: float = 0.0
    action_invariance: float = 0.0
    mask_invariance: float = 0.0
    budget: float = 10.0


@dataclass
class RedundancyLossOutput:
    total: torch.Tensor
    sufficiency: torch.Tensor
    redundancy: torch.Tensor
    action_invariance: torch.Tensor
    mask_invariance: torch.Tensor
    budget: torch.Tensor
    metrics: Dict[str, torch.Tensor]


@dataclass
class RedundancyForwardOutput:
    losses: RedundancyLossOutput
    teacher_action: torch.Tensor
    gated_original_action: torch.Tensor
    original_channel_mask: torch.Tensor
    gated_augmented_action: Optional[torch.Tensor]
    augmented_channel_mask: Optional[torch.Tensor]
    initial_action_noise: torch.Tensor


def _extract_channel_mask(auxiliary_output: Mapping[str, Any]) -> torch.Tensor:
    gate_info = auxiliary_output.get("visual_gate")
    if not isinstance(gate_info, Mapping):
        raise KeyError("Gated PiZero output is missing visual_gate diagnostics.")

    # With num_groups == feature_dim, group_mask is already one value/channel.
    group_mask = gate_info.get("group_mask")
    if not torch.is_tensor(group_mask):
        raise TypeError("visual_gate.group_mask must be a Tensor.")
    if group_mask.ndim != 2:
        raise ValueError(
            f"visual_gate.group_mask must be [B,D], got {tuple(group_mask.shape)}."
        )
    return group_mask


def build_augmented_batch(
    batch: Mapping[str, Any],
    augmenter: MildVisualAugmenter,
    *,
    augmentation_step: int,
) -> Dict[str, Any]:
    augmented = dict(batch)
    augmented["image"] = augmenter.augment_batch(
        batch["image"],
        batch["sample_id"],
        augmentation_step=augmentation_step,
        view_name="channel_redundancy_augmented",
    )
    return augmented


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_redundancy_losses(
    *,
    teacher_action: torch.Tensor,
    gated_original_action: torch.Tensor,
    original_channel_mask: torch.Tensor,
    gated_augmented_action: Optional[torch.Tensor],
    augmented_channel_mask: Optional[torch.Tensor],
    redundancy_cache: Optional[ChannelRedundancyCache],
    target_keep_ratio: float,
    weights: RedundancyLossWeights,
    smooth_l1_beta: float,
) -> RedundancyLossOutput:
    if gated_original_action.shape != teacher_action.shape:
        raise ValueError(
            "Teacher/original action shape mismatch: "
            f"{tuple(teacher_action.shape)} vs {tuple(gated_original_action.shape)}."
        )
    if original_channel_mask.ndim != 2:
        raise ValueError("original_channel_mask must have shape [B,D].")
    if not (0.0 < target_keep_ratio < 1.0):
        raise ValueError("target_keep_ratio must be in (0,1).")

    teacher = teacher_action.detach().float()
    original_action = gated_original_action.float()
    original_mask = original_channel_mask.float()

    sufficiency = F.smooth_l1_loss(
        original_action,
        teacher,
        reduction="mean",
        beta=float(smooth_l1_beta),
    )

    use_augmentation = (
        gated_augmented_action is not None
        or augmented_channel_mask is not None
    )
    if use_augmentation:
        if gated_augmented_action is None or augmented_channel_mask is None:
            raise ValueError(
                "gated_augmented_action and augmented_channel_mask must both be provided."
            )
        if gated_augmented_action.shape != gated_original_action.shape:
            raise ValueError("Original/augmented action shape mismatch.")
        if augmented_channel_mask.shape != original_channel_mask.shape:
            raise ValueError("Original/augmented mask shape mismatch.")

        augmented_action = gated_augmented_action.float()
        augmented_mask = augmented_channel_mask.float()
        action_invariance = F.smooth_l1_loss(
            original_action,
            augmented_action,
            reduction="mean",
            beta=float(smooth_l1_beta),
        )
        mask_invariance = torch.mean(torch.abs(original_mask - augmented_mask))
    else:
        augmented_mask = None
        action_invariance = _zero(original_action)
        mask_invariance = _zero(original_action)

    target = torch.as_tensor(
        float(target_keep_ratio),
        device=original_mask.device,
        dtype=original_mask.dtype,
    )
    original_budget = torch.mean((original_mask.mean(dim=-1) - target).square())
    if augmented_mask is None:
        budget = original_budget
    else:
        augmented_budget = torch.mean((augmented_mask.mean(dim=-1) - target).square())
        budget = 0.5 * (original_budget + augmented_budget)

    if float(weights.redundancy) > 0.0:
        if redundancy_cache is None:
            raise ValueError(
                "A ChannelRedundancyCache is required when redundancy weight > 0."
            )
        original_red = redundancy_cache.compute_loss(original_mask)
        if augmented_mask is None:
            redundancy = original_red.loss
            red_pair_coselection = original_red.redundant_pair_coselection_mean
            weighted_red_pair_coselection = (
                original_red.weighted_redundant_pair_coselection_mean
            )
            all_pair_coselection = original_red.all_pair_coselection_mean
            effective_num_channels = original_red.effective_num_channels
        else:
            augmented_red = redundancy_cache.compute_loss(augmented_mask)
            redundancy = 0.5 * (original_red.loss + augmented_red.loss)
            red_pair_coselection = 0.5 * (
                original_red.redundant_pair_coselection_mean
                + augmented_red.redundant_pair_coselection_mean
            )
            weighted_red_pair_coselection = 0.5 * (
                original_red.weighted_redundant_pair_coselection_mean
                + augmented_red.weighted_redundant_pair_coselection_mean
            )
            all_pair_coselection = 0.5 * (
                original_red.all_pair_coselection_mean
                + augmented_red.all_pair_coselection_mean
            )
            effective_num_channels = 0.5 * (
                original_red.effective_num_channels
                + augmented_red.effective_num_channels
            )
    else:
        redundancy = _zero(original_action)
        red_pair_coselection = _zero(original_action)
        weighted_red_pair_coselection = _zero(original_action)
        all_pair_coselection = _zero(original_action)
        mask_sum = original_mask.sum(dim=-1)
        effective_num_channels = (
            mask_sum.square()
            / original_mask.square().sum(dim=-1).clamp_min(1e-8)
        ).mean()

    total = (
        float(weights.sufficiency) * sufficiency
        + float(weights.redundancy) * redundancy
        + float(weights.action_invariance) * action_invariance
        + float(weights.mask_invariance) * mask_invariance
        + float(weights.budget) * budget
    )

    all_masks = (
        original_mask
        if augmented_mask is None
        else torch.cat([original_mask, augmented_mask], dim=0)
    )
    within_sample_channel_std = all_masks.std(dim=-1, unbiased=False).mean()
    if all_masks.shape[0] > 1:
        across_sample_mask_std = all_masks.std(dim=0, unbiased=False).mean()
    else:
        across_sample_mask_std = torch.zeros_like(within_sample_channel_std)

    metrics = {
        "mask_mean": all_masks.mean(),
        "mask_min": all_masks.min(),
        "mask_max": all_masks.max(),
        "within_sample_channel_std": within_sample_channel_std,
        "across_sample_mask_std": across_sample_mask_std,
        "effective_num_channels": effective_num_channels,
        "redundant_pair_coselection_mean": red_pair_coselection,
        "weighted_redundant_pair_coselection_mean": (
            weighted_red_pair_coselection
        ),
        "all_pair_coselection_mean": all_pair_coselection,
        "weighted_sufficiency": float(weights.sufficiency) * sufficiency,
        "weighted_redundancy": float(weights.redundancy) * redundancy,
        "weighted_action_invariance": (
            float(weights.action_invariance) * action_invariance
        ),
        "weighted_mask_invariance": float(weights.mask_invariance) * mask_invariance,
        "weighted_budget": float(weights.budget) * budget,
        "used_augmentation_branch": torch.as_tensor(
            float(use_augmentation),
            device=original_mask.device,
        ),
    }

    return RedundancyLossOutput(
        total=total,
        sufficiency=sufficiency,
        redundancy=redundancy,
        action_invariance=action_invariance,
        mask_invariance=mask_invariance,
        budget=budget,
        metrics=metrics,
    )


def forward_redundancy_batch(
    *,
    policy,
    visual_gate: nn.Module,
    batch: Mapping[str, Any],
    teacher_cache: TeacherActionCache,
    redundancy_cache: Optional[ChannelRedundancyCache],
    augmenter: MildVisualAugmenter,
    base_noise_seed: int,
    augmentation_step: int,
    target_keep_ratio: float,
    loss_weights: RedundancyLossWeights,
    smooth_l1_beta: float = 1.0,
) -> RedundancyForwardOutput:
    """Run only the branches required by the active loss combination."""
    original_inputs = prepare_pizero_batch_inputs(policy, batch)

    model = unwrap_pizero_model(policy)
    initial_action_noise = build_initial_action_noise(
        sample_ids=batch["sample_id"],
        horizon_steps=int(model.horizon_steps),
        action_dim=int(model.action_dim),
        device=policy.device,
        dtype=policy.dtype,
        base_seed=int(base_noise_seed),
    )

    teacher_action = teacher_cache.get_tensor(
        batch["sample_id"],
        device=policy.device,
        dtype=policy.dtype,
    ).detach()

    gated_original_action, original_aux = compute_gated_action(
        policy,
        original_inputs,
        visual_gate,
        initial_action_noise,
    )
    original_channel_mask = _extract_channel_mask(original_aux)

    use_augmentation_branch = (
        float(loss_weights.action_invariance) > 0.0
        or float(loss_weights.mask_invariance) > 0.0
    )

    gated_augmented_action: Optional[torch.Tensor] = None
    augmented_channel_mask: Optional[torch.Tensor] = None

    if use_augmentation_branch:
        augmented_batch = build_augmented_batch(
            batch,
            augmenter,
            augmentation_step=augmentation_step,
        )
        augmented_inputs = prepare_pizero_batch_inputs(policy, augmented_batch)
        gated_augmented_action, augmented_aux = compute_gated_action(
            policy,
            augmented_inputs,
            visual_gate,
            initial_action_noise,
        )
        augmented_channel_mask = _extract_channel_mask(augmented_aux)

    losses = compute_redundancy_losses(
        teacher_action=teacher_action,
        gated_original_action=gated_original_action,
        original_channel_mask=original_channel_mask,
        gated_augmented_action=gated_augmented_action,
        augmented_channel_mask=augmented_channel_mask,
        redundancy_cache=redundancy_cache,
        target_keep_ratio=target_keep_ratio,
        weights=loss_weights,
        smooth_l1_beta=smooth_l1_beta,
    )

    return RedundancyForwardOutput(
        losses=losses,
        teacher_action=teacher_action,
        gated_original_action=gated_original_action,
        original_channel_mask=original_channel_mask,
        gated_augmented_action=gated_augmented_action,
        augmented_channel_mask=augmented_channel_mask,
        initial_action_noise=initial_action_noise,
    )


def gate_gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Dynamic gate contains a non-finite gradient.")
        squared_norm += float(torch.sum(gradient * gradient).cpu())
    return squared_norm ** 0.5


def verify_frozen_policy_has_no_grad(policy) -> None:
    model = unwrap_pizero_model(policy)
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if names:
        raise RuntimeError(
            "Frozen PiZero unexpectedly received gradients: "
            f"{names[:10]}"
        )


def metrics_to_float(output: RedundancyForwardOutput) -> Dict[str, float]:
    losses = output.losses
    metrics = {
        "total_loss": float(losses.total.detach().cpu()),
        "sufficiency_loss": float(losses.sufficiency.detach().cpu()),
        "redundancy_loss": float(losses.redundancy.detach().cpu()),
        "action_invariance_loss": float(losses.action_invariance.detach().cpu()),
        "mask_invariance_loss": float(losses.mask_invariance.detach().cpu()),
        "budget_loss": float(losses.budget.detach().cpu()),
    }
    metrics.update(
        {
            key: float(value.detach().cpu())
            for key, value in losses.metrics.items()
        }
    )
    return metrics
