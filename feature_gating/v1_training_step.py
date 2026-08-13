# wx:Dynamic gate v3

"""Forward and optimization utilities for label-free Dynamic Gate V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .augmentations import MildVisualAugmenter
from .fixed_noise_teacher import (
    build_initial_action_noise,
    compute_gated_action,
    unwrap_pizero_model,
)
from .teacher_cache import TeacherActionCache
from .training_step import prepare_pizero_batch_inputs
from .v1_losses import V1LossOutput, V1LossWeights, compute_v1_losses


@dataclass
class V1ForwardOutput:
    losses: V1LossOutput
    teacher_action: torch.Tensor
    gated_original_action: torch.Tensor
    gated_augmented_action: torch.Tensor
    original_group_mask: torch.Tensor
    augmented_group_mask: torch.Tensor
    initial_action_noise: torch.Tensor


def _extract_group_mask(auxiliary_output: Mapping[str, Any]) -> torch.Tensor:
    gate_info = auxiliary_output.get("visual_gate")
    if not isinstance(gate_info, Mapping):
        raise KeyError("Gated PiZero output is missing visual_gate diagnostics.")
    group_mask = gate_info.get("group_mask")
    if not torch.is_tensor(group_mask):
        raise TypeError("visual_gate.group_mask must be a Tensor.")
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
        view_name="v1_augmented",
    )
    return augmented


def forward_v1_batch(
    *,
    policy,
    visual_gate: nn.Module,
    batch: Mapping[str, Any],
    teacher_cache: TeacherActionCache,
    augmenter: MildVisualAugmenter,
    base_noise_seed: int,
    augmentation_step: int,
    target_keep_ratio: float,
    loss_weights: V1LossWeights,
    smooth_l1_beta: float = 1.0,
) -> V1ForwardOutput:
    """Run teacher-cache, original-gated, and augmented-gated branches."""
    original_inputs = prepare_pizero_batch_inputs(policy, batch)
    augmented_batch = build_augmented_batch(
        batch,
        augmenter,
        augmentation_step=augmentation_step,
    )
    augmented_inputs = prepare_pizero_batch_inputs(policy, augmented_batch)

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
    gated_augmented_action, augmented_aux = compute_gated_action(
        policy,
        augmented_inputs,
        visual_gate,
        initial_action_noise,
    )

    original_group_mask = _extract_group_mask(original_aux)
    augmented_group_mask = _extract_group_mask(augmented_aux)

    losses = compute_v1_losses(
        teacher_action=teacher_action,
        gated_original_action=gated_original_action,
        gated_augmented_action=gated_augmented_action,
        original_group_mask=original_group_mask,
        augmented_group_mask=augmented_group_mask,
        target_keep_ratio=target_keep_ratio,
        weights=loss_weights,
        smooth_l1_beta=smooth_l1_beta,
    )

    return V1ForwardOutput(
        losses=losses,
        teacher_action=teacher_action,
        gated_original_action=gated_original_action,
        gated_augmented_action=gated_augmented_action,
        original_group_mask=original_group_mask,
        augmented_group_mask=augmented_group_mask,
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


def metrics_to_float(output: V1ForwardOutput) -> Dict[str, float]:
    losses = output.losses
    metrics = {
        "total_loss": float(losses.total.detach().cpu()),
        "sufficiency_loss": float(losses.sufficiency.detach().cpu()),
        "action_invariance_loss": float(
            losses.action_invariance.detach().cpu()
        ),
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
