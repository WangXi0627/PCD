# wx:Dynamic gate v2

"""Single-batch data flow for stage-2 dynamic-gate training validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .fixed_noise_teacher import (
    build_initial_action_noise,
    compute_gated_action,
    compute_teacher_action,
    unwrap_pizero_model,
)


@dataclass
class ForwardPair:
    teacher_action: torch.Tensor
    gated_action: torch.Tensor
    initial_action_noise: torch.Tensor
    auxiliary_output: Dict[str, Any]


@dataclass
class TrainingStepResult:
    loss: float
    gate_grad_norm: float
    teacher_requires_grad: bool
    gated_requires_grad: bool
    model_parameters_with_grad: int
    gate_parameters_with_grad: int
    group_mask_mean: Optional[float]
    group_mask_min: Optional[float]
    group_mask_max: Optional[float]


def prepare_pizero_batch_inputs(
    policy,
    batch: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert a collated rollout-query batch into PiZero infer_action inputs.

    The stored proprio is the raw environment eef_pos used during rollout
    collection, so each sample is passed through the original environment
    adapter preprocessing path before batching.
    """
    images = batch["image"]
    instructions = batch["instruction"]
    proprios = batch["proprio"]

    batch_size = len(images)
    if len(instructions) != batch_size or int(proprios.shape[0]) != batch_size:
        raise ValueError(
            "Batch fields have inconsistent sizes: "
            f"images={batch_size}, instructions={len(instructions)}, "
            f"proprios={tuple(proprios.shape)}."
        )

    processed_samples = []
    for index in range(batch_size):
        raw_proprio = proprios[index].detach().cpu().numpy().astype(np.float32)
        processed = policy.env_adapter.preprocess(
            np.asarray(images[index]),
            str(instructions[index]),
            raw_proprio,
        )
        processed_samples.append(processed)

    input_ids = torch.cat(
        [sample["input_ids"] for sample in processed_samples],
        dim=0,
    )
    pixel_values = torch.cat(
        [sample["pixel_values"] for sample in processed_samples],
        dim=0,
    )
    attention_mask = torch.cat(
        [sample["attention_mask"] for sample in processed_samples],
        dim=0,
    )
    normalized_proprios = torch.cat(
        [sample["proprios"] for sample in processed_samples],
        dim=0,
    )

    model = unwrap_pizero_model(policy)
    causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(
            attention_mask,
            dtype=policy.dtype,
        )
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
        causal_mask
    )

    model_inputs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values.to(dtype=policy.dtype),
        "image_text_proprio_mask": image_text_proprio_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": normalized_proprios.to(dtype=policy.dtype),
    }

    return {
        key: value.to(policy.device)
        for key, value in model_inputs.items()
    }


def forward_teacher_and_gate(
    policy,
    visual_gate: nn.Module,
    batch: Mapping[str, Any],
    base_noise_seed: int = 0,
) -> ForwardPair:
    """Run teacher and gated branches with identical inputs and noise."""
    model_inputs = prepare_pizero_batch_inputs(policy, batch)
    model = unwrap_pizero_model(policy)

    initial_action_noise = build_initial_action_noise(
        sample_ids=batch["sample_id"],
        horizon_steps=int(model.horizon_steps),
        action_dim=int(model.action_dim),
        device=policy.device,
        dtype=policy.dtype,
        base_seed=base_noise_seed,
    )

    teacher_action = compute_teacher_action(
        policy,
        model_inputs,
        initial_action_noise,
    )

    gated_action, auxiliary_output = compute_gated_action(
        policy,
        model_inputs,
        visual_gate,
        initial_action_noise,
    )

    return ForwardPair(
        teacher_action=teacher_action,
        gated_action=gated_action,
        initial_action_noise=initial_action_noise,
        auxiliary_output=auxiliary_output,
    )


def action_matching_loss(
    gated_action: torch.Tensor,
    teacher_action: torch.Tensor,
) -> torch.Tensor:
    """Stage-2 smoke loss in the raw PiZero action-output space."""
    if gated_action.shape != teacher_action.shape:
        raise ValueError(
            f"Action shape mismatch: gated={tuple(gated_action.shape)}, "
            f"teacher={tuple(teacher_action.shape)}."
        )
    return F.mse_loss(gated_action.float(), teacher_action.float())


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Encountered a non-finite gate gradient.")
        total += float(gradient.norm().cpu())
    return total


def run_training_step(
    policy,
    visual_gate: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    base_noise_seed: int = 0,
    max_grad_norm: Optional[float] = None,
) -> TrainingStepResult:
    """Execute one optimizer step while updating only visual_gate."""
    optimizer.zero_grad(set_to_none=True)

    pair = forward_teacher_and_gate(
        policy=policy,
        visual_gate=visual_gate,
        batch=batch,
        base_noise_seed=base_noise_seed,
    )

    loss = action_matching_loss(pair.gated_action, pair.teacher_action)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {loss}.")

    loss.backward()

    model = unwrap_pizero_model(policy)
    model_parameters_with_grad = sum(
        1 for parameter in model.parameters() if parameter.grad is not None
    )
    gate_parameters_with_grad = sum(
        1 for parameter in visual_gate.parameters() if parameter.grad is not None
    )

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            visual_gate.parameters(),
            max_norm=float(max_grad_norm),
        )

    gate_grad_norm = _gradient_norm(list(visual_gate.parameters()))
    optimizer.step()

    gate_info = pair.auxiliary_output.get("visual_gate")
    group_mask = None
    if isinstance(gate_info, Mapping):
        candidate = gate_info.get("group_mask")
        if torch.is_tensor(candidate):
            group_mask = candidate.detach().float()

    return TrainingStepResult(
        loss=float(loss.detach().cpu()),
        gate_grad_norm=gate_grad_norm,
        teacher_requires_grad=bool(pair.teacher_action.requires_grad),
        gated_requires_grad=bool(pair.gated_action.requires_grad),
        model_parameters_with_grad=model_parameters_with_grad,
        gate_parameters_with_grad=gate_parameters_with_grad,
        group_mask_mean=(
            None if group_mask is None else float(group_mask.mean().cpu())
        ),
        group_mask_min=(
            None if group_mask is None else float(group_mask.min().cpu())
        ),
        group_mask_max=(
            None if group_mask is None else float(group_mask.max().cpu())
        ),
    )
