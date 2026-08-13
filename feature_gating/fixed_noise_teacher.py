# wx:Dynamic gate v2

"""Deterministic per-sample action noise and frozen PiZero teacher helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Sequence

import torch
from torch import nn


MAX_TORCH_SEED = 2**63 - 1


def stable_noise_seed(sample_id: str, base_seed: int = 0) -> int:
    """Create a process-stable seed from a query sample ID."""
    token = f"{int(base_seed)}|{str(sample_id)}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % MAX_TORCH_SEED


def build_initial_action_noise(
    sample_ids: Sequence[str],
    horizon_steps: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base_seed: int = 0,
) -> torch.Tensor:
    """
    Build one independent deterministic noise tensor per query.

    Noise is generated on CPU in float32 so it does not depend on the global
    RNG state, batch order, batch size, or CUDA generator state. The final
    tensor is then moved to the model device and dtype.
    """
    if not sample_ids:
        raise ValueError("sample_ids must not be empty.")
    if horizon_steps <= 0 or action_dim <= 0:
        raise ValueError(
            f"Invalid action shape: horizon_steps={horizon_steps}, action_dim={action_dim}."
        )

    noise_rows = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_noise_seed(sample_id, base_seed=base_seed))
        noise = torch.randn(
            (1, int(horizon_steps), int(action_dim)),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noise_rows.append(noise)

    return torch.cat(noise_rows, dim=0).to(device=device, dtype=dtype)


def unwrap_pizero_model(policy_or_model) -> nn.Module:
    """Return the underlying PiZero module from a policy or compiled wrapper."""
    model = getattr(policy_or_model, "model", policy_or_model)
    return getattr(model, "_orig_mod", model)


def compute_teacher_action(
    policy_or_model,
    model_inputs: Mapping[str, torch.Tensor],
    initial_action_noise: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen, ungated PiZero branch under torch.no_grad()."""
    model = unwrap_pizero_model(policy_or_model)

    with torch.no_grad():
        action = model.infer_action(
            **dict(model_inputs),
            visual_gate=None,
            initial_action_noise=initial_action_noise,
            return_aux=False,
        )

    return action.detach()


def compute_gated_action(
    policy_or_model,
    model_inputs: Mapping[str, torch.Tensor],
    visual_gate: nn.Module,
    initial_action_noise: torch.Tensor,
):
    """Run the gated branch while preserving gradients to visual_gate."""
    model = unwrap_pizero_model(policy_or_model)
    return model.infer_action(
        **dict(model_inputs),
        visual_gate=visual_gate,
        initial_action_noise=initial_action_noise,
        return_aux=True,
    )
