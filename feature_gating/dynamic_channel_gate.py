# wx:Dynamic gate v0

"""
Dynamic visual feature gates.

Stage 0 provides two gates:

1. IdentityVisualGate
   Returns visual features unchanged. It is used to verify that the new
   visual-gate interface does not alter the original PiZero behavior.

2. DynamicChannelGate
   Generates a context-conditioned group-wise channel mask from:
   - pooled projected visual features;
   - pooled text embeddings;
   - normalized proprioception.

The gate operates only on projected visual patch features. It does not
modify text tokens.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn


GateAux = Dict[str, torch.Tensor]
GateOutput = Union[torch.Tensor, Tuple[torch.Tensor, GateAux]]


def _probability_to_logit(probability: float) -> float:
    """Convert a probability in (0, 1) into a finite logit."""
    if not (0.0 < probability < 1.0):
        raise ValueError(
            f"probability must be in (0, 1), got {probability}"
        )
    return math.log(probability / (1.0 - probability))


class IdentityVisualGate(nn.Module):
    """
    Exact identity gate used for interface verification.

    It accepts the same keyword arguments as DynamicChannelGate but returns
    image_features unchanged.
    """

    def forward(
        self,
        *,
        image_features: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        proprios: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> GateOutput:
        if image_features.ndim != 3:
            raise ValueError(
                "image_features must have shape [B, N, D], "
                f"got {tuple(image_features.shape)}"
            )

        if not return_aux:
            return image_features

        batch_size, _, feature_dim = image_features.shape
        channel_mask = torch.ones(
            (batch_size, feature_dim),
            device=image_features.device,
            dtype=image_features.dtype,
        )
        group_mask = torch.ones(
            (batch_size, 1),
            device=image_features.device,
            dtype=image_features.dtype,
        )

        return image_features, {
            "group_logits": torch.zeros_like(group_mask),
            "group_mask": group_mask,
            "channel_mask": channel_mask,
            "applied_channel_mask": channel_mask,
        }


class DynamicChannelGate(nn.Module):
    """
    Context-conditioned group-wise channel gate.

    Parameters
    ----------
    feature_dim:
        Last dimension of projected image features. PiZero uses 2048.
    proprio_dim:
        Dimension of the normalized proprio input. PiZero uses 8.
    num_groups:
        Number of channel groups. feature_dim must be divisible by num_groups.
    hidden_dim:
        Hidden dimension of the gate MLP.
    target_keep_ratio:
        Initial and regularization target for the mean soft mask value.
        Must be strictly smaller than 1. Use IdentityVisualGate for an exact
        all-one mask.
    temperature:
        Sigmoid temperature.
    rescale:
        If True, divide the applied mask by target_keep_ratio. Stage 0 should
        normally keep this False so that masking and feature amplification are
        not mixed together.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        proprio_dim: int = 8,
        num_groups: int = 64,
        hidden_dim: int = 512,
        target_keep_ratio: float = 0.95,
        temperature: float = 1.0,
        rescale: bool = False,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive, got {feature_dim}"
            )
        if proprio_dim <= 0:
            raise ValueError(
                f"proprio_dim must be positive, got {proprio_dim}"
            )
        if num_groups <= 0:
            raise ValueError(
                f"num_groups must be positive, got {num_groups}"
            )
        if feature_dim % num_groups != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by "
                f"num_groups={num_groups}"
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        if not (0.0 < target_keep_ratio < 1.0):
            raise ValueError(
                "target_keep_ratio must be in (0, 1). "
                "Use IdentityVisualGate for an exact all-one mask. "
                f"Got {target_keep_ratio}."
            )
        if temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {temperature}"
            )

        self.feature_dim = int(feature_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_groups = int(num_groups)
        self.group_size = self.feature_dim // self.num_groups
        self.hidden_dim = int(hidden_dim)
        self.target_keep_ratio = float(target_keep_ratio)
        self.temperature = float(temperature)
        self.rescale = bool(rescale)

        self.visual_norm = nn.LayerNorm(self.feature_dim)
        self.text_norm = nn.LayerNorm(self.feature_dim)
        self.proprio_norm = nn.LayerNorm(self.proprio_dim)

        gate_input_dim = (
            self.feature_dim
            + self.feature_dim
            + self.proprio_dim
        )

        self.mask_generator = nn.Sequential(
            nn.Linear(gate_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_groups),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the final layer so that every group initially has
        approximately target_keep_ratio.

        The final-layer weights are zero, so the initial gate is stable and
        independent of the input. Training later makes it context-dependent.
        """
        final_layer = self.mask_generator[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(
                "The final mask generator layer must be nn.Linear."
            )

        nn.init.zeros_(final_layer.weight)

        initial_logit = (
            _probability_to_logit(self.target_keep_ratio)
            * self.temperature
        )
        nn.init.constant_(final_layer.bias, initial_logit)

    def _pool_proprio(self, proprios: torch.Tensor) -> torch.Tensor:
        """
        Convert proprio to [B, proprio_dim].

        Supported shapes:
        - [B, proprio_dim]
        - [B, T, proprio_dim]
        """
        if proprios.ndim == 2:
            proprio_context = proprios
        elif proprios.ndim == 3:
            proprio_context = proprios.mean(dim=1)
        else:
            raise ValueError(
                "proprios must have shape [B, P] or [B, T, P], "
                f"got {tuple(proprios.shape)}"
            )

        if proprio_context.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio_dim={self.proprio_dim}, "
                f"got {proprio_context.shape[-1]}"
            )

        return proprio_context

    def _validate_inputs(
        self,
        image_features: torch.Tensor,
        text_context: torch.Tensor,
        proprios: torch.Tensor,
    ) -> None:
        if image_features.ndim != 3:
            raise ValueError(
                "image_features must have shape [B, N, D], "
                f"got {tuple(image_features.shape)}"
            )

        if image_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {image_features.shape[-1]}"
            )

        if text_context.ndim != 2:
            raise ValueError(
                "text_context must have shape [B, D], "
                f"got {tuple(text_context.shape)}"
            )

        if text_context.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected text context dimension {self.feature_dim}, "
                f"got {text_context.shape[-1]}"
            )

        if text_context.shape[0] != image_features.shape[0]:
            raise ValueError(
                "image_features and text_context batch sizes differ: "
                f"{image_features.shape[0]} vs {text_context.shape[0]}"
            )

        if proprios.shape[0] != image_features.shape[0]:
            raise ValueError(
                "image_features and proprios batch sizes differ: "
                f"{image_features.shape[0]} vs {proprios.shape[0]}"
            )

        parameter_device = next(self.parameters()).device
        if image_features.device != parameter_device:
            raise RuntimeError(
                "DynamicChannelGate and image_features must be on the same "
                f"device. Gate device={parameter_device}, "
                f"feature device={image_features.device}."
            )

    def forward(
        self,
        *,
        image_features: torch.Tensor,
        text_context: Optional[torch.Tensor],
        proprios: Optional[torch.Tensor],
        return_aux: bool = False,
    ) -> GateOutput:
        if text_context is None:
            raise ValueError(
                "DynamicChannelGate requires text_context."
            )
        if proprios is None:
            raise ValueError(
                "DynamicChannelGate requires proprios."
            )

        proprio_context = self._pool_proprio(proprios)
        self._validate_inputs(
            image_features=image_features,
            text_context=text_context,
            proprios=proprio_context,
        )

        # The frozen PiZero model runs in bfloat16. Keep the lightweight gate
        # in its own parameter dtype, normally float32, for stable training.
        compute_dtype = self.visual_norm.weight.dtype

        visual_context = image_features.mean(dim=1).to(
            dtype=compute_dtype
        )
        text_context_fp = text_context.to(dtype=compute_dtype)
        proprio_context_fp = proprio_context.to(dtype=compute_dtype)

        normalized_visual = self.visual_norm(visual_context)
        normalized_text = self.text_norm(text_context_fp)
        normalized_proprio = self.proprio_norm(proprio_context_fp)

        gate_input = torch.cat(
            [
                normalized_visual,
                normalized_text,
                normalized_proprio,
            ],
            dim=-1,
        )

        group_logits = self.mask_generator(gate_input)
        group_mask = torch.sigmoid(
            group_logits / self.temperature
        )

        channel_mask = group_mask.repeat_interleave(
            self.group_size,
            dim=-1,
        )

        applied_channel_mask = channel_mask
        if self.rescale:
            applied_channel_mask = (
                applied_channel_mask / self.target_keep_ratio
            )

        feature_mask = applied_channel_mask.to(
            dtype=image_features.dtype
        ).unsqueeze(1)

        gated_features = image_features * feature_mask

        if not return_aux:
            return gated_features

        return gated_features, {
            "group_logits": group_logits,
            "group_mask": group_mask,
            "channel_mask": channel_mask,
            "applied_channel_mask": applied_channel_mask,
        }

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, "
            f"proprio_dim={self.proprio_dim}, "
            f"num_groups={self.num_groups}, "
            f"group_size={self.group_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"target_keep_ratio={self.target_keep_ratio}, "
            f"temperature={self.temperature}, "
            f"rescale={self.rescale}"
        )