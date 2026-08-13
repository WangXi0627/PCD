# wx:Dynamic gate v1

"""
PiZero dynamic visual-gate inference wrapper.

This wrapper is used for stage-1 closed-loop integration verification.

Supported modes
---------------
identity:
    Use IdentityVisualGate. The projected visual features are returned
    unchanged. This mode is used for lossless closed-loop regression.

dynamic:
    Use DynamicChannelGate. The gate may be initialized from scratch or
    loaded from an offline-trained checkpoint.

Important
---------
- PiZero remains completely frozen.
- No optimizer is created.
- No backward pass or test-time adaptation is performed.
- Each policy query invokes PiZero action inference exactly once.
- torch.compile is disabled for the stage-1 external-gate path.
"""

from __future__ import annotations

import os.path as osp
from typing import Any, Dict, Mapping, Optional

import torch

from feature_gating import DynamicChannelGate, IdentityVisualGate
from simpler_env.policies.pizero.pizero_model import PiZeroInference


def _normalize_optional_path(path: Optional[str]) -> Optional[str]:
    """Normalize command-line representations of an optional path."""
    if path is None:
        return None

    normalized = str(path).strip()

    if normalized.lower() in {"", "none", "null"}:
        return None

    return normalized


def _strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Remove common wrappers from checkpoint parameter names.

    Supported examples:
    - module.mask_generator.0.weight
    - dynamic_gate.mask_generator.0.weight
    - visual_gate.mask_generator.0.weight
    """
    prefixes = (
        "module.",
        "dynamic_gate.",
        "visual_gate.",
        "gate.",
    )

    cleaned: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = str(key)

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def _extract_gate_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:
    """
    Extract a gate state dict from common checkpoint layouts.

    Supported layouts:
    {
        "gate_state_dict": ...
    }

    {
        "state_dict": ...
    }

    {
        "model": ...
    }

    or a bare state dict.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Dynamic-gate checkpoint must be a mapping, "
            f"got {type(checkpoint)}."
        )

    for key in (
        "gate_state_dict",
        "dynamic_gate_state_dict",
        "visual_gate_state_dict",
        "state_dict",
        "model",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, Mapping):
            tensor_candidate = {
                str(name): value
                for name, value in candidate.items()
                if torch.is_tensor(value)
            }

            if tensor_candidate:
                return _strip_state_dict_prefix(tensor_candidate)

    bare_state_dict = {
        str(name): value
        for name, value in checkpoint.items()
        if torch.is_tensor(value)
    }

    if bare_state_dict:
        return _strip_state_dict_prefix(bare_state_dict)

    raise ValueError(
        "Cannot find a tensor state dict in the dynamic-gate checkpoint."
    )


class PiZeroDynamicGateInference(PiZeroInference):
    """
    Single-forward dynamic visual-gate inference for PiZero.

    Parameters
    ----------
    dynamic_feature_gate:
        Registration switch used by properties.py and get_policy().
    dynamic_gate_mode:
        "identity" or "dynamic".
    dynamic_gate_checkpoint:
        Optional checkpoint for DynamicChannelGate. It must be omitted in
        identity mode.
    dynamic_gate_num_groups:
        Number of group-wise channel gates.
    dynamic_gate_hidden_dim:
        Hidden size of the gate generator.
    dynamic_gate_target_keep_ratio:
        Initial mean soft-mask value for an untrained dynamic gate.
    dynamic_gate_temperature:
        Sigmoid temperature.
    dynamic_gate_rescale:
        Whether to divide the applied mask by target_keep_ratio.
    dynamic_gate_checkpoint_strict:
        Whether checkpoint loading uses strict=True.
    dynamic_gate_verbose:
        Print gate diagnostics during inference.
    dynamic_gate_log_every:
        Print diagnostics every N policy queries.
    """

    VALID_GATE_MODES = {
        "identity",
        "dynamic",
    }

    def __init__(
        self,
        dynamic_feature_gate: bool = False,
        dynamic_gate_mode: str = "identity",
        dynamic_gate_checkpoint: Optional[str] = None,
        dynamic_gate_num_groups: int = 64,
        dynamic_gate_hidden_dim: int = 512,
        dynamic_gate_target_keep_ratio: float = 0.99,
        dynamic_gate_temperature: float = 1.0,
        dynamic_gate_rescale: bool = False,
        dynamic_gate_checkpoint_strict: bool = True,
        dynamic_gate_verbose: bool = False,
        dynamic_gate_log_every: int = 1,
        *args,
        **kwargs,
    ) -> None:
        self.dynamic_feature_gate = bool(dynamic_feature_gate)

        normalized_mode = str(dynamic_gate_mode).strip().lower()
        if normalized_mode not in self.VALID_GATE_MODES:
            raise ValueError(
                f"Unknown dynamic_gate_mode={dynamic_gate_mode!r}. "
                f"Available modes: {sorted(self.VALID_GATE_MODES)}"
            )

        self.dynamic_gate_mode = normalized_mode
        self.dynamic_gate_checkpoint = _normalize_optional_path(
            dynamic_gate_checkpoint
        )

        self.dynamic_gate_num_groups = int(
            dynamic_gate_num_groups
        )
        self.dynamic_gate_hidden_dim = int(
            dynamic_gate_hidden_dim
        )
        self.dynamic_gate_target_keep_ratio = float(
            dynamic_gate_target_keep_ratio
        )
        self.dynamic_gate_temperature = float(
            dynamic_gate_temperature
        )
        self.dynamic_gate_rescale = bool(
            dynamic_gate_rescale
        )
        self.dynamic_gate_checkpoint_strict = bool(
            dynamic_gate_checkpoint_strict
        )
        self.dynamic_gate_verbose = bool(
            dynamic_gate_verbose
        )
        self.dynamic_gate_log_every = int(
            dynamic_gate_log_every
        )

        if self.dynamic_gate_log_every <= 0:
            raise ValueError(
                "dynamic_gate_log_every must be positive, "
                f"got {self.dynamic_gate_log_every}."
            )

        requested_compile = bool(
            kwargs.get("use_torch_compile", False)
        )

        # Stage 0 deliberately disables torch.compile for an external gate.
        # Keep the same restriction during stage-1 closed-loop verification.
        kwargs["use_torch_compile"] = False

        super().__init__(*args, **kwargs)

        self._requested_torch_compile = requested_compile
        self._query_index = 0
        self.last_gate_diagnostics: Optional[
            Dict[str, Any]
        ] = None

        self.visual_gate = self._build_visual_gate()

        if self.dynamic_gate_mode == "dynamic":
            self._load_dynamic_gate_checkpoint_if_needed()

        self.visual_gate.to(device=self.device)
        self.visual_gate.eval()

        # Stage 1 is inference only. Explicitly freeze the gate as well.
        for parameter in self.visual_gate.parameters():
            parameter.requires_grad = False

        if self.dynamic_gate_verbose:
            self._print_initialization_summary()

    def _build_visual_gate(self):
        if self.dynamic_gate_mode == "identity":
            if self.dynamic_gate_checkpoint is not None:
                raise ValueError(
                    "dynamic_gate_checkpoint must be omitted when "
                    "dynamic_gate_mode='identity'."
                )

            return IdentityVisualGate()

        feature_dim = int(
            self.model.image_text_hidden_size
        )
        proprio_dim = int(
            self.model.proprio_dim
        )

        gate = DynamicChannelGate(
            feature_dim=feature_dim,
            proprio_dim=proprio_dim,
            num_groups=self.dynamic_gate_num_groups,
            hidden_dim=self.dynamic_gate_hidden_dim,
            target_keep_ratio=(
                self.dynamic_gate_target_keep_ratio
            ),
            temperature=self.dynamic_gate_temperature,
            rescale=self.dynamic_gate_rescale,
        )

        # Keep the small gate in float32. The gate implementation converts
        # bfloat16 PiZero inputs into its own compute dtype.
        gate.to(
            device=self.device,
            dtype=torch.float32,
        )

        return gate

    def _load_dynamic_gate_checkpoint_if_needed(
        self,
    ) -> None:
        if self.dynamic_gate_checkpoint is None:
            return

        if not osp.isfile(self.dynamic_gate_checkpoint):
            raise FileNotFoundError(
                "Dynamic-gate checkpoint does not exist: "
                f"{self.dynamic_gate_checkpoint}"
            )

        checkpoint = torch.load(
            self.dynamic_gate_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        self._validate_checkpoint_metadata(checkpoint)

        state_dict = _extract_gate_state_dict(
            checkpoint
        )

        incompatible = self.visual_gate.load_state_dict(
            state_dict,
            strict=self.dynamic_gate_checkpoint_strict,
        )

        if (
            not self.dynamic_gate_checkpoint_strict
            and self.dynamic_gate_verbose
        ):
            print(
                "[PiZeroDynamicGate] checkpoint missing keys:",
                list(incompatible.missing_keys),
            )
            print(
                "[PiZeroDynamicGate] checkpoint unexpected keys:",
                list(incompatible.unexpected_keys),
            )

    def _validate_checkpoint_metadata(
        self,
        checkpoint: Any,
    ) -> None:
        if not isinstance(checkpoint, Mapping):
            return

        expected_values = {
            "feature_dim": int(
                self.model.image_text_hidden_size
            ),
            "proprio_dim": int(
                self.model.proprio_dim
            ),
            "num_groups": self.dynamic_gate_num_groups,
            "hidden_dim": self.dynamic_gate_hidden_dim,
        }

        for key, expected in expected_values.items():
            if key not in checkpoint:
                continue

            actual = int(checkpoint[key])

            if actual != expected:
                raise ValueError(
                    f"Checkpoint metadata mismatch for {key}: "
                    f"checkpoint={actual}, configured={expected}."
                )

    def _print_initialization_summary(self) -> None:
        num_gate_parameters = sum(
            parameter.numel()
            for parameter in self.visual_gate.parameters()
        )

        print()
        print(
            "[PiZeroDynamicGate] Initialized "
            "PiZeroDynamicGateInference"
        )
        print(
            "[PiZeroDynamicGate] gate_mode:",
            self.dynamic_gate_mode,
        )
        print(
            "[PiZeroDynamicGate] checkpoint:",
            self.dynamic_gate_checkpoint,
        )
        print(
            "[PiZeroDynamicGate] requested_torch_compile:",
            self._requested_torch_compile,
        )
        print(
            "[PiZeroDynamicGate] actual_torch_compile: False"
        )
        print(
            "[PiZeroDynamicGate] num_groups:",
            self.dynamic_gate_num_groups,
        )
        print(
            "[PiZeroDynamicGate] target_keep_ratio:",
            self.dynamic_gate_target_keep_ratio,
        )
        print(
            "[PiZeroDynamicGate] rescale:",
            self.dynamic_gate_rescale,
        )
        print(
            "[PiZeroDynamicGate] gate_parameters:",
            num_gate_parameters,
        )
        print(
            "[PiZeroDynamicGate] PiZero trainable parameters:",
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        )

    @staticmethod
    def _tensor_statistics(
        tensor: Optional[torch.Tensor],
        prefix: str,
    ) -> Dict[str, float]:
        if tensor is None:
            return {}

        values = tensor.detach().float()

        return {
            f"{prefix}_mean": float(
                values.mean().cpu()
            ),
            f"{prefix}_min": float(
                values.min().cpu()
            ),
            f"{prefix}_max": float(
                values.max().cpu()
            ),
            f"{prefix}_std": float(
                values.std(unbiased=False).cpu()
            ),
        }

    def _build_gate_diagnostics(
        self,
        auxiliary_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "query_index": int(self._query_index),
            "gate_mode": self.dynamic_gate_mode,
            "checkpoint_loaded": (
                self.dynamic_gate_checkpoint is not None
            ),
            "num_groups": int(
                self.dynamic_gate_num_groups
            ),
        }

        if not isinstance(auxiliary_output, Mapping):
            return diagnostics

        gate_info = auxiliary_output.get(
            "visual_gate"
        )

        if not isinstance(gate_info, Mapping):
            return diagnostics

        group_mask = gate_info.get("group_mask")
        channel_mask = gate_info.get("channel_mask")
        applied_channel_mask = gate_info.get(
            "applied_channel_mask"
        )

        diagnostics.update(
            self._tensor_statistics(
                group_mask,
                "group_mask",
            )
        )
        diagnostics.update(
            self._tensor_statistics(
                channel_mask,
                "channel_mask",
            )
        )
        diagnostics.update(
            self._tensor_statistics(
                applied_channel_mask,
                "applied_channel_mask",
            )
        )

        if torch.is_tensor(group_mask):
            diagnostics["group_mask_shape"] = tuple(
                group_mask.shape
            )

        if torch.is_tensor(channel_mask):
            diagnostics["channel_mask_shape"] = tuple(
                channel_mask.shape
            )

        return diagnostics

    def _print_query_diagnostics(
        self,
        diagnostics: Dict[str, Any],
    ) -> None:
        if (
            self._query_index
            % self.dynamic_gate_log_every
            != 0
        ):
            return

        message = [
            "[PiZeroDynamicGate]",
            f"query={diagnostics.get('query_index')}",
            f"mode={diagnostics.get('gate_mode')}",
        ]

        if "group_mask_mean" in diagnostics:
            message.append(
                "group_mean="
                f"{diagnostics['group_mask_mean']:.6f}"
            )
            message.append(
                "group_min="
                f"{diagnostics['group_mask_min']:.6f}"
            )
            message.append(
                "group_max="
                f"{diagnostics['group_mask_max']:.6f}"
            )

        if "channel_mask_shape" in diagnostics:
            message.append(
                "channel_shape="
                f"{diagnostics['channel_mask_shape']}"
            )

        print(" ".join(message))

    def reset(
        self,
        instruction,
        seed=None,
    ) -> None:
        super().reset(
            instruction,
            seed=seed,
        )

        self._query_index = 0
        self.last_gate_diagnostics = None

    # wx:Dynamic gate v3
    # @torch.inference_mode()
    # def step(
    #     self,
    #     image,
    #     instruction,
    #     proprio,
    #     *args,
    #     **kwargs,
    # ):
    #     """
    #     Execute one PiZero policy query with one visual-gate forward.

    #     The external gate is passed through the stage-0 explicit interface.
    #     No hook, optimizer, backward pass, or repeated action inference is
    #     used.
    #     """
    #     inputs = self.preprocess_inputs(
    #         image,
    #         instruction,
    #         proprio,
    #     )

    #     raw_actions, auxiliary_output = (
    #         super().forward_actions(
    #             inputs,
    #             visual_gate=self.visual_gate,
    #             return_aux=True,
    #         )
    #     )

    #     diagnostics = self._build_gate_diagnostics(
    #         auxiliary_output
    #     )

    #     self.last_gate_diagnostics = diagnostics

    #     if self.dynamic_gate_verbose:
    #         self._print_query_diagnostics(
    #             diagnostics
    #         )

    #     actions = self.env_adapter.postprocess(
    #         raw_actions[0].float().cpu().numpy()
    #     )

    #     self._query_index += 1

    #     return raw_actions, actions
    # wx:Dynamic gate v3
    @torch.inference_mode()
    def step(
        self,
        image,
        instruction,
        proprio,
        *args,
        **kwargs,
    ):
        """
        Execute one PiZero query with exactly one gated policy forward.
        """
        episode_id = kwargs.pop(
            "episode_id",
            self._current_episode_id,
        )
        query_index = kwargs.pop(
            "query_index",
            self._fallback_query_index,
        )
        task = kwargs.pop(
            "task",
            self.evaluation_task_name,
        )

        inputs = self.preprocess_inputs(
            image,
            instruction,
            proprio,
        )

        initial_action_noise = (
            self.build_deterministic_action_noise(
                task=task,
                episode_id=episode_id,
                query_index=query_index,
            )
        )

        raw_actions, auxiliary_output = (
            super().forward_actions(
                inputs,
                visual_gate=self.visual_gate,
                initial_action_noise=initial_action_noise,
                return_aux=True,
            )
        )

        diagnostics = self._build_gate_diagnostics(
            auxiliary_output
        )
        diagnostics["episode_id"] = (
            None if episode_id is None else int(episode_id)
        )
        diagnostics["action_noise_deterministic"] = bool(
            self.deterministic_action_noise
        )

        self.last_gate_diagnostics = diagnostics

        if self.dynamic_gate_verbose:
            self._print_query_diagnostics(
                diagnostics
            )

        actions = self.env_adapter.postprocess(
            raw_actions[0].float().cpu().numpy()
        )

        self._query_index += 1
        self._fallback_query_index += 1

        return raw_actions, actions
    # wx:Dynamic gate v3