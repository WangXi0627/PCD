# wx:Dynamic gate v2

"""Checkpoint helpers for DynamicChannelGate training and inference."""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .dynamic_channel_gate import DynamicChannelGate


CHECKPOINT_VERSION = "dynamic_gate_v1"


def _gate_config_from_module(gate: DynamicChannelGate) -> Dict[str, Any]:
    return {
        "feature_dim": int(gate.feature_dim),
        "proprio_dim": int(gate.proprio_dim),
        "num_groups": int(gate.num_groups),
        "hidden_dim": int(gate.hidden_dim),
        "target_keep_ratio": float(gate.target_keep_ratio),
        "temperature": float(gate.temperature),
        "rescale": bool(gate.rescale),
    }


def save_dynamic_gate_checkpoint(
    path: str,
    gate: DynamicChannelGate,
    *,
    stage: str,
    policy_config: Mapping[str, Any],
    global_step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Save a checkpoint that is both resumable and directly readable by the
    stage-1 PiZeroDynamicGateInference loader.
    """
    gate_config = _gate_config_from_module(gate)

    checkpoint: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "stage": str(stage),
        "gate_state_dict": gate.state_dict(),

        # Top-level compatibility fields used by the stage-1 wrapper.
        "feature_dim": gate_config["feature_dim"],
        "proprio_dim": gate_config["proprio_dim"],
        "num_groups": gate_config["num_groups"],
        "hidden_dim": gate_config["hidden_dim"],
        "target_keep_ratio": gate_config["target_keep_ratio"],
        "temperature": gate_config["temperature"],
        "rescale": gate_config["rescale"],

        "gate_config": gate_config,
        "policy_config": dict(policy_config),
        "training_state": {
            "global_step": int(global_step),
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
        },
    }

    if extra is not None:
        checkpoint["extra"] = dict(extra)

    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    return path


def load_checkpoint_payload(path: str) -> Dict[str, Any]:
    path = osp.abspath(path)
    if not osp.isfile(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(payload)}.")

    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_version={version!r}; expected {CHECKPOINT_VERSION!r}."
        )

    if "gate_state_dict" not in payload:
        raise KeyError("Checkpoint is missing gate_state_dict.")
    if "gate_config" not in payload:
        raise KeyError("Checkpoint is missing gate_config.")
    if "policy_config" not in payload:
        raise KeyError("Checkpoint is missing policy_config.")

    return payload


def validate_checkpoint_compatibility(
    payload: Mapping[str, Any],
    *,
    expected_feature_dim: Optional[int] = None,
    expected_proprio_dim: Optional[int] = None,
    expected_num_groups: Optional[int] = None,
    expected_hidden_dim: Optional[int] = None,
    expected_base_checkpoint: Optional[str] = None,
) -> None:
    gate_config = payload["gate_config"]
    policy_config = payload["policy_config"]

    expected_values = {
        "feature_dim": expected_feature_dim,
        "proprio_dim": expected_proprio_dim,
        "num_groups": expected_num_groups,
        "hidden_dim": expected_hidden_dim,
    }

    for key, expected in expected_values.items():
        if expected is None:
            continue
        actual = int(gate_config[key])
        if actual != int(expected):
            raise ValueError(
                f"Checkpoint {key} mismatch: checkpoint={actual}, expected={expected}."
            )

    if expected_base_checkpoint is not None:
        actual_checkpoint = str(policy_config.get("base_checkpoint", ""))
        if osp.abspath(actual_checkpoint) != osp.abspath(expected_base_checkpoint):
            raise ValueError(
                "Base checkpoint mismatch: "
                f"checkpoint={actual_checkpoint!r}, expected={expected_base_checkpoint!r}."
            )


def build_gate_from_checkpoint(
    path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    strict: bool = True,
) -> Tuple[DynamicChannelGate, Dict[str, Any]]:
    payload = load_checkpoint_payload(path)
    config = payload["gate_config"]

    gate = DynamicChannelGate(
        feature_dim=int(config["feature_dim"]),
        proprio_dim=int(config["proprio_dim"]),
        num_groups=int(config["num_groups"]),
        hidden_dim=int(config["hidden_dim"]),
        target_keep_ratio=float(config["target_keep_ratio"]),
        temperature=float(config["temperature"]),
        rescale=bool(config["rescale"]),
    )
    gate.load_state_dict(payload["gate_state_dict"], strict=strict)

    if device is not None:
        gate.to(device=device, dtype=dtype)

    return gate, payload


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
) -> int:
    training_state = payload.get("training_state", {})
    optimizer_state = training_state.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return int(training_state.get("global_step", 0))
