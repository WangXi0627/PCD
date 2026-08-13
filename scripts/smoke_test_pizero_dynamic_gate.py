# wx:Dynamic gate v0

#!/usr/bin/env python3
"""
Stage-0 smoke test for the PiZero dynamic visual-gate interface.

Example
-------
python scripts/smoke_test_pizero_dynamic_gate.py \
    --checkpoint-path /path/to/open_pi_zero \
    --episode-dir rollouts/pizero/google_robot_pick_coke_can/episode_000000
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import tempfile
from typing import Dict, Tuple

import numpy as np
import torch


REPO_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from feature_gating import (  # noqa: E402
    DynamicChannelGate,
    IdentityVisualGate,
)
from simpler_env.policies.pizero.pizero_model import (  # noqa: E402
    PiZeroInference,
)


DEFAULT_CFG_DIR = osp.join(
    REPO_ROOT,
    "simpler_env",
    "policies",
    "pizero",
    "open_pi_zero",
    "config",
    "eval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help=(
            "Directory containing the PiZero bridge/fractal checkpoint "
            "files, matching PiZeroInference.checkpoint_path."
        ),
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        required=True,
        help=(
            "Episode directory containing trajectory.npz and metadata.json."
        ),
    )
    parser.add_argument(
        "--cfg-dir",
        type=str,
        default=DEFAULT_CFG_DIR,
    )
    parser.add_argument(
        "--policy-setup",
        choices=[
            "auto",
            "google_robot",
            "widowx_bridge",
        ],
        default="auto",
    )
    parser.add_argument(
        "--flow-sampling",
        choices=["beta", "uniform"],
        default="beta",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--target-keep-ratio",
        type=float,
        default=0.75,
        help=(
            "Use a visible non-identity ratio for the smoke test. "
            "The eventual training initialization can use 0.95."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--skip-grad-check",
        action="store_true",
        help=(
            "Skip the full backward pass through frozen PiZero. "
            "Useful for the first low-memory interface test."
        ),
    )

    return parser.parse_args()


def load_episode_query(
    episode_dir: str,
    query_index: int,
) -> Tuple[np.ndarray, str, np.ndarray, Dict]:
    trajectory_path = osp.join(
        episode_dir,
        "trajectory.npz",
    )
    metadata_path = osp.join(
        episode_dir,
        "metadata.json",
    )

    if not osp.isfile(trajectory_path):
        raise FileNotFoundError(trajectory_path)
    if not osp.isfile(metadata_path):
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with np.load(
        trajectory_path,
        allow_pickle=False,
    ) as trajectory:
        images = trajectory["images"]
        proprios = trajectory["policy_proprios"]

        num_queries = images.shape[0]
        if not (0 <= query_index < num_queries):
            raise IndexError(
                f"query_index={query_index} is outside "
                f"[0, {num_queries})."
            )

        image = np.asarray(
            images[query_index]
        ).copy()
        proprio = np.asarray(
            proprios[query_index]
        ).copy()

    instructions = metadata.get("instructions", [])
    if len(instructions) <= query_index:
        raise ValueError(
            "metadata.instructions does not contain "
            f"query_index={query_index}."
        )

    instruction = str(instructions[query_index])

    return image, instruction, proprio, metadata


def infer_policy_setup(
    requested_setup: str,
    metadata: Dict,
) -> str:
    if requested_setup != "auto":
        return requested_setup

    task = str(metadata.get("task", ""))

    if task.startswith("google_robot"):
        return "google_robot"
    if task.startswith("widowx"):
        return "widowx_bridge"

    raise ValueError(
        "Could not infer policy setup from metadata task: "
        f"{task!r}. Pass --policy-setup explicitly."
    )


def make_initial_noise(
    policy: PiZeroInference,
    inputs: Dict[str, torch.Tensor],
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = int(inputs["pixel_values"].shape[0])

    return torch.randn(
        (
            batch_size,
            policy.model.horizon_steps,
            policy.model.action_dim,
        ),
        device=policy.device,
        dtype=policy.dtype,
    )


def tensor_max_abs_difference(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    return float(
        (first.float() - second.float())
        .abs()
        .max()
        .detach()
        .cpu()
    )


def test_checkpoint_roundtrip(
    *,
    policy: PiZeroInference,
    inputs: Dict[str, torch.Tensor],
    initial_noise: torch.Tensor,
    original_gate: DynamicChannelGate,
    original_actions: torch.Tensor,
) -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        checkpoint_path = osp.join(
            temporary_dir,
            "dynamic_gate_stage0.pt",
        )

        torch.save(
            {
                "gate_state_dict": original_gate.state_dict(),
                "feature_dim": original_gate.feature_dim,
                "proprio_dim": original_gate.proprio_dim,
                "num_groups": original_gate.num_groups,
                "hidden_dim": original_gate.hidden_dim,
                "target_keep_ratio": (
                    original_gate.target_keep_ratio
                ),
                "temperature": original_gate.temperature,
                "rescale": original_gate.rescale,
            },
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        loaded_gate = DynamicChannelGate(
            feature_dim=int(checkpoint["feature_dim"]),
            proprio_dim=int(checkpoint["proprio_dim"]),
            num_groups=int(checkpoint["num_groups"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            target_keep_ratio=float(
                checkpoint["target_keep_ratio"]
            ),
            temperature=float(checkpoint["temperature"]),
            rescale=bool(checkpoint["rescale"]),
        )
        loaded_gate.load_state_dict(
            checkpoint["gate_state_dict"],
            strict=True,
        )
        loaded_gate.to(
            device=policy.device,
            dtype=torch.float32,
        )
        loaded_gate.eval()

        loaded_actions = policy.forward_actions(
            inputs,
            visual_gate=loaded_gate,
            initial_action_noise=initial_noise,
        )

        if not torch.allclose(
            original_actions,
            loaded_actions,
            atol=0.0,
            rtol=0.0,
        ):
            difference = tensor_max_abs_difference(
                original_actions,
                loaded_actions,
            )
            raise AssertionError(
                "Gate checkpoint roundtrip changed the output. "
                f"max_abs_diff={difference}"
            )


def run_gradient_check(
    *,
    policy: PiZeroInference,
    inputs: Dict[str, torch.Tensor],
    initial_noise: torch.Tensor,
    gate: DynamicChannelGate,
) -> None:
    gate.train()
    gate.zero_grad(set_to_none=True)

    for parameter in policy.model.parameters():
        parameter.grad = None

    gated_actions, _ = (
        policy.forward_actions_with_gate_grad(
            inputs,
            visual_gate=gate,
            initial_action_noise=initial_noise,
            return_aux=True,
        )
    )

    loss = gated_actions.float().square().mean()
    loss.backward()

    gate_gradients = [
        parameter.grad
        for parameter in gate.parameters()
        if parameter.grad is not None
    ]

    if len(gate_gradients) == 0:
        raise AssertionError(
            "No gradients reached DynamicChannelGate."
        )

    finite_gradient_norm = 0.0
    for gradient in gate_gradients:
        if not torch.isfinite(gradient).all():
            raise AssertionError(
                "DynamicChannelGate contains a non-finite gradient."
            )
        finite_gradient_norm += float(
            gradient.float().norm().detach().cpu()
        )

    if finite_gradient_norm <= 0.0:
        raise AssertionError(
            "DynamicChannelGate gradient norm is zero."
        )

    model_parameters_with_grad = [
        name
        for name, parameter in policy.model.named_parameters()
        if parameter.grad is not None
    ]

    if model_parameters_with_grad:
        raise AssertionError(
            "Frozen PiZero parameters unexpectedly received gradients: "
            f"{model_parameters_with_grad[:10]}"
        )

    print(
        "[PASS] Gate gradient reached trainable parameters; "
        f"total_gradient_norm={finite_gradient_norm:.6f}"
    )
    print(
        "[PASS] Frozen PiZero parameters have no gradients."
    )

    gate.eval()
    gate.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()

    image, instruction, proprio, metadata = (
        load_episode_query(
            episode_dir=args.episode_dir,
            query_index=args.query_index,
        )
    )

    policy_setup = infer_policy_setup(
        requested_setup=args.policy_setup,
        metadata=metadata,
    )

    print("========== Stage-0 Dynamic Gate Smoke Test ==========")
    print(f"task: {metadata.get('task')}")
    print(f"episode_id: {metadata.get('episode_id')}")
    print(f"query_index: {args.query_index}")
    print(f"final_success: {metadata.get('final_success')}")
    print(f"policy_setup: {policy_setup}")
    print(f"instruction: {instruction}")

    policy = PiZeroInference(
        cfg_dir=args.cfg_dir,
        checkpoint_path=args.checkpoint_path,
        policy_setup=policy_setup,
        flow_sampling=args.flow_sampling,
        use_ddp=False,
        use_naive=False,
        use_torch_compile=False,
        seed=args.seed,
    )

    inputs = policy.preprocess_inputs(
        image=image,
        instruction=instruction,
        proprio=proprio,
    )

    initial_noise = make_initial_noise(
        policy=policy,
        inputs=inputs,
        seed=args.seed,
    )

    baseline_actions = policy.forward_actions(
        inputs,
        initial_action_noise=initial_noise,
    )

    identity_gate = IdentityVisualGate().to(
        device=policy.device
    )
    identity_gate.eval()

    identity_actions, identity_aux = (
        policy.forward_actions(
            inputs,
            visual_gate=identity_gate,
            initial_action_noise=initial_noise,
            return_aux=True,
        )
    )

    identity_difference = tensor_max_abs_difference(
        baseline_actions,
        identity_actions,
    )

    print(
        "[INFO] baseline vs identity max_abs_diff:",
        identity_difference,
    )

    if not torch.allclose(
        baseline_actions,
        identity_actions,
        atol=args.atol,
        rtol=args.rtol,
    ):
        raise AssertionError(
            "Identity gate changed the PiZero action. "
            f"max_abs_diff={identity_difference}, "
            f"atol={args.atol}, rtol={args.rtol}"
        )

    identity_mask = identity_aux[
        "visual_gate"
    ]["channel_mask"]

    if not torch.all(identity_mask == 1):
        raise AssertionError(
            "IdentityVisualGate did not return an all-one mask."
        )

    print(
        "[PASS] Identity gate reproduces baseline action."
    )

    feature_dim = int(policy.model.image_text_hidden_size)
    proprio_dim = int(policy.model.proprio_dim)

    dynamic_gate = DynamicChannelGate(
        feature_dim=feature_dim,
        proprio_dim=proprio_dim,
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        target_keep_ratio=args.target_keep_ratio,
        temperature=1.0,
        rescale=False,
    )
    dynamic_gate.to(
        device=policy.device,
        dtype=torch.float32,
    )
    dynamic_gate.eval()

    dynamic_actions, dynamic_aux = (
        policy.forward_actions(
            inputs,
            visual_gate=dynamic_gate,
            initial_action_noise=initial_noise,
            return_aux=True,
        )
    )

    visual_gate_aux = dynamic_aux["visual_gate"]
    group_mask = visual_gate_aux["group_mask"]
    channel_mask = visual_gate_aux["channel_mask"]

    expected_channel_shape = (
        int(inputs["pixel_values"].shape[0]),
        feature_dim,
    )
    expected_group_shape = (
        int(inputs["pixel_values"].shape[0]),
        args.num_groups,
    )

    if tuple(channel_mask.shape) != expected_channel_shape:
        raise AssertionError(
            "Unexpected channel-mask shape. "
            f"Expected {expected_channel_shape}, "
            f"got {tuple(channel_mask.shape)}."
        )

    if tuple(group_mask.shape) != expected_group_shape:
        raise AssertionError(
            "Unexpected group-mask shape. "
            f"Expected {expected_group_shape}, "
            f"got {tuple(group_mask.shape)}."
        )

    dynamic_difference = tensor_max_abs_difference(
        baseline_actions,
        dynamic_actions,
    )

    print(
        "[INFO] dynamic group mask shape:",
        tuple(group_mask.shape),
    )
    print(
        "[INFO] dynamic channel mask shape:",
        tuple(channel_mask.shape),
    )
    print(
        "[INFO] dynamic mask mean:",
        float(channel_mask.mean().detach().cpu()),
    )
    print(
        "[INFO] baseline vs dynamic max_abs_diff:",
        dynamic_difference,
    )

    if dynamic_difference <= 0.0:
        raise AssertionError(
            "Non-identity dynamic gate did not change the action."
        )

    print(
        "[PASS] Dynamic gate is applied to the action path."
    )

    test_checkpoint_roundtrip(
        policy=policy,
        inputs=inputs,
        initial_noise=initial_noise,
        original_gate=dynamic_gate,
        original_actions=dynamic_actions,
    )
    print(
        "[PASS] Dynamic gate checkpoint roundtrip is deterministic."
    )

    if not args.skip_grad_check:
        run_gradient_check(
            policy=policy,
            inputs=inputs,
            initial_noise=initial_noise,
            gate=dynamic_gate,
        )
    else:
        print(
            "[SKIP] Full gate gradient check was skipped."
        )

    print("========== All requested stage-0 checks passed ==========")


if __name__ == "__main__":
    main()