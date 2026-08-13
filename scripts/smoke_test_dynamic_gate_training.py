# wx:Dynamic gate v2

#!/usr/bin/env python3
"""Stage-2 end-to-end smoke test for dynamic-gate training infrastructure."""

from __future__ import annotations

import argparse
import gc
import os.path as osp
import sys
from typing import Dict, List, Mapping

import numpy as np
import torch


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating import DynamicChannelGate, IdentityVisualGate  # noqa: E402
from feature_gating.checkpoint import (  # noqa: E402
    build_gate_from_checkpoint,
    save_dynamic_gate_checkpoint,
    validate_checkpoint_compatibility,
)
from feature_gating.fixed_noise_teacher import (  # noqa: E402
    build_initial_action_noise,
    compute_gated_action,
    compute_teacher_action,
    unwrap_pizero_model,
)
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
    load_dynamic_gate_manifest,
)
from feature_gating.training_step import (  # noqa: E402
    action_matching_loss,
    prepare_pizero_batch_inputs,
    run_training_step,
)
from simpler_env.policies.pizero.pizero_model import PiZeroInference  # noqa: E402


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
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, required=True)
    parser.add_argument("--cfg-dir", type=str, default=DEFAULT_CFG_DIR)
    parser.add_argument(
        "--policy-setup",
        choices=["auto", "google_robot", "widowx_bridge"],
        default="auto",
    )
    parser.add_argument("--flow-sampling", choices=["beta", "uniform"], default="beta")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--target-keep-ratio", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--allow-failure-samples", action="store_true")
    parser.add_argument("--skip-wrapper-compat-check", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    return parser.parse_args()


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().detach().cpu())


def snapshot_gate(gate: DynamicChannelGate) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in gate.named_parameters()
    }


def gate_changed(before: Mapping[str, torch.Tensor], gate: DynamicChannelGate) -> bool:
    for name, parameter in gate.named_parameters():
        if not torch.equal(before[name], parameter.detach().cpu()):
            return True
    return False


def snapshot_model_slices(model, max_tensors: int = 3, max_values: int = 1024):
    snapshots = {}
    for name, parameter in model.named_parameters():
        flat = parameter.detach().view(-1)
        snapshots[name] = flat[: min(max_values, flat.numel())].cpu().clone()
        if len(snapshots) >= max_tensors:
            break
    return snapshots


def assert_model_slices_unchanged(model, snapshots) -> None:
    named_parameters = dict(model.named_parameters())
    for name, expected in snapshots.items():
        actual = named_parameters[name].detach().view(-1)[: expected.numel()].cpu()
        if not torch.equal(expected, actual):
            raise AssertionError(f"Frozen PiZero parameter changed: {name}")


def get_samples(args: argparse.Namespace):
    require_success = None if args.allow_failure_samples else True
    try:
        dataset = RolloutQueryDataset(
            args.manifest,
            split=args.split,
            require_success=require_success,
            filter_instruction_changed=False,
        )
    except RuntimeError:
        if args.allow_failure_samples:
            raise
        print("[WARN] No successful samples matched; falling back to all samples.")
        dataset = RolloutQueryDataset(
            args.manifest,
            split=args.split,
            require_success=None,
            filter_instruction_changed=False,
        )

    count = min(max(1, int(args.num_samples)), len(dataset))
    samples = [dataset[index] for index in range(count)]
    return dataset, samples


def run_wrapper_compatibility_check(
    args: argparse.Namespace,
    sample: Mapping,
    reference_action: torch.Tensor,
    reference_noise: torch.Tensor,
) -> None:
    from contrast_policies.pizero_dynamic_gate import PiZeroDynamicGateInference

    wrapper = PiZeroDynamicGateInference(
        dynamic_feature_gate=True,
        dynamic_gate_mode="dynamic",
        dynamic_gate_checkpoint=osp.abspath(args.output_checkpoint),
        dynamic_gate_num_groups=args.num_groups,
        dynamic_gate_hidden_dim=args.hidden_dim,
        dynamic_gate_target_keep_ratio=args.target_keep_ratio,
        dynamic_gate_temperature=args.temperature,
        dynamic_gate_rescale=args.rescale,
        dynamic_gate_checkpoint_strict=True,
        dynamic_gate_verbose=False,
        cfg_dir=args.cfg_dir,
        checkpoint_path=args.checkpoint_path,
        policy_setup=args.policy_setup,
        flow_sampling=args.flow_sampling,
        use_ddp=False,
        use_naive=False,
        use_torch_compile=False,
        seed=args.seed,
    )

    inputs = wrapper.preprocess_inputs(
        sample["image"],
        sample["instruction"],
        sample["proprio"],
    )
    action = wrapper.forward_actions(
        inputs,
        visual_gate=wrapper.visual_gate,
        initial_action_noise=reference_noise.to(
            device=wrapper.device,
            dtype=wrapper.dtype,
        ),
        return_aux=False,
    )

    difference = max_abs_diff(reference_action, action.detach().cpu())
    if not torch.allclose(
        reference_action,
        action.detach().cpu(),
        atol=args.atol,
        rtol=args.rtol,
    ):
        raise AssertionError(
            "Stage-1 wrapper checkpoint compatibility failed: "
            f"max_abs_diff={difference}."
        )

    print(
        "[PASS] Stage-1 PiZeroDynamicGateInference loaded the training "
        f"checkpoint; max_abs_diff={difference:.10f}."
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = load_dynamic_gate_manifest(args.manifest)
    if args.policy_setup == "auto":
        args.policy_setup = str(manifest["policy_setup"])
    elif args.policy_setup != str(manifest["policy_setup"]):
        raise ValueError(
            f"policy_setup mismatch: CLI={args.policy_setup}, "
            f"manifest={manifest['policy_setup']}."
        )

    dataset, samples = get_samples(args)
    batch = collate_rollout_queries(samples)

    print("========== Stage-2 Dynamic Gate Training Smoke Test ==========")
    print(f"manifest: {osp.abspath(args.manifest)}")
    print(f"policy_setup: {args.policy_setup}")
    print(f"dataset queries: {len(dataset)}")
    print(f"smoke samples: {batch['sample_id']}")

    policy = PiZeroInference(
        cfg_dir=args.cfg_dir,
        checkpoint_path=args.checkpoint_path,
        policy_setup=args.policy_setup,
        flow_sampling=args.flow_sampling,
        use_ddp=False,
        use_naive=False,
        use_torch_compile=False,
        seed=args.seed,
    )
    model = unwrap_pizero_model(policy)

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("PiZero contains trainable parameters in stage 2.")

    gate = DynamicChannelGate(
        feature_dim=int(model.image_text_hidden_size),
        proprio_dim=int(model.proprio_dim),
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        target_keep_ratio=args.target_keep_ratio,
        temperature=args.temperature,
        rescale=args.rescale,
    ).to(device=policy.device, dtype=torch.float32)
    gate.train()

    optimizer = torch.optim.Adam(gate.parameters(), lr=args.learning_rate)

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if optimizer_parameter_ids & model_parameter_ids:
        raise AssertionError("PiZero parameters unexpectedly entered the optimizer.")
    print("[PASS] Optimizer contains gate parameters only.")

    model_inputs = prepare_pizero_batch_inputs(policy, batch)

    cpu_rng_before = torch.random.get_rng_state().clone()
    noise_1 = build_initial_action_noise(
        batch["sample_id"],
        model.horizon_steps,
        model.action_dim,
        policy.device,
        policy.dtype,
        base_seed=args.base_noise_seed,
    )
    cpu_rng_after = torch.random.get_rng_state().clone()
    noise_2 = build_initial_action_noise(
        batch["sample_id"],
        model.horizon_steps,
        model.action_dim,
        policy.device,
        policy.dtype,
        base_seed=args.base_noise_seed,
    )

    if not torch.equal(cpu_rng_before, cpu_rng_after):
        raise AssertionError("Fixed-noise generation modified the global CPU RNG state.")
    if not torch.equal(noise_1, noise_2):
        raise AssertionError("Per-sample fixed noise is not deterministic.")
    print("[PASS] Per-sample fixed noise is deterministic and RNG-independent.")

    noise_before_forward = noise_1.detach().clone()
    teacher_1 = compute_teacher_action(policy, model_inputs, noise_1)
    teacher_2 = compute_teacher_action(policy, model_inputs, noise_1)
    if not torch.equal(noise_before_forward, noise_1):
        raise AssertionError("Teacher forward modified initial_action_noise.")
    if not torch.allclose(teacher_1, teacher_2, atol=0.0, rtol=0.0):
        raise AssertionError("Teacher action is not deterministic under fixed noise.")
    if teacher_1.requires_grad:
        raise AssertionError("Teacher action unexpectedly requires gradients.")
    print("[PASS] Frozen teacher is deterministic and detached.")

    identity_gate = IdentityVisualGate().to(policy.device)
    identity_action, _ = compute_gated_action(
        policy,
        model_inputs,
        identity_gate,
        noise_1,
    )
    identity_difference = max_abs_diff(teacher_1, identity_action)
    if not torch.allclose(
        teacher_1,
        identity_action,
        atol=args.atol,
        rtol=args.rtol,
    ):
        raise AssertionError(
            f"Identity gate differs from teacher; max_abs_diff={identity_difference}."
        )
    print(
        "[PASS] Identity gate reproduces teacher action; "
        f"max_abs_diff={identity_difference:.10f}."
    )

    dynamic_action_1, dynamic_aux_1 = compute_gated_action(
        policy,
        model_inputs,
        gate,
        noise_1,
    )
    dynamic_action_2, _ = compute_gated_action(
        policy,
        model_inputs,
        gate,
        noise_1,
    )
    if not torch.allclose(dynamic_action_1, dynamic_action_2, atol=0.0, rtol=0.0):
        raise AssertionError("Gated action is not deterministic under fixed noise.")
    if not dynamic_action_1.requires_grad:
        raise AssertionError("Gated action does not retain a gradient path to the gate.")

    gate_info = dynamic_aux_1.get("visual_gate")
    if not isinstance(gate_info, Mapping):
        raise AssertionError("Missing visual_gate auxiliary output.")
    if tuple(gate_info["group_mask"].shape) != (len(samples), args.num_groups):
        raise AssertionError(
            f"Unexpected group mask shape: {tuple(gate_info['group_mask'].shape)}"
        )
    if tuple(gate_info["channel_mask"].shape) != (
        len(samples),
        int(model.image_text_hidden_size),
    ):
        raise AssertionError(
            f"Unexpected channel mask shape: {tuple(gate_info['channel_mask'].shape)}"
        )
    print("[PASS] Dynamic gate produces deterministic group/channel masks.")

    model_snapshots = snapshot_model_slices(model)
    gate_before_step = snapshot_gate(gate)
    first_result = run_training_step(
        policy=policy,
        visual_gate=gate,
        optimizer=optimizer,
        batch=batch,
        base_noise_seed=args.base_noise_seed,
        max_grad_norm=args.max_grad_norm,
    )

    if first_result.teacher_requires_grad:
        raise AssertionError("Teacher action requires gradients during training step.")
    if not first_result.gated_requires_grad:
        raise AssertionError("Gated action lost its gradient path.")
    if first_result.model_parameters_with_grad != 0:
        raise AssertionError(
            f"PiZero received gradients: {first_result.model_parameters_with_grad} parameters."
        )
    if first_result.gate_parameters_with_grad <= 0:
        raise AssertionError("No gate parameters received gradients.")
    if first_result.gate_grad_norm <= 0.0:
        raise AssertionError("Gate gradient norm is zero.")
    if not gate_changed(gate_before_step, gate):
        raise AssertionError("Gate parameters did not change after optimizer.step().")
    assert_model_slices_unchanged(model, model_snapshots)
    print(
        "[PASS] One-step optimization updated gate only; "
        f"loss={first_result.loss:.8f}, grad_norm={first_result.gate_grad_norm:.8f}."
    )

    losses: List[float] = [first_result.loss]
    for step in range(1, max(1, int(args.steps))):
        result = run_training_step(
            policy=policy,
            visual_gate=gate,
            optimizer=optimizer,
            batch=batch,
            base_noise_seed=args.base_noise_seed,
            max_grad_norm=args.max_grad_norm,
        )
        losses.append(result.loss)

        if not np.isfinite(result.loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {result.loss}")
        if step == 1 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(
                f"[TRAIN] step={step + 1:04d} loss={result.loss:.8f} "
                f"grad_norm={result.gate_grad_norm:.8f} "
                f"mask_mean={result.group_mask_mean}"
            )

    tail_best = min(losses[-min(5, len(losses)):])
    if len(losses) >= 2 and tail_best >= losses[0]:
        raise AssertionError(
            "Small-sample overfit loss did not decrease: "
            f"initial={losses[0]:.8f}, tail_best={tail_best:.8f}, "
            f"final={losses[-1]:.8f}."
        )
    print(
        "[PASS] Small-sample overfit loss decreased: "
        f"initial={losses[0]:.8f}, tail_best={tail_best:.8f}, "
        f"final={losses[-1]:.8f}."
    )

    policy_config = {
        "policy_setup": args.policy_setup,
        "base_checkpoint": osp.abspath(args.checkpoint_path),
        "flow_sampling": args.flow_sampling,
        "horizon_steps": int(model.horizon_steps),
        "action_dim": int(model.action_dim),
        "cfg_dir": osp.abspath(args.cfg_dir),
    }
    save_dynamic_gate_checkpoint(
        args.output_checkpoint,
        gate,
        stage="training_smoke",
        policy_config=policy_config,
        global_step=len(losses),
        optimizer=optimizer,
        extra={
            "manifest": osp.abspath(args.manifest),
            "sample_ids": list(batch["sample_id"]),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "base_noise_seed": int(args.base_noise_seed),
        },
    )
    print(f"[PASS] Saved checkpoint: {osp.abspath(args.output_checkpoint)}")

    loaded_gate, payload = build_gate_from_checkpoint(
        args.output_checkpoint,
        device=policy.device,
        dtype=torch.float32,
        strict=True,
    )
    validate_checkpoint_compatibility(
        payload,
        expected_feature_dim=int(model.image_text_hidden_size),
        expected_proprio_dim=int(model.proprio_dim),
        expected_num_groups=args.num_groups,
        expected_hidden_dim=args.hidden_dim,
        expected_base_checkpoint=osp.abspath(args.checkpoint_path),
    )
    loaded_gate.eval()
    gate.eval()

    reference_action, _ = compute_gated_action(
        policy,
        model_inputs,
        gate,
        noise_1,
    )
    loaded_action, _ = compute_gated_action(
        policy,
        model_inputs,
        loaded_gate,
        noise_1,
    )
    roundtrip_difference = max_abs_diff(reference_action, loaded_action)
    if not torch.allclose(reference_action, loaded_action, atol=0.0, rtol=0.0):
        raise AssertionError(
            f"Checkpoint roundtrip changed action; max_abs_diff={roundtrip_difference}."
        )
    print("[PASS] Checkpoint roundtrip reproduces gate action exactly.")

    first_batch = collate_rollout_queries([samples[0]])
    first_inputs = prepare_pizero_batch_inputs(policy, first_batch)
    first_noise = build_initial_action_noise(
        first_batch["sample_id"],
        model.horizon_steps,
        model.action_dim,
        policy.device,
        policy.dtype,
        base_seed=args.base_noise_seed,
    )
    first_reference_action, _ = compute_gated_action(
        policy,
        first_inputs,
        loaded_gate,
        first_noise,
    )
    first_reference_action_cpu = first_reference_action.detach().cpu()
    first_noise_cpu = first_noise.detach().cpu()

    if not args.skip_wrapper_compat_check:
        del optimizer
        del loaded_gate
        del gate
        del policy
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_wrapper_compatibility_check(
            args,
            sample=samples[0],
            reference_action=first_reference_action_cpu,
            reference_noise=first_noise_cpu,
        )
    else:
        print("[SKIP] Stage-1 wrapper compatibility check skipped.")

    print("========== Stage-2 smoke test passed ==========")


if __name__ == "__main__":
    main()
