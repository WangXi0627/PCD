# wx:Dynamic gate v3

#!/usr/bin/env python3
"""Train label-free Dynamic Gate V1 with cached PiZero teacher actions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.augmentations import (  # noqa: E402
    MildAugmentationConfig,
    MildVisualAugmenter,
)
from feature_gating.checkpoint import (  # noqa: E402
    load_checkpoint_payload,
    restore_optimizer_state,
    save_dynamic_gate_checkpoint,
    validate_checkpoint_compatibility,
)
from feature_gating.dynamic_channel_gate import DynamicChannelGate  # noqa: E402
from feature_gating.fixed_noise_teacher import unwrap_pizero_model  # noqa: E402
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
)
from feature_gating.samplers import TaskEpisodeQueryBatchSampler  # noqa: E402
from feature_gating.teacher_cache import TeacherActionCache  # noqa: E402
from feature_gating.v1_losses import V1LossWeights  # noqa: E402
from feature_gating.v1_training_step import (  # noqa: E402
    forward_v1_batch,
    gate_gradient_norm,
    metrics_to_float,
    verify_frozen_policy_has_no_grad,
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

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-teacher-cache", required=True)
    parser.add_argument("--validation-teacher-cache", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-setup",
        choices=["google_robot", "widowx_bridge"],
        required=True,
    )
    parser.add_argument(
        "--flow-sampling",
        choices=["beta", "uniform"],
        default="beta",
    )
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
        help="0 means full validation.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--target-keep-ratio", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale", action="store_true")

    parser.add_argument("--lambda-suf", type=float, default=1.0)
    parser.add_argument("--lambda-action-inv", type=float, default=1.0)
    parser.add_argument("--lambda-mask-inv", type=float, default=0.1)
    parser.add_argument("--lambda-budget", type=float, default=10.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)

    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=0)

    parser.add_argument("--budget-tolerance", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_metric_dicts(
    metric_dicts: Iterable[Mapping[str, float]],
) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    count = 0

    for metrics in metric_dicts:
        count += 1
        for key, value in metrics.items():
            totals[key] += float(value)

    if count == 0:
        raise RuntimeError("No metric dictionaries were provided.")

    return {
        key: value / count
        for key, value in totals.items()
    }


def write_jsonl(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(dict(row), ensure_ascii=False)
            + "\n"
        )


def cuda_memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "peak_allocated_gib": 0.0,
            "peak_reserved_gib": 0.0,
        }

    gib = float(1024**3)
    return {
        "allocated_gib": (
            torch.cuda.memory_allocated() / gib
        ),
        "reserved_gib": (
            torch.cuda.memory_reserved() / gib
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / gib
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / gib
        ),
    }


def cuda_memory_summary() -> str:
    stats = cuda_memory_stats()
    return (
        f"allocated={stats['allocated_gib']:.2f}GiB "
        f"reserved={stats['reserved_gib']:.2f}GiB "
        f"peak_allocated={stats['peak_allocated_gib']:.2f}GiB "
        f"peak_reserved={stats['peak_reserved_gib']:.2f}GiB"
    )


def cleanup_cuda(
    *,
    empty_cache: bool = True,
    reset_peak: bool = False,
) -> None:
    """
    Release Python references and optionally return unused CUDA cache blocks
    to the allocator. This cannot free tensors that are still referenced.
    """
    gc.collect()

    if not torch.cuda.is_available():
        return

    if empty_cache:
        torch.cuda.empty_cache()

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


def is_under_budget(
    validation_metrics: Mapping[str, float],
    *,
    target_keep_ratio: float,
    tolerance: float,
) -> bool:
    return (
        abs(
            float(validation_metrics["mask_mean"])
            - float(target_keep_ratio)
        )
        <= float(tolerance)
    )


def read_best_metrics_from_history(
    history_path: str,
    *,
    target_keep_ratio: float,
    budget_tolerance: float,
) -> Tuple[float, float]:
    """
    Recover best validation metrics from an existing history.jsonl.

    This prevents --resume from resetting best_total/best_action to infinity
    and overwriting older best checkpoints with a worse resumed epoch.
    """
    best_total = float("inf")
    best_action = float("inf")

    if not osp.isfile(history_path):
        return best_total, best_action

    with open(history_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {history_path} "
                    f"at line {line_number}."
                ) from error

            validation = row.get("validation")
            if not isinstance(validation, Mapping):
                continue
            if "mask_mean" not in validation:
                continue

            if not is_under_budget(
                validation,
                target_keep_ratio=target_keep_ratio,
                tolerance=budget_tolerance,
            ):
                continue

            if "total_loss" in validation:
                best_total = min(
                    best_total,
                    float(validation["total_loss"]),
                )

            if "sufficiency_loss" in validation:
                best_action = min(
                    best_action,
                    float(validation["sufficiency_loss"]),
                )

    return best_total, best_action


def validate_resume_gate_config(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    gate_config = payload.get("gate_config", {})

    float_pairs = {
        "target_keep_ratio": args.target_keep_ratio,
        "temperature": args.temperature,
    }
    for key, expected in float_pairs.items():
        if key not in gate_config:
            continue
        actual = float(gate_config[key])
        if abs(actual - float(expected)) > 1e-8:
            raise ValueError(
                f"Resume checkpoint {key} mismatch: "
                f"checkpoint={actual}, requested={expected}."
            )

    if "rescale" in gate_config:
        actual_rescale = bool(gate_config["rescale"])
        if actual_rescale != bool(args.rescale):
            raise ValueError(
                "Resume checkpoint rescale mismatch: "
                f"checkpoint={actual_rescale}, "
                f"requested={bool(args.rescale)}."
            )


def run_validation(
    *,
    policy,
    gate: DynamicChannelGate,
    loader: DataLoader,
    teacher_cache: TeacherActionCache,
    augmenter: MildVisualAugmenter,
    args: argparse.Namespace,
) -> Dict[str, float]:
    was_training = gate.training
    gate.eval()
    metric_rows = []

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if (
                    args.max_validation_batches > 0
                    and batch_index
                    >= args.max_validation_batches
                ):
                    break

                output = forward_v1_batch(
                    policy=policy,
                    visual_gate=gate,
                    batch=batch,
                    teacher_cache=teacher_cache,
                    augmenter=augmenter,
                    base_noise_seed=args.base_noise_seed,
                    augmentation_step=0,
                    target_keep_ratio=args.target_keep_ratio,
                    loss_weights=V1LossWeights(
                        sufficiency=args.lambda_suf,
                        action_invariance=args.lambda_action_inv,
                        mask_invariance=args.lambda_mask_inv,
                        budget=args.lambda_budget,
                    ),
                    smooth_l1_beta=args.smooth_l1_beta,
                )

                batch_metrics = metrics_to_float(output)
                metric_rows.append(batch_metrics)

                # Do not leave the previous validation output alive while
                # constructing the next PiZero forward.
                del output
                del batch_metrics
                del batch

    except torch.cuda.OutOfMemoryError:
        print(
            "[OOM] CUDA OOM during validation; "
            f"memory=[{cuda_memory_summary()}]",
            flush=True,
        )
        cleanup_cuda(empty_cache=True, reset_peak=False)
        raise

    finally:
        gate.train(was_training)

    validation_metrics = average_metric_dicts(metric_rows)

    # Validation uses different allocation shapes from training. Returning
    # unused blocks here reduces allocator fragmentation across epochs.
    cleanup_cuda(empty_cache=True, reset_peak=False)

    return validation_metrics


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.steps_per_epoch <= 0:
        raise ValueError("steps-per-epoch must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.validation_batch_size <= 0:
        raise ValueError(
            "validation-batch-size must be positive."
        )
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient-accumulation-steps must be positive."
        )
    if args.log_every <= 0:
        raise ValueError("log-every must be positive.")
    if args.max_validation_batches < 0:
        raise ValueError(
            "max-validation-batches must be non-negative."
        )

    set_training_seed(args.training_seed)

    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    history_path = osp.join(output_dir, "history.jsonl")

    train_dataset = RolloutQueryDataset(
        args.manifest,
        split="train",
    )
    validation_dataset = RolloutQueryDataset(
        args.manifest,
        split="validation",
    )

    if train_dataset.policy_setup != args.policy_setup:
        raise ValueError(
            "Manifest/policy setup mismatch: "
            f"manifest={train_dataset.policy_setup}, "
            f"requested={args.policy_setup}."
        )
    if validation_dataset.policy_setup != args.policy_setup:
        raise ValueError(
            "Validation manifest/policy setup mismatch."
        )

    train_cache = TeacherActionCache(
        args.train_teacher_cache
    )
    validation_cache = TeacherActionCache(
        args.validation_teacher_cache
    )

    train_cache.validate(
        manifest_path=args.manifest,
        policy_setup=args.policy_setup,
        split="train",
        base_checkpoint=args.checkpoint_path,
        flow_sampling=args.flow_sampling,
        base_noise_seed=args.base_noise_seed,
    )
    validation_cache.validate(
        manifest_path=args.manifest,
        policy_setup=args.policy_setup,
        split="validation",
        base_checkpoint=args.checkpoint_path,
        flow_sampling=args.flow_sampling,
        base_noise_seed=args.base_noise_seed,
    )

    train_sampler = TaskEpisodeQueryBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.sampler_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_rollout_queries,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rollout_queries,
    )

    policy = PiZeroInference(
        cfg_dir=args.cfg_dir,
        checkpoint_path=args.checkpoint_path,
        policy_setup=args.policy_setup,
        flow_sampling=args.flow_sampling,
        use_ddp=False,
        use_naive=False,
        use_torch_compile=False,
        seed=args.training_seed,
    )

    model = unwrap_pizero_model(policy)
    if any(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "PiZero must be frozen before V1 training."
        )

    gate = DynamicChannelGate(
        feature_dim=int(model.image_text_hidden_size),
        proprio_dim=int(model.proprio_dim),
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        target_keep_ratio=args.target_keep_ratio,
        temperature=args.temperature,
        rescale=args.rescale,
    ).to(
        device=policy.device,
        dtype=torch.float32,
    )
    gate.train()

    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    resume_payload = None

    if args.resume is not None:
        resume_payload = load_checkpoint_payload(
            args.resume
        )
        validate_checkpoint_compatibility(
            resume_payload,
            expected_feature_dim=int(
                model.image_text_hidden_size
            ),
            expected_proprio_dim=int(model.proprio_dim),
            expected_num_groups=args.num_groups,
            expected_hidden_dim=args.hidden_dim,
            expected_base_checkpoint=args.checkpoint_path,
        )
        validate_resume_gate_config(
            resume_payload,
            args,
        )

        gate.load_state_dict(
            resume_payload["gate_state_dict"],
            strict=True,
        )
        global_step = restore_optimizer_state(
            optimizer,
            resume_payload,
        )
        start_epoch = (
            int(
                resume_payload
                .get("extra", {})
                .get("epoch", -1)
            )
            + 1
        )

        print(
            f"[RESUME] epoch={start_epoch}, "
            f"global_step={global_step}"
        )

    augmenter = MildVisualAugmenter(
        MildAugmentationConfig(),
        base_seed=args.augmentation_seed,
    )
    weights = V1LossWeights(
        sufficiency=args.lambda_suf,
        action_invariance=args.lambda_action_inv,
        mask_invariance=args.lambda_mask_inv,
        budget=args.lambda_budget,
    )

    policy_config = {
        "policy_setup": args.policy_setup,
        "base_checkpoint": osp.abspath(
            args.checkpoint_path
        ),
        "flow_sampling": args.flow_sampling,
        "horizon_steps": int(model.horizon_steps),
        "action_dim": int(model.action_dim),
    }
    training_config = vars(args).copy()

    history_best_total, history_best_action = (
        read_best_metrics_from_history(
            history_path,
            target_keep_ratio=args.target_keep_ratio,
            budget_tolerance=args.budget_tolerance,
        )
    )

    payload_extra = (
        {}
        if resume_payload is None
        else resume_payload.get("extra", {})
    )

    best_total = min(
        history_best_total,
        float(
            payload_extra.get(
                "best_total_under_budget",
                float("inf"),
            )
        ),
    )
    best_action = min(
        history_best_action,
        float(
            payload_extra.get(
                "best_action_under_budget",
                float("inf"),
            )
        ),
    )

    cleanup_cuda(empty_cache=True, reset_peak=True)

    print(
        "========== Dynamic Gate V1 Training =========="
    )
    print("train queries:", len(train_dataset))
    print(
        "validation queries:",
        len(validation_dataset),
    )
    print("output:", output_dir)
    print(
        "initial memory:",
        cuda_memory_summary(),
    )
    print(
        "best history:",
        f"total={best_total:.8f}",
        f"action={best_action:.8f}",
    )

    if start_epoch >= args.epochs:
        print(
            "[INFO] Resume checkpoint has already reached "
            f"epoch {start_epoch - 1}; requested epochs="
            f"{args.epochs}. Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_metric_rows = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(
            f"[EPOCH] start epoch={epoch:03d} "
            f"memory=[{cuda_memory_summary()}]"
        )

        for batch_index, batch in enumerate(
            train_loader
        ):
            try:
                output = forward_v1_batch(
                    policy=policy,
                    visual_gate=gate,
                    batch=batch,
                    teacher_cache=train_cache,
                    augmenter=augmenter,
                    base_noise_seed=args.base_noise_seed,
                    augmentation_step=(
                        epoch * args.steps_per_epoch
                        + batch_index
                    ),
                    target_keep_ratio=(
                        args.target_keep_ratio
                    ),
                    loss_weights=weights,
                    smooth_l1_beta=args.smooth_l1_beta,
                )

                scaled_loss = (
                    output.losses.total
                    / args.gradient_accumulation_steps
                )

                if not bool(
                    torch.isfinite(scaled_loss).item()
                ):
                    raise FloatingPointError(
                        "Encountered non-finite "
                        "V1 training loss."
                    )

                scaled_loss.backward()

                # Convert every logged value to a plain Python float before
                # deleting the graph-carrying V1ForwardOutput.
                batch_metrics = metrics_to_float(output)
                epoch_metric_rows.append(batch_metrics)

                # Important: remove graph references before the next forward.
                # Python evaluates the right-hand side of `output = ...`
                # before replacing the old `output`, so leaving it bound can
                # increase the next iteration's peak memory.
                del scaled_loss
                del output

            except torch.cuda.OutOfMemoryError:
                print(
                    "[OOM] CUDA OOM during training "
                    f"at epoch={epoch}, "
                    f"batch_index={batch_index}; "
                    f"memory=[{cuda_memory_summary()}]",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                cleanup_cuda(
                    empty_cache=True,
                    reset_peak=False,
                )
                raise

            should_step = (
                (
                    batch_index + 1
                )
                % args.gradient_accumulation_steps
                == 0
                or (
                    batch_index + 1
                    == len(train_loader)
                )
            )

            if should_step:
                verify_frozen_policy_has_no_grad(
                    policy
                )

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        gate.parameters(),
                        max_norm=args.max_grad_norm,
                    )

                grad_norm = gate_gradient_norm(
                    list(gate.parameters())
                )

                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
                global_step += 1

                if (
                    global_step
                    % args.log_every
                    == 0
                ):
                    print(
                        f"[TRAIN] epoch={epoch:03d} "
                        f"step={global_step:06d} "
                        f"total="
                        f"{batch_metrics['total_loss']:.8f} "
                        f"suf="
                        f"{batch_metrics['sufficiency_loss']:.8f} "
                        f"action_inv="
                        f"{batch_metrics['action_invariance_loss']:.8f} "
                        f"mask_inv="
                        f"{batch_metrics['mask_invariance_loss']:.8f} "
                        f"budget="
                        f"{batch_metrics['budget_loss']:.8f} "
                        f"mask_mean="
                        f"{batch_metrics['mask_mean']:.6f} "
                        f"grad_norm={grad_norm:.6f} "
                        f"memory=[{cuda_memory_summary()}]",
                        flush=True,
                    )

            del batch_metrics
            del batch

        train_metrics = average_metric_dicts(
            epoch_metric_rows
        )
        del epoch_metric_rows

        validation_metrics = run_validation(
            policy=policy,
            gate=gate,
            loader=validation_loader,
            teacher_cache=validation_cache,
            augmenter=augmenter,
            args=args,
        )

        under_budget = is_under_budget(
            validation_metrics,
            target_keep_ratio=args.target_keep_ratio,
            tolerance=args.budget_tolerance,
        )

        improved_total = (
            under_budget
            and validation_metrics["total_loss"]
            < best_total
        )
        improved_action = (
            under_budget
            and validation_metrics[
                "sufficiency_loss"
            ]
            < best_action
        )

        if improved_total:
            best_total = float(
                validation_metrics["total_loss"]
            )
        if improved_action:
            best_action = float(
                validation_metrics[
                    "sufficiency_loss"
                ]
            )

        epoch_memory = cuda_memory_stats()

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "under_budget": under_budget,
            "cuda_memory": epoch_memory,
        }
        write_jsonl(history_path, row)

        print(
            f"[VAL] epoch={epoch:03d} "
            f"total="
            f"{validation_metrics['total_loss']:.8f} "
            f"suf="
            f"{validation_metrics['sufficiency_loss']:.8f} "
            f"action_inv="
            f"{validation_metrics['action_invariance_loss']:.8f} "
            f"mask_inv="
            f"{validation_metrics['mask_invariance_loss']:.8f} "
            f"budget="
            f"{validation_metrics['budget_loss']:.8f} "
            f"mask_mean="
            f"{validation_metrics['mask_mean']:.6f} "
            f"within_std="
            f"{validation_metrics['within_sample_group_std']:.6f} "
            f"across_std="
            f"{validation_metrics['across_sample_mask_std']:.6f} "
            f"under_budget={under_budget} "
            f"memory=[{cuda_memory_summary()}]",
            flush=True,
        )

        extra = {
            "epoch": epoch,
            "training_config": training_config,
            "train_metrics": train_metrics,
            "validation_metrics": (
                validation_metrics
            ),
            "under_budget": under_budget,
            "best_total_under_budget": best_total,
            "best_action_under_budget": best_action,
            "cuda_memory": epoch_memory,
            "train_teacher_cache": osp.abspath(
                args.train_teacher_cache
            ),
            "validation_teacher_cache": osp.abspath(
                args.validation_teacher_cache
            ),
        }

        save_dynamic_gate_checkpoint(
            osp.join(output_dir, "last.pt"),
            gate,
            stage="v1",
            policy_config=policy_config,
            global_step=global_step,
            optimizer=optimizer,
            extra=extra,
        )

        if improved_total:
            save_dynamic_gate_checkpoint(
                osp.join(
                    output_dir,
                    "best_total_under_budget.pt",
                ),
                gate,
                stage="v1",
                policy_config=policy_config,
                global_step=global_step,
                optimizer=optimizer,
                extra=extra,
            )
            print(
                "[SAVE] "
                "best_total_under_budget.pt"
            )

        if improved_action:
            save_dynamic_gate_checkpoint(
                osp.join(
                    output_dir,
                    "best_action_under_budget.pt",
                ),
                gate,
                stage="v1",
                policy_config=policy_config,
                global_step=global_step,
                optimizer=optimizer,
                extra=extra,
            )
            print(
                "[SAVE] "
                "best_action_under_budget.pt"
            )

        # Do cleanup only after every checkpoint has finished writing.
        cleanup_cuda(
            empty_cache=True,
            reset_peak=False,
        )

        print(
            f"[EPOCH] end epoch={epoch:03d} "
            f"memory=[{cuda_memory_summary()}]",
            flush=True,
        )

    print(
        "========== V1 training finished =========="
    )


if __name__ == "__main__":
    main()
