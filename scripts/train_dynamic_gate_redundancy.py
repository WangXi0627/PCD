# wx:Dynamic gate v3.5

#!/usr/bin/env python3
"""Train a channel-level Dynamic Gate with progressive redundancy objectives."""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

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
from feature_gating.channel_redundancy import (  # noqa: E402
    ChannelRedundancyCache,
)
from feature_gating.checkpoint import (  # noqa: E402
    load_checkpoint_payload,
    restore_optimizer_state,
    save_dynamic_gate_checkpoint,
    validate_checkpoint_compatibility,
)
from feature_gating.dynamic_channel_gate import DynamicChannelGate  # noqa: E402
from feature_gating.fixed_noise_teacher import unwrap_pizero_model  # noqa: E402
from feature_gating.redundancy_training_step import (  # noqa: E402
    RedundancyLossWeights,
    forward_redundancy_batch,
    gate_gradient_norm,
    metrics_to_float,
    verify_frozen_policy_has_no_grad,
)
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
)
from feature_gating.samplers import TaskEpisodeQueryBatchSampler  # noqa: E402
from feature_gating.teacher_cache import TeacherActionCache  # noqa: E402
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
    parser.add_argument("--redundancy-cache", default=None)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-setup",
        choices=["google_robot", "widowx_bridge"],
        required=True,
    )
    parser.add_argument("--flow-sampling", choices=["beta", "uniform"], default="beta")
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--max-validation-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Channel-level gate: num_groups must equal feature_dim (2048 for PiZero).
    parser.add_argument("--num-groups", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--target-keep-ratio", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale", action="store_true")

    parser.add_argument("--lambda-suf", type=float, default=1.0)
    parser.add_argument("--lambda-redundancy", type=float, default=0.0)
    parser.add_argument("--lambda-action-inv", type=float, default=0.0)
    parser.add_argument("--lambda-mask-inv", type=float, default=0.0)
    parser.add_argument("--lambda-budget", type=float, default=10.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)

    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--budget-tolerance", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_metric_dicts(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    count = 0
    for row in rows:
        count += 1
        for key, value in row.items():
            totals[key] += float(value)
    if count == 0:
        raise RuntimeError("No metric rows were provided.")
    return {key: value / count for key, value in totals.items()}


def write_jsonl(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    gib = float(1024**3)
    return (
        f"allocated={torch.cuda.memory_allocated()/gib:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved()/gib:.2f}GiB "
        f"peak_allocated={torch.cuda.max_memory_allocated()/gib:.2f}GiB "
        f"peak_reserved={torch.cuda.max_memory_reserved()/gib:.2f}GiB"
    )


def cleanup_cuda(*, reset_peak: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()


def under_budget(metrics: Mapping[str, float], args: argparse.Namespace) -> bool:
    return (
        abs(float(metrics["mask_mean"]) - float(args.target_keep_ratio))
        <= float(args.budget_tolerance)
    )


def read_best_from_history(
    history_path: str,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    best_total = float("inf")
    best_action = float("inf")
    best_redundancy = float("inf")
    if not osp.isfile(history_path):
        return best_total, best_action, best_redundancy

    with open(history_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            validation = row.get("validation")
            if not isinstance(validation, Mapping) or not under_budget(validation, args):
                continue
            best_total = min(best_total, float(validation["total_loss"]))
            best_action = min(best_action, float(validation["sufficiency_loss"]))
            best_redundancy = min(
                best_redundancy,
                float(validation.get("redundancy_loss", float("inf"))),
            )
    return best_total, best_action, best_redundancy


def validate_resume_config(payload: Mapping[str, Any], args: argparse.Namespace) -> None:
    config = payload.get("gate_config", {})
    for key, expected in {
        "target_keep_ratio": args.target_keep_ratio,
        "temperature": args.temperature,
    }.items():
        if key in config and abs(float(config[key]) - float(expected)) > 1e-8:
            raise ValueError(
                f"Resume checkpoint {key} mismatch: "
                f"checkpoint={config[key]}, requested={expected}."
            )
    if "rescale" in config and bool(config["rescale"]) != bool(args.rescale):
        raise ValueError("Resume checkpoint rescale mismatch.")


def run_validation(
    *,
    policy,
    gate: DynamicChannelGate,
    loader: DataLoader,
    teacher_cache: TeacherActionCache,
    redundancy_cache: Optional[ChannelRedundancyCache],
    augmenter: MildVisualAugmenter,
    weights: RedundancyLossWeights,
    args: argparse.Namespace,
) -> Dict[str, float]:
    was_training = gate.training
    gate.eval()
    rows = []
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_validation_batches > 0 and batch_index >= args.max_validation_batches:
                    break
                output = forward_redundancy_batch(
                    policy=policy,
                    visual_gate=gate,
                    batch=batch,
                    teacher_cache=teacher_cache,
                    redundancy_cache=redundancy_cache,
                    augmenter=augmenter,
                    base_noise_seed=args.base_noise_seed,
                    augmentation_step=0,
                    target_keep_ratio=args.target_keep_ratio,
                    loss_weights=weights,
                    smooth_l1_beta=args.smooth_l1_beta,
                )
                rows.append(metrics_to_float(output))
                del output, batch
    finally:
        gate.train(was_training)
    metrics = average_metric_dicts(rows)
    cleanup_cuda(reset_peak=False)
    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("epochs and steps-per-epoch must be positive.")
    if args.batch_size <= 0 or args.validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive.")
    if args.lambda_redundancy < 0:
        raise ValueError("lambda-redundancy must be non-negative.")

    set_training_seed(args.training_seed)
    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    history_path = osp.join(output_dir, "history.jsonl")

    train_dataset = RolloutQueryDataset(args.manifest, split="train")
    validation_dataset = RolloutQueryDataset(args.manifest, split="validation")
    if train_dataset.policy_setup != args.policy_setup:
        raise ValueError("Manifest/policy setup mismatch.")

    train_teacher_cache = TeacherActionCache(args.train_teacher_cache)
    validation_teacher_cache = TeacherActionCache(args.validation_teacher_cache)
    train_teacher_cache.validate(
        manifest_path=args.manifest,
        policy_setup=args.policy_setup,
        split="train",
        base_checkpoint=args.checkpoint_path,
        flow_sampling=args.flow_sampling,
        base_noise_seed=args.base_noise_seed,
    )
    validation_teacher_cache.validate(
        manifest_path=args.manifest,
        policy_setup=args.policy_setup,
        split="validation",
        base_checkpoint=args.checkpoint_path,
        flow_sampling=args.flow_sampling,
        base_noise_seed=args.base_noise_seed,
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
    feature_dim = int(model.image_text_hidden_size)
    proprio_dim = int(model.proprio_dim)

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("PiZero must remain frozen.")
    if args.num_groups != feature_dim:
        raise ValueError(
            "This script implements true dimension-level selection, so "
            f"--num-groups must equal feature_dim={feature_dim}; got {args.num_groups}."
        )

    redundancy_cache: Optional[ChannelRedundancyCache] = None
    if args.lambda_redundancy > 0:
        if args.redundancy_cache is None:
            raise ValueError("--redundancy-cache is required when lambda-redundancy > 0.")
        redundancy_cache = ChannelRedundancyCache(args.redundancy_cache)
        redundancy_cache.validate(
            manifest_path=args.manifest,
            policy_setup=args.policy_setup,
            split="train",
            base_checkpoint=args.checkpoint_path,
            flow_sampling=args.flow_sampling,
            feature_dim=feature_dim,
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

    gate = DynamicChannelGate(
        feature_dim=feature_dim,
        proprio_dim=proprio_dim,
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        target_keep_ratio=args.target_keep_ratio,
        temperature=args.temperature,
        rescale=args.rescale,
    ).to(device=policy.device, dtype=torch.float32)
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
        resume_payload = load_checkpoint_payload(args.resume)
        validate_checkpoint_compatibility(
            resume_payload,
            expected_feature_dim=feature_dim,
            expected_proprio_dim=proprio_dim,
            expected_num_groups=args.num_groups,
            expected_hidden_dim=args.hidden_dim,
            expected_base_checkpoint=args.checkpoint_path,
        )
        validate_resume_config(resume_payload, args)
        gate.load_state_dict(resume_payload["gate_state_dict"], strict=True)
        global_step = restore_optimizer_state(optimizer, resume_payload)
        start_epoch = int(resume_payload.get("extra", {}).get("epoch", -1)) + 1
        print(f"[RESUME] epoch={start_epoch}, global_step={global_step}")

    augmenter = MildVisualAugmenter(
        MildAugmentationConfig(),
        base_seed=args.augmentation_seed,
    )
    weights = RedundancyLossWeights(
        sufficiency=args.lambda_suf,
        redundancy=args.lambda_redundancy,
        action_invariance=args.lambda_action_inv,
        mask_invariance=args.lambda_mask_inv,
        budget=args.lambda_budget,
    )

    policy_config = {
        "policy_setup": args.policy_setup,
        "base_checkpoint": osp.abspath(args.checkpoint_path),
        "flow_sampling": args.flow_sampling,
        "horizon_steps": int(model.horizon_steps),
        "action_dim": int(model.action_dim),
    }
    training_config = vars(args).copy()

    best_total, best_action, best_redundancy = read_best_from_history(
        history_path,
        args,
    )
    if resume_payload is not None:
        extra = resume_payload.get("extra", {})
        best_total = min(best_total, float(extra.get("best_total", float("inf"))))
        best_action = min(best_action, float(extra.get("best_action", float("inf"))))
        best_redundancy = min(
            best_redundancy,
            float(extra.get("best_redundancy", float("inf"))),
        )

    cleanup_cuda(reset_peak=True)
    print("========== Channel-Level Redundancy Gate Training ==========")
    print("train queries:", len(train_dataset))
    print("validation queries:", len(validation_dataset))
    print("feature_dim / num_groups:", feature_dim, "/", args.num_groups)
    print("use augmentation branch:", args.lambda_action_inv > 0 or args.lambda_mask_inv > 0)
    print("loss weights:", weights)
    print("output:", output_dir)
    print("initial memory:", cuda_memory_summary())

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        train_rows = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for batch_index, batch in enumerate(train_loader):
            output = forward_redundancy_batch(
                policy=policy,
                visual_gate=gate,
                batch=batch,
                teacher_cache=train_teacher_cache,
                redundancy_cache=redundancy_cache,
                augmenter=augmenter,
                base_noise_seed=args.base_noise_seed,
                augmentation_step=epoch * args.steps_per_epoch + batch_index,
                target_keep_ratio=args.target_keep_ratio,
                loss_weights=weights,
                smooth_l1_beta=args.smooth_l1_beta,
            )
            scaled_loss = output.losses.total / args.gradient_accumulation_steps
            if not bool(torch.isfinite(scaled_loss).item()):
                raise FloatingPointError("Encountered non-finite training loss.")
            scaled_loss.backward()

            batch_metrics = metrics_to_float(output)
            train_rows.append(batch_metrics)
            del scaled_loss, output, batch

            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(train_loader)
            )
            if should_step:
                verify_frozen_policy_has_no_grad(policy)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        gate.parameters(),
                        max_norm=args.max_grad_norm,
                    )
                grad_norm = gate_gradient_norm(list(gate.parameters()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    print(
                        f"[TRAIN] epoch={epoch:03d} step={global_step:06d} "
                        f"total={batch_metrics['total_loss']:.8f} "
                        f"suf={batch_metrics['sufficiency_loss']:.8f} "
                        f"red={batch_metrics['redundancy_loss']:.6f} "
                        f"wred={batch_metrics['weighted_redundancy']:.8f} "
                        f"action_inv={batch_metrics['action_invariance_loss']:.8f} "
                        f"mask_inv={batch_metrics['mask_invariance_loss']:.8f} "
                        f"budget={batch_metrics['budget_loss']:.8f} "
                        f"mask_mean={batch_metrics['mask_mean']:.6f} "
                        f"eff_ch={batch_metrics['effective_num_channels']:.2f} "
                        f"grad_norm={grad_norm:.6f} "
                        f"memory=[{cuda_memory_summary()}]",
                        flush=True,
                    )
            del batch_metrics

        train_metrics = average_metric_dicts(train_rows)
        del train_rows

        validation_metrics = run_validation(
            policy=policy,
            gate=gate,
            loader=validation_loader,
            teacher_cache=validation_teacher_cache,
            redundancy_cache=redundancy_cache,
            augmenter=augmenter,
            weights=weights,
            args=args,
        )

        budget_ok = under_budget(validation_metrics, args)
        improved_total = budget_ok and validation_metrics["total_loss"] < best_total
        improved_action = budget_ok and validation_metrics["sufficiency_loss"] < best_action
        improved_redundancy = (
            args.lambda_redundancy > 0
            and budget_ok
            and validation_metrics["redundancy_loss"] < best_redundancy
        )
        if improved_total:
            best_total = float(validation_metrics["total_loss"])
        if improved_action:
            best_action = float(validation_metrics["sufficiency_loss"])
        if improved_redundancy:
            best_redundancy = float(validation_metrics["redundancy_loss"])

        print(
            f"[VAL] epoch={epoch:03d} "
            f"total={validation_metrics['total_loss']:.8f} "
            f"suf={validation_metrics['sufficiency_loss']:.8f} "
            f"red={validation_metrics['redundancy_loss']:.6f} "
            f"wred={validation_metrics['weighted_redundancy']:.8f} "
            f"action_inv={validation_metrics['action_invariance_loss']:.8f} "
            f"mask_inv={validation_metrics['mask_invariance_loss']:.8f} "
            f"budget={validation_metrics['budget_loss']:.8f} "
            f"mask_mean={validation_metrics['mask_mean']:.6f} "
            f"channel_std={validation_metrics['within_sample_channel_std']:.6f} "
            f"eff_ch={validation_metrics['effective_num_channels']:.2f} "
            f"budget_ok={budget_ok}",
            flush=True,
        )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "budget_ok": budget_ok,
        }
        write_jsonl(history_path, row)

        extra = {
            "epoch": epoch,
            "training_config": training_config,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "best_total": best_total,
            "best_action": best_action,
            "best_redundancy": best_redundancy,
            "redundancy_cache": (
                None if args.redundancy_cache is None else osp.abspath(args.redundancy_cache)
            ),
        }
        save_dynamic_gate_checkpoint(
            osp.join(output_dir, "last.pt"),
            gate,
            stage="channel_redundancy_v1",
            policy_config=policy_config,
            global_step=global_step,
            optimizer=optimizer,
            extra=extra,
        )
        if improved_total:
            save_dynamic_gate_checkpoint(
                osp.join(output_dir, "best_total_under_budget.pt"),
                gate,
                stage="channel_redundancy_v1",
                policy_config=policy_config,
                global_step=global_step,
                optimizer=optimizer,
                extra=extra,
            )
            print("[SAVE] best_total_under_budget.pt")
        if improved_action:
            save_dynamic_gate_checkpoint(
                osp.join(output_dir, "best_action_under_budget.pt"),
                gate,
                stage="channel_redundancy_v1",
                policy_config=policy_config,
                global_step=global_step,
                optimizer=optimizer,
                extra=extra,
            )
            print("[SAVE] best_action_under_budget.pt")
        if improved_redundancy:
            save_dynamic_gate_checkpoint(
                osp.join(output_dir, "best_redundancy_under_budget.pt"),
                gate,
                stage="channel_redundancy_v1",
                policy_config=policy_config,
                global_step=global_step,
                optimizer=optimizer,
                extra=extra,
            )
            print("[SAVE] best_redundancy_under_budget.pt")

        cleanup_cuda(reset_peak=False)
        print(f"[EPOCH] end epoch={epoch:03d} memory=[{cuda_memory_summary()}]")

    print("========== Channel-level training finished ==========")


if __name__ == "__main__":
    main()
