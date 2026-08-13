# wx:Dynamic gate v3

#!/usr/bin/env python3
"""Evaluate a Dynamic Gate V1 checkpoint on an offline manifest split."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import defaultdict
from typing import Dict, Mapping

import torch
from torch.utils.data import DataLoader


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.augmentations import MildVisualAugmenter  # noqa: E402
from feature_gating.checkpoint import (  # noqa: E402
    build_gate_from_checkpoint,
    validate_checkpoint_compatibility,
)
from feature_gating.fixed_noise_teacher import unwrap_pizero_model  # noqa: E402
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
)
from feature_gating.teacher_cache import TeacherActionCache  # noqa: E402
from feature_gating.v1_losses import V1LossWeights  # noqa: E402
from feature_gating.v1_training_step import forward_v1_batch, metrics_to_float  # noqa: E402
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
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--policy-setup", choices=["google_robot", "widowx_bridge"], required=True)
    parser.add_argument("--flow-sampling", choices=["beta", "uniform"], default="beta")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--lambda-suf", type=float, default=1.0)
    parser.add_argument("--lambda-action-inv", type=float, default=1.0)
    parser.add_argument("--lambda-mask-inv", type=float, default=0.1)
    parser.add_argument("--lambda-budget", type=float, default=10.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RolloutQueryDataset(args.manifest, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rollout_queries,
    )
    teacher_cache = TeacherActionCache(args.teacher_cache)
    teacher_cache.validate(
        manifest_path=args.manifest,
        policy_setup=args.policy_setup,
        split=args.split,
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
        seed=0,
    )
    model = unwrap_pizero_model(policy)
    gate, payload = build_gate_from_checkpoint(
        args.gate_checkpoint,
        device=policy.device,
        dtype=torch.float32,
    )
    validate_checkpoint_compatibility(
        payload,
        expected_feature_dim=int(model.image_text_hidden_size),
        expected_proprio_dim=int(model.proprio_dim),
        expected_base_checkpoint=args.checkpoint_path,
    )
    gate.eval()

    gate_config = payload["gate_config"]
    weights = V1LossWeights(
        sufficiency=args.lambda_suf,
        action_invariance=args.lambda_action_inv,
        mask_invariance=args.lambda_mask_inv,
        budget=args.lambda_budget,
    )
    augmenter = MildVisualAugmenter(base_seed=args.augmentation_seed)

    totals: Dict[str, float] = defaultdict(float)
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            output = forward_v1_batch(
                policy=policy,
                visual_gate=gate,
                batch=batch,
                teacher_cache=teacher_cache,
                augmenter=augmenter,
                base_noise_seed=args.base_noise_seed,
                augmentation_step=0,
                target_keep_ratio=float(gate_config["target_keep_ratio"]),
                loss_weights=weights,
                smooth_l1_beta=args.smooth_l1_beta,
            )
            metrics = metrics_to_float(output)
            for key, value in metrics.items():
                totals[key] += value
            count += 1

    if count == 0:
        raise RuntimeError("No evaluation batches were processed.")
    averaged = {key: value / count for key, value in totals.items()}
    result = {
        "manifest": osp.abspath(args.manifest),
        "split": args.split,
        "gate_checkpoint": osp.abspath(args.gate_checkpoint),
        "num_batches": count,
        "metrics": averaged,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        output_path = osp.abspath(args.output_json)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print("[OK] Saved:", output_path)


if __name__ == "__main__":
    main()
