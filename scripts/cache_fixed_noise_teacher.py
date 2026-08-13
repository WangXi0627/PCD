# wx:Dynamic gate v3

#!/usr/bin/env python3
"""Cache fixed-noise frozen PiZero teacher actions for one manifest split."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.fixed_noise_teacher import (  # noqa: E402
    build_initial_action_noise,
    compute_teacher_action,
    stable_noise_seed,
    unwrap_pizero_model,
)
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
)
from feature_gating.teacher_cache import (  # noqa: E402
    save_teacher_cache,
    sha256_file,
)
from feature_gating.training_step import prepare_pizero_batch_inputs  # noqa: E402
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
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-setup",
        choices=["google_robot", "widowx_bridge"],
        required=True,
    )
    parser.add_argument("--flow-sampling", choices=["beta", "uniform"], default="beta")
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = RolloutQueryDataset(
        args.manifest,
        split=args.split,
        filter_instruction_changed=False,
        max_cached_episodes=2,
    )
    if dataset.policy_setup != args.policy_setup:
        raise ValueError(
            f"Manifest policy_setup={dataset.policy_setup!r} does not match "
            f"--policy-setup={args.policy_setup!r}."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
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
        seed=0,
    )
    model = unwrap_pizero_model(policy)

    sample_ids: List[str] = []
    noise_seeds: List[int] = []
    teacher_action_batches: List[np.ndarray] = []
    processed = 0

    print("========== Cache Fixed-Noise Teacher ==========")
    print("manifest:", osp.abspath(args.manifest))
    print("split:", args.split)
    print("dataset queries:", len(dataset))
    print("output:", osp.abspath(args.output_dir))

    for batch_index, batch in enumerate(loader):
        if args.max_samples > 0:
            remaining = args.max_samples - processed
            if remaining <= 0:
                break
            if len(batch["sample_id"]) > remaining:
                keep = remaining
                batch = {
                    key: (
                        value[:keep]
                        if isinstance(value, list) or torch.is_tensor(value)
                        else value
                    )
                    for key, value in batch.items()
                }

        model_inputs = prepare_pizero_batch_inputs(policy, batch)
        initial_noise = build_initial_action_noise(
            sample_ids=batch["sample_id"],
            horizon_steps=int(model.horizon_steps),
            action_dim=int(model.action_dim),
            device=policy.device,
            dtype=policy.dtype,
            base_seed=args.base_noise_seed,
        )
        teacher_action = compute_teacher_action(
            policy,
            model_inputs,
            initial_noise,
        )

        batch_ids = [str(value) for value in batch["sample_id"]]
        sample_ids.extend(batch_ids)
        noise_seeds.extend(
            stable_noise_seed(value, base_seed=args.base_noise_seed)
            for value in batch_ids
        )
        teacher_action_batches.append(
            teacher_action.detach().float().cpu().numpy()
        )
        processed += len(batch_ids)

        if batch_index == 0:
            repeated = compute_teacher_action(
                policy,
                model_inputs,
                initial_noise,
            )
            if not torch.equal(teacher_action, repeated):
                difference = float(
                    (teacher_action.float() - repeated.float())
                    .abs()
                    .max()
                    .cpu()
                )
                raise AssertionError(
                    "Teacher action is not deterministic under fixed noise; "
                    f"max_abs_diff={difference}."
                )

        if processed % 100 == 0 or processed == len(dataset):
            print(f"[CACHE] processed={processed}")

    if not teacher_action_batches:
        raise RuntimeError("No teacher actions were generated.")

    teacher_actions = np.concatenate(teacher_action_batches, axis=0)
    metadata = {
        "manifest_path": osp.abspath(args.manifest),
        "manifest_sha256": sha256_file(osp.abspath(args.manifest)),
        "policy_setup": args.policy_setup,
        "split": args.split,
        "base_checkpoint": osp.abspath(args.checkpoint_path),
        "flow_sampling": args.flow_sampling,
        "base_noise_seed": int(args.base_noise_seed),
        "dtype": "float32",
    }

    cache_dir = save_teacher_cache(
        args.output_dir,
        sample_ids=sample_ids,
        noise_seeds=noise_seeds,
        teacher_actions=teacher_actions,
        metadata=metadata,
        overwrite=args.overwrite,
    )

    print("[OK] Teacher cache saved:", cache_dir)
    print("samples:", len(sample_ids))
    print("action shape:", teacher_actions.shape)


if __name__ == "__main__":
    main()
