# wx:Dynamic gate v3.5

#!/usr/bin/env python3
"""Build global redundant-channel pairs from frozen PiZero projector features.

The script extracts the output of ``multi_modal_projector`` for every query in
one manifest split, pools visual tokens into one signed feature vector per
query, computes the global Pearson correlation matrix across queries, and saves
high-correlation channel pairs for later mask-weighted redundancy training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.channel_redundancy import (  # noqa: E402
    save_channel_redundancy_cache,
)
from feature_gating.fixed_noise_teacher import (  # noqa: E402
    build_initial_action_noise,
    unwrap_pizero_model,
)
from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    collate_rollout_queries,
)
from feature_gating.teacher_cache import normalize_path, sha256_file  # noqa: E402
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
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-setup",
        choices=["google_robot", "widowx_bridge"],
        required=True,
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--flow-sampling", choices=["beta", "uniform"], default="beta")
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-noise-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pooling",
        choices=["mean", "rms"],
        default="mean",
        help="How to pool the 256 projected visual tokens into one vector/query.",
    )
    parser.add_argument("--top-k-per-channel", type=int, default=16)
    parser.add_argument("--min-abs-correlation", type=float, default=0.70)
    parser.add_argument("--weight-power", type=float, default=2.0)
    parser.add_argument(
        "--correlation-device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_projector_features(
    *,
    policy,
    model_inputs: Dict[str, torch.Tensor],
    initial_action_noise: torch.Tensor,
) -> torch.Tensor:
    """Run one frozen PiZero inference and capture multi_modal_projector output."""
    model = unwrap_pizero_model(policy)
    captured: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        if torch.is_tensor(output):
            tensor = output
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            tensor = output[0]
        else:
            raise TypeError(
                "multi_modal_projector output must be a Tensor or tensor-first tuple/list; "
                f"got {type(output)}."
            )
        captured.append(tensor.detach())

    handle = model.multi_modal_projector.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model.infer_action(
                **dict(model_inputs),
                visual_gate=None,
                initial_action_noise=initial_action_noise,
                return_aux=False,
            )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(
            "Expected exactly one multi_modal_projector call per PiZero query batch, "
            f"captured {len(captured)} outputs."
        )

    features = captured[0]
    if features.ndim != 3:
        raise ValueError(
            "Projected visual features must have shape [B,N,D], "
            f"got {tuple(features.shape)}."
        )
    return features


def pool_projector_features(features: torch.Tensor, pooling: str) -> torch.Tensor:
    features = features.float()
    if pooling == "mean":
        return features.mean(dim=1)
    if pooling == "rms":
        return torch.sqrt(features.square().mean(dim=1) + 1e-8)
    raise ValueError(f"Unknown pooling={pooling!r}.")


def compute_correlation_matrix(
    pooled_features: torch.Tensor,
    *,
    device: torch.device,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Pearson correlation across query samples for every channel pair."""
    if pooled_features.ndim != 2:
        raise ValueError(
            f"pooled_features must be [N,D], got {tuple(pooled_features.shape)}."
        )
    if pooled_features.shape[0] < 3:
        raise ValueError("At least three query samples are required for correlation.")

    x = pooled_features.to(device=device, dtype=torch.float64)
    mean = x.mean(dim=0)
    centered = x - mean
    variance = centered.square().sum(dim=0) / float(x.shape[0] - 1)
    std = torch.sqrt(variance.clamp_min(0.0))

    valid = std > eps
    normalized = torch.zeros_like(centered)
    normalized[:, valid] = centered[:, valid] / std[valid]

    correlation = normalized.transpose(0, 1).matmul(normalized)
    correlation = correlation / float(x.shape[0] - 1)
    correlation = correlation.clamp(min=-1.0, max=1.0)

    invalid = ~valid
    if invalid.any():
        correlation[invalid, :] = 0.0
        correlation[:, invalid] = 0.0

    return (
        correlation.float().cpu(),
        mean.float().cpu(),
        std.float().cpu(),
    )


def select_redundant_pairs(
    correlation: torch.Tensor,
    *,
    top_k_per_channel: int,
    min_abs_correlation: float,
    weight_power: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a square [D,D] matrix.")
    if top_k_per_channel <= 0:
        raise ValueError("top_k_per_channel must be positive.")
    if not (0.0 <= min_abs_correlation <= 1.0):
        raise ValueError("min_abs_correlation must be in [0,1].")
    if weight_power <= 0:
        raise ValueError("weight_power must be positive.")

    feature_dim = int(correlation.shape[0])
    k = min(int(top_k_per_channel), feature_dim - 1)
    absolute = correlation.abs().clone()
    absolute.fill_diagonal_(-float("inf"))

    values, indices = torch.topk(absolute, k=k, dim=1)
    pair_to_weight: Dict[Tuple[int, int], float] = {}

    for channel in range(feature_dim):
        for rank in range(k):
            value = float(values[channel, rank])
            if not np.isfinite(value) or value < min_abs_correlation:
                continue
            neighbor = int(indices[channel, rank])
            first, second = sorted((channel, neighbor))
            if first == second:
                continue
            weight = value ** float(weight_power)
            previous = pair_to_weight.get((first, second))
            if previous is None or weight > previous:
                pair_to_weight[(first, second)] = weight

    if not pair_to_weight:
        raise RuntimeError(
            "No redundant channel pairs survived the threshold. Lower "
            "--min-abs-correlation or increase --top-k-per-channel."
        )

    sorted_items = sorted(pair_to_weight.items())
    channel_i = np.asarray([key[0] for key, _ in sorted_items], dtype=np.int64)
    channel_j = np.asarray([key[1] for key, _ in sorted_items], dtype=np.int64)
    weights = np.asarray([weight for _, weight in sorted_items], dtype=np.float32)

    pair_abs_corr = np.power(weights, 1.0 / float(weight_power))
    summary = {
        "num_pairs": float(len(weights)),
        "pair_abs_corr_min": float(pair_abs_corr.min()),
        "pair_abs_corr_mean": float(pair_abs_corr.mean()),
        "pair_abs_corr_max": float(pair_abs_corr.max()),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
    }
    return channel_i, channel_j, weights, summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative.")

    set_seed(args.seed)

    dataset = RolloutQueryDataset(args.manifest, split=args.split)
    if dataset.policy_setup != args.policy_setup:
        raise ValueError(
            f"Manifest policy setup={dataset.policy_setup!r}, "
            f"requested={args.policy_setup!r}."
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
        seed=args.seed,
    )
    model = unwrap_pizero_model(policy)

    pooled_rows: List[torch.Tensor] = []
    sample_ids: List[str] = []
    num_seen = 0

    print("========== Build Channel Redundancy Cache ==========")
    print("dataset queries:", len(dataset))
    print("split:", args.split)
    print("pooling:", args.pooling)

    for batch_index, batch in enumerate(loader):
        remaining = None
        if args.max_samples > 0:
            remaining = args.max_samples - num_seen
            if remaining <= 0:
                break
            if len(batch["sample_id"]) > remaining:
                keep = int(remaining)
                batch = {
                    key: (
                        value[:keep]
                        if torch.is_tensor(value)
                        else value[:keep]
                    )
                    for key, value in batch.items()
                }

        model_inputs = prepare_pizero_batch_inputs(policy, batch)
        initial_action_noise = build_initial_action_noise(
            sample_ids=batch["sample_id"],
            horizon_steps=int(model.horizon_steps),
            action_dim=int(model.action_dim),
            device=policy.device,
            dtype=policy.dtype,
            base_seed=args.base_noise_seed,
        )

        projected = extract_projector_features(
            policy=policy,
            model_inputs=model_inputs,
            initial_action_noise=initial_action_noise,
        )
        pooled = pool_projector_features(projected, args.pooling)
        pooled_rows.append(pooled.cpu())
        sample_ids.extend([str(value) for value in batch["sample_id"]])
        num_seen += int(pooled.shape[0])

        if (batch_index + 1) % 50 == 0:
            print(f"[EXTRACT] batches={batch_index + 1} samples={num_seen}")

        del projected, pooled, model_inputs, initial_action_noise, batch

    pooled_features = torch.cat(pooled_rows, dim=0)
    if pooled_features.shape[0] != len(sample_ids):
        raise RuntimeError("Pooled feature/sample ID count mismatch.")

    print("pooled feature matrix:", tuple(pooled_features.shape))

    correlation_device = torch.device(
        "cuda" if args.correlation_device == "cuda" else "cpu"
    )
    if correlation_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--correlation-device cuda requested but CUDA is unavailable.")

    correlation, channel_mean, channel_std = compute_correlation_matrix(
        pooled_features,
        device=correlation_device,
    )

    channel_i, channel_j, weights, pair_summary = select_redundant_pairs(
        correlation,
        top_k_per_channel=args.top_k_per_channel,
        min_abs_correlation=args.min_abs_correlation,
        weight_power=args.weight_power,
    )

    metadata = {
        "manifest_path": normalize_path(args.manifest),
        "manifest_sha256": sha256_file(normalize_path(args.manifest)),
        "sample_ids_sha256": hash_strings(sample_ids),
        "policy_setup": args.policy_setup,
        "split": args.split,
        "base_checkpoint": normalize_path(args.checkpoint_path),
        "flow_sampling": args.flow_sampling,
        "base_noise_seed": int(args.base_noise_seed),
        "pooling": args.pooling,
        "num_samples": int(pooled_features.shape[0]),
        "num_visual_tokens": int(model.num_image_tokens) if hasattr(model, "num_image_tokens") else None,
        "top_k_per_channel": int(args.top_k_per_channel),
        "min_abs_correlation": float(args.min_abs_correlation),
        "weight_power": float(args.weight_power),
        "pair_summary": pair_summary,
    }

    output_dir = save_channel_redundancy_cache(
        args.output_dir,
        channel_i=channel_i,
        channel_j=channel_j,
        weights=weights,
        channel_mean=channel_mean.numpy(),
        channel_std=channel_std.numpy(),
        metadata=metadata,
        overwrite=args.overwrite,
    )

    print("correlation matrix:", tuple(correlation.shape))
    print("selected pairs:", len(channel_i))
    print("pair summary:", json.dumps(pair_summary, indent=2))
    print("saved:", output_dir)
    print("========== Channel redundancy cache finished ==========")


if __name__ == "__main__":
    main()
