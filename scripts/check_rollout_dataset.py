# wx:Rollout collect

#!/usr/bin/env python3
# scripts/check_rollout_dataset.py

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


REQUIRED_ARRAY_KEYS = [
    "images",
    "next_images",
    "policy_proprios",
    "next_policy_proprios",
    "raw_action_chunks",
    "raw_action_lengths",
    "executed_action_chunks",
    "executed_action_lengths",
    "query_indices",
    "env_step_starts",
    "env_step_ends",
    "terminated",
    "truncated",
    "success_at_end",
    "instruction_changed",
]

REQUIRED_METADATA_KEYS = [
    "task",
    "episode_id",
    "seed",
    "instructions",
    "next_instructions",
    "final_success",
    "terminated",
    "truncated",
    "num_policy_queries",
    "num_environment_steps",
    "policy_name",
    "checkpoint",
    "collection_version",
]


def iter_episode_dirs(root: str) -> List[str]:
    episode_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = osp.basename(dirpath)
        if base.startswith(".episode_"):
            continue
        if "trajectory.npz" in filenames and "metadata.json" in filenames:
            episode_dirs.append(dirpath)
    episode_dirs.sort()
    return episode_dirs


def load_metadata(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_no_object_dtype(arrays: Dict[str, np.ndarray], episode_dir: str) -> None:
    for key, arr in arrays.items():
        if arr.dtype == np.dtype("O"):
            raise AssertionError(f"{episode_dir}: {key} has object dtype.")


def assert_finite(arr: np.ndarray, name: str, episode_dir: str) -> None:
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"{episode_dir}: {name} contains NaN or Inf.")


def check_padding_zero(
    chunks: np.ndarray,
    lengths: np.ndarray,
    name: str,
    episode_dir: str,
) -> None:
    if chunks.ndim != 3:
        raise AssertionError(
            f"{episode_dir}: {name} must be 3D [Q, max_len, dim], got {chunks.shape}."
        )

    q, max_len, _ = chunks.shape

    if lengths.shape != (q,):
        raise AssertionError(
            f"{episode_dir}: {name}_lengths shape mismatch: {lengths.shape}, expected {(q,)}."
        )

    if np.any(lengths < 0) or np.any(lengths > max_len):
        raise AssertionError(
            f"{episode_dir}: {name}_lengths contain invalid values. "
            f"min={lengths.min() if len(lengths) else 'NA'}, max={lengths.max() if len(lengths) else 'NA'}, max_len={max_len}"
        )

    for i, length in enumerate(lengths):
        length = int(length)
        if length < max_len:
            pad = chunks[i, length:, :]
            if not np.allclose(pad, 0.0):
                raise AssertionError(
                    f"{episode_dir}: {name} padding after length={length} is not all zero at query {i}."
                )


def parse_success_from_log(result_dir: str, episode_id: int) -> Optional[bool]:
    """
    Optional log check. Searches result_dir/000*.log for:
        Episode {episode_id} finished with success True/False.
    """
    if result_dir is None:
        return None

    if not osp.isdir(result_dir):
        return None

    log_files = [
        osp.join(result_dir, f)
        for f in os.listdir(result_dir)
        if f.startswith("000") and f.endswith(".log")
    ]

    if not log_files:
        return None

    pattern = re.compile(
        rf"Episode\s+{episode_id}\s+finished\s+with\s+success\s+(True|False)",
        re.IGNORECASE,
    )

    for log_path in log_files:
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        match = pattern.search(text)
        if match:
            return match.group(1).lower() == "true"

    return None


def save_sample_pair(
    image: np.ndarray,
    next_image: np.ndarray,
    out_path: str,
) -> None:
    try:
        from PIL import Image
    except Exception:
        print("[WARN] PIL is not installed; skip sample image export.")
        return

    if image.dtype != np.uint8 or next_image.dtype != np.uint8:
        print("[WARN] sample images are not uint8; skip export.")
        return

    panel = np.concatenate([image, next_image], axis=1)
    Image.fromarray(panel).save(out_path)


def check_episode(
    episode_dir: str,
    result_dir: Optional[str] = None,
) -> Dict:
    npz_path = osp.join(episode_dir, "trajectory.npz")
    meta_path = osp.join(episode_dir, "metadata.json")

    data = np.load(npz_path, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    metadata = load_metadata(meta_path)

    for key in REQUIRED_ARRAY_KEYS:
        if key not in arrays:
            raise AssertionError(f"{episode_dir}: missing array key {key}")

    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            raise AssertionError(f"{episode_dir}: missing metadata key {key}")

    assert_no_object_dtype(arrays, episode_dir)

    images = arrays["images"]
    next_images = arrays["next_images"]
    policy_proprios = arrays["policy_proprios"]
    next_policy_proprios = arrays["next_policy_proprios"]
    raw_action_chunks = arrays["raw_action_chunks"]
    raw_action_lengths = arrays["raw_action_lengths"]
    executed_action_chunks = arrays["executed_action_chunks"]
    executed_action_lengths = arrays["executed_action_lengths"]
    query_indices = arrays["query_indices"]
    env_step_starts = arrays["env_step_starts"]
    env_step_ends = arrays["env_step_ends"]
    instruction_changed = arrays["instruction_changed"]

    q = int(metadata["num_policy_queries"])

    first_dim_keys = [
        "images",
        "next_images",
        "policy_proprios",
        "next_policy_proprios",
        "raw_action_chunks",
        "raw_action_lengths",
        "executed_action_chunks",
        "executed_action_lengths",
        "query_indices",
        "env_step_starts",
        "env_step_ends",
        "terminated",
        "truncated",
        "success_at_end",
        "instruction_changed",
    ]

    for key in first_dim_keys:
        if arrays[key].shape[0] != q:
            raise AssertionError(
                f"{episode_dir}: {key}.shape[0]={arrays[key].shape[0]} != num_policy_queries={q}"
            )

    if images.dtype != np.uint8:
        raise AssertionError(f"{episode_dir}: images dtype must be uint8, got {images.dtype}")
    if next_images.dtype != np.uint8:
        raise AssertionError(f"{episode_dir}: next_images dtype must be uint8, got {next_images.dtype}")

    if images.ndim != 4 or images.shape[-1] != 3:
        raise AssertionError(f"{episode_dir}: images must be [Q,H,W,3], got {images.shape}")
    if next_images.ndim != 4 or next_images.shape[-1] != 3:
        raise AssertionError(f"{episode_dir}: next_images must be [Q,H,W,3], got {next_images.shape}")

    if policy_proprios.dtype != np.float32:
        raise AssertionError(
            f"{episode_dir}: policy_proprios dtype must be float32, got {policy_proprios.dtype}"
        )
    if next_policy_proprios.dtype != np.float32:
        raise AssertionError(
            f"{episode_dir}: next_policy_proprios dtype must be float32, got {next_policy_proprios.dtype}"
        )
    if raw_action_chunks.dtype != np.float32:
        raise AssertionError(
            f"{episode_dir}: raw_action_chunks dtype must be float32, got {raw_action_chunks.dtype}"
        )
    if executed_action_chunks.dtype != np.float32:
        raise AssertionError(
            f"{episode_dir}: executed_action_chunks dtype must be float32, got {executed_action_chunks.dtype}"
        )

    for key, arr in arrays.items():
        assert_finite(arr, key, episode_dir)

    check_padding_zero(
        raw_action_chunks,
        raw_action_lengths,
        "raw_action_chunks",
        episode_dir,
    )
    check_padding_zero(
        executed_action_chunks,
        executed_action_lengths,
        "executed_action_chunks",
        episode_dir,
    )

    expected_query_indices = np.arange(q, dtype=np.int32)
    if not np.array_equal(query_indices, expected_query_indices):
        raise AssertionError(
            f"{episode_dir}: query_indices are not continuous from 0. "
            f"got={query_indices}, expected={expected_query_indices}"
        )

    if np.any(env_step_ends < env_step_starts):
        raise AssertionError(
            f"{episode_dir}: env_step_ends contains value smaller than env_step_starts."
        )

    if q > 1:
        if np.any(env_step_starts[1:] < env_step_ends[:-1]):
            raise AssertionError(
                f"{episode_dir}: env step intervals overlap or are not monotonic."
            )

    if len(metadata["instructions"]) != q:
        raise AssertionError(
            f"{episode_dir}: len(instructions)={len(metadata['instructions'])} != Q={q}"
        )
    if len(metadata["next_instructions"]) != q:
        raise AssertionError(
            f"{episode_dir}: len(next_instructions)={len(metadata['next_instructions'])} != Q={q}"
        )

    log_success = parse_success_from_log(result_dir, int(metadata["episode_id"]))
    if log_success is not None and bool(metadata["final_success"]) != bool(log_success):
        raise AssertionError(
            f"{episode_dir}: final_success={metadata['final_success']} "
            f"does not match log success={log_success}."
        )

    return {
        "episode_dir": episode_dir,
        "task": metadata["task"],
        "episode_id": int(metadata["episode_id"]),
        "num_policy_queries": q,
        "num_environment_steps": int(metadata["num_environment_steps"]),
        "final_success": bool(metadata["final_success"]),
        "num_instruction_changed": int(instruction_changed.sum()),
        "image_shape": tuple(images.shape[1:]),
        "proprio_dim": int(policy_proprios.shape[-1]) if policy_proprios.ndim >= 2 else 0,
        "raw_action_shape": tuple(raw_action_chunks.shape[1:]),
        "executed_action_shape": tuple(executed_action_chunks.shape[1:]),
        "mean_executed_len": float(executed_action_lengths.mean()) if q > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--export-samples", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episode_dirs = iter_episode_dirs(args.root)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    if len(episode_dirs) == 0:
        raise RuntimeError(f"No episode rollout files found under: {args.root}")

    print(f"[INFO] Found {len(episode_dirs)} episode directories.")

    summaries = []
    for episode_dir in episode_dirs:
        summary = check_episode(episode_dir, result_dir=args.result_dir)
        summaries.append(summary)

    total_queries = sum(s["num_policy_queries"] for s in summaries)
    total_env_steps = sum(s["num_environment_steps"] for s in summaries)
    total_instruction_changed = sum(s["num_instruction_changed"] for s in summaries)
    success_rate = sum(s["final_success"] for s in summaries) / max(1, len(summaries))

    print("\n========== Rollout Dataset Summary ==========")
    print(f"episodes: {len(summaries)}")
    print(f"total_policy_queries: {total_queries}")
    print(f"total_environment_steps: {total_env_steps}")
    print(f"success_rate: {success_rate:.4f}")
    print(f"instruction_changed_queries: {total_instruction_changed}")
    print(f"avg_queries_per_episode: {total_queries / max(1, len(summaries)):.2f}")
    print(f"avg_env_steps_per_episode: {total_env_steps / max(1, len(summaries)):.2f}")

    tasks = sorted(set(s["task"] for s in summaries))
    print(f"tasks: {tasks}")

    print("\nFirst episode summary:")
    for k, v in summaries[0].items():
        print(f"  {k}: {v}")

    if args.export_samples is not None:
        os.makedirs(args.export_samples, exist_ok=True)

        rng = random.Random(args.seed)
        chosen = rng.choice(episode_dirs)
        data = np.load(osp.join(chosen, "trajectory.npz"), allow_pickle=False)
        q = data["images"].shape[0]

        if q > 0:
            query_id = rng.randrange(q)
            out_path = osp.join(
                args.export_samples,
                f"sample_{osp.basename(chosen)}_query_{query_id:04d}.png",
            )
            save_sample_pair(
                data["images"][query_id],
                data["next_images"][query_id],
                out_path,
            )
            print(f"\n[INFO] Exported sample image pair to: {out_path}")

    print("\n[OK] Rollout dataset check passed.")


if __name__ == "__main__":
    main()