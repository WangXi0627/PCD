# wx:Dynamic gate v2

#!/usr/bin/env python3
"""Validate a dynamic-gate manifest and all referenced rollout episodes."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Set, Tuple

import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.rollout_dataset import (  # noqa: E402
    RolloutQueryDataset,
    load_dynamic_gate_manifest,
)


REQUIRED_ARRAY_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--max-samples-to-read", type=int, default=32)
    return parser.parse_args()


def resolve_episode_dir(rollout_root: str, record: Dict[str, Any]) -> str:
    relpath = record.get("episode_relpath", record.get("episode_dir"))
    if relpath is None:
        raise KeyError(f"Episode record is missing episode path: {record}")
    if osp.isabs(str(relpath)):
        return str(relpath)
    return osp.abspath(osp.join(rollout_root, str(relpath)))


def check_episode(episode_dir: str, expected_record: Dict[str, Any]) -> Tuple[int, int]:
    trajectory_path = osp.join(episode_dir, "trajectory.npz")
    metadata_path = osp.join(episode_dir, "metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with np.load(trajectory_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    for key in REQUIRED_ARRAY_KEYS:
        if key not in arrays:
            raise KeyError(f"{episode_dir}: missing array key {key}.")

    num_queries = int(metadata["num_policy_queries"])
    if num_queries != int(expected_record["num_policy_queries"]):
        raise ValueError(
            f"{episode_dir}: manifest num_policy_queries="
            f"{expected_record['num_policy_queries']} != metadata={num_queries}."
        )

    if str(metadata["task"]) != str(expected_record["task"]):
        raise ValueError(f"{episode_dir}: task mismatch.")
    if int(metadata["episode_id"]) != int(expected_record["episode_id"]):
        raise ValueError(f"{episode_dir}: episode_id mismatch.")
    if bool(metadata["final_success"]) != bool(expected_record["final_success"]):
        raise ValueError(f"{episode_dir}: final_success mismatch.")
    if len(metadata["instructions"]) != num_queries:
        raise ValueError(f"{episode_dir}: instructions length mismatch.")

    for key in REQUIRED_ARRAY_KEYS:
        if arrays[key].shape[0] != num_queries:
            raise ValueError(
                f"{episode_dir}: {key}.shape[0]={arrays[key].shape[0]} "
                f"!= num_queries={num_queries}."
            )
        if arrays[key].dtype == np.dtype("O"):
            raise ValueError(f"{episode_dir}: {key} has object dtype.")
        if np.issubdtype(arrays[key].dtype, np.floating):
            if not np.all(np.isfinite(arrays[key])):
                raise ValueError(f"{episode_dir}: {key} contains NaN/Inf.")

    if arrays["images"].dtype != np.uint8:
        raise ValueError(f"{episode_dir}: images must be uint8.")
    if arrays["policy_proprios"].dtype != np.float32:
        raise ValueError(f"{episode_dir}: policy_proprios must be float32.")
    if arrays["raw_action_chunks"].dtype != np.float32:
        raise ValueError(f"{episode_dir}: raw_action_chunks must be float32.")
    if arrays["executed_action_chunks"].dtype != np.float32:
        raise ValueError(f"{episode_dir}: executed_action_chunks must be float32.")

    expected_indices = np.arange(num_queries, dtype=np.int32)
    if not np.array_equal(arrays["query_indices"], expected_indices):
        raise ValueError(f"{episode_dir}: query_indices are not continuous from zero.")

    return num_queries, int(bool(metadata["final_success"]))


def main() -> None:
    args = parse_args()
    manifest = load_dynamic_gate_manifest(args.manifest)
    rollout_root = osp.abspath(manifest["rollout_root"])

    episode_keys: Set[Tuple[str, int]] = set()
    split_keys = defaultdict(set)
    task_split_counts = defaultdict(Counter)
    sample_ids: Set[str] = set()

    total_queries = 0
    success_episodes = 0

    for record in manifest["episodes"]:
        split = str(record["split"])
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid split={split!r}.")

        key = (str(record["task"]), int(record["episode_id"]))
        if key in episode_keys:
            raise ValueError(f"Duplicate episode in manifest: {key}.")
        episode_keys.add(key)
        split_keys[split].add(key)
        task_split_counts[str(record["task"])][split] += 1

        episode_dir = resolve_episode_dir(rollout_root, record)
        queries, success = check_episode(episode_dir, record)
        total_queries += queries
        success_episodes += success

        for query_index in range(queries):
            sample_id = (
                f"{record['task']}/episode_{int(record['episode_id']):06d}/"
                f"query_{query_index:06d}"
            )
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)

    overlap = split_keys["train"] & split_keys["validation"]
    if overlap:
        raise ValueError(f"Train/validation episode overlap: {sorted(overlap)}")

    for split in ("train", "validation"):
        dataset = RolloutQueryDataset(
            args.manifest,
            split=split,
            require_success=None,
            filter_instruction_changed=False,
        )
        read_count = min(len(dataset), max(0, int(args.max_samples_to_read)))
        for index in range(read_count):
            sample = dataset[index]
            if sample["image"].dtype != np.uint8:
                raise ValueError(f"{sample['sample_id']}: image is not uint8.")
            if sample["proprio"].dtype != np.float32:
                raise ValueError(f"{sample['sample_id']}: proprio is not float32.")
            if sample["raw_action_chunk"].dtype != np.float32:
                raise ValueError(f"{sample['sample_id']}: raw action is not float32.")
            if not np.all(np.isfinite(sample["proprio"])):
                raise ValueError(f"{sample['sample_id']}: proprio contains NaN/Inf.")

    print("========== Dynamic Gate Dataset Check ==========")
    print(f"manifest: {osp.abspath(args.manifest)}")
    print(f"policy_setup: {manifest['policy_setup']}")
    print(f"episodes: {len(manifest['episodes'])}")
    print(f"queries: {total_queries}")
    print(f"success_episodes: {success_episodes}")
    print(f"failure_episodes: {len(manifest['episodes']) - success_episodes}")

    for task in sorted(task_split_counts):
        print(
            f"{task}: train={task_split_counts[task]['train']}, "
            f"validation={task_split_counts[task]['validation']}"
        )

    print("[OK] Dynamic-gate dataset check passed.")


if __name__ == "__main__":
    main()
