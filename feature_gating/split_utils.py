# wx:Dynamic gate v2

"""Episode discovery and deterministic train/validation splitting."""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .rollout_dataset import MANIFEST_VERSION


VALID_POLICY_SETUPS = {"google_robot", "widowx_bridge"}


def infer_policy_setup_from_task(task: str) -> str:
    task = str(task)
    if task.startswith("google_robot"):
        return "google_robot"
    if task.startswith("widowx"):
        return "widowx_bridge"
    raise ValueError(f"Cannot infer PiZero policy setup from task={task!r}.")


def stable_uint64(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _stable_episode_sort_key(record: Mapping[str, Any], seed: int) -> Tuple[int, int]:
    token = (
        f"{seed}|{record['task']}|{record['episode_id']}|"
        f"{record.get('seed')}|{record.get('final_success')}"
    )
    return stable_uint64(token), int(record["episode_id"])


def scan_rollout_episodes(
    rollout_root: str,
    policy_setup: str,
) -> List[Dict[str, Any]]:
    """Discover complete episode directories for one PiZero policy setup."""
    rollout_root = osp.abspath(rollout_root)
    policy_setup = str(policy_setup)

    if policy_setup not in VALID_POLICY_SETUPS:
        raise ValueError(
            f"policy_setup must be one of {sorted(VALID_POLICY_SETUPS)}, "
            f"got {policy_setup!r}."
        )
    if not osp.isdir(rollout_root):
        raise NotADirectoryError(rollout_root)

    records: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(rollout_root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".episode_")]

        if "trajectory.npz" not in filenames or "metadata.json" not in filenames:
            continue

        metadata_path = osp.join(dirpath, "metadata.json")
        trajectory_path = osp.join(dirpath, "trajectory.npz")

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        task = str(metadata.get("task", ""))
        try:
            episode_policy_setup = infer_policy_setup_from_task(task)
        except ValueError:
            continue

        if episode_policy_setup != policy_setup:
            continue

        with np.load(trajectory_path, allow_pickle=False) as data:
            if "query_indices" not in data.files:
                raise KeyError(f"{trajectory_path}: missing query_indices.")
            query_indices = np.asarray(data["query_indices"])
            num_queries_npz = int(query_indices.shape[0])

        num_queries_meta = int(metadata.get("num_policy_queries", -1))
        if num_queries_meta != num_queries_npz:
            raise ValueError(
                f"{dirpath}: metadata num_policy_queries={num_queries_meta} "
                f"!= trajectory query count={num_queries_npz}."
            )

        episode_relpath = osp.relpath(dirpath, rollout_root)
        records.append(
            {
                "task": task,
                "episode_id": int(metadata["episode_id"]),
                "seed": metadata.get("seed"),
                "episode_relpath": episode_relpath,
                "trajectory_relpath": osp.join(episode_relpath, "trajectory.npz"),
                "metadata_relpath": osp.join(episode_relpath, "metadata.json"),
                "final_success": bool(metadata["final_success"]),
                "num_policy_queries": num_queries_npz,
                "num_environment_steps": int(metadata.get("num_environment_steps", -1)),
                "policy_name": str(metadata.get("policy_name", "")),
                "checkpoint": str(metadata.get("checkpoint", "")),
                "policy_setup": episode_policy_setup,
            }
        )

    records.sort(key=lambda item: (item["task"], int(item["episode_id"])))
    return records


def _allocate_validation_counts(
    class_sizes: Mapping[bool, int],
    target_total: int,
) -> Dict[bool, int]:
    """Allocate validation slots across success/failure groups."""
    total = sum(class_sizes.values())
    if total <= 1 or target_total <= 0:
        return {label: 0 for label in class_sizes}

    target_total = min(int(target_total), total - 1)
    allocations = {label: 0 for label in class_sizes}

    eligible = [label for label, size in class_sizes.items() if size >= 2]
    if target_total >= len(eligible):
        for label in eligible:
            allocations[label] = 1

    remaining = target_total - sum(allocations.values())

    while remaining > 0:
        candidates = []
        for label, size in class_sizes.items():
            # Keep at least one episode of a multi-sample class in train.
            capacity = size - 1 if size >= 2 else size
            if allocations[label] >= capacity:
                continue

            desired = target_total * (size / total)
            deficit = desired - allocations[label]
            candidates.append((deficit, size, str(label), label))

        if not candidates:
            break

        candidates.sort(reverse=True)
        chosen = candidates[0][-1]
        allocations[chosen] += 1
        remaining -= 1

    return allocations


def assign_episode_splits(
    records: Sequence[Mapping[str, Any]],
    validation_count_per_task: int = 4,
    split_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Assign deterministic episode-level train/validation splits per task."""
    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw_record in records:
        record = dict(raw_record)
        grouped[str(record["task"])].append(record)

    output: List[Dict[str, Any]] = []

    for task in sorted(grouped):
        task_records = grouped[task]
        if len(task_records) == 1:
            task_records[0]["split"] = "train"
            output.extend(task_records)
            continue

        target_validation = min(
            int(validation_count_per_task),
            len(task_records) - 1,
        )

        by_success: MutableMapping[bool, List[Dict[str, Any]]] = defaultdict(list)
        for record in task_records:
            by_success[bool(record["final_success"])].append(record)

        for label in by_success:
            by_success[label].sort(
                key=lambda record: _stable_episode_sort_key(record, split_seed)
            )

        allocations = _allocate_validation_counts(
            {label: len(items) for label, items in by_success.items()},
            target_validation,
        )

        validation_keys = set()
        for label, items in by_success.items():
            for record in items[: allocations.get(label, 0)]:
                validation_keys.add((record["task"], int(record["episode_id"])))

        # Fill any remaining validation slots deterministically.
        if len(validation_keys) < target_validation:
            remaining_records = sorted(
                task_records,
                key=lambda record: _stable_episode_sort_key(record, split_seed + 1),
            )
            for record in remaining_records:
                key = (record["task"], int(record["episode_id"]))
                if key in validation_keys:
                    continue
                validation_keys.add(key)
                if len(validation_keys) == target_validation:
                    break

        for record in task_records:
            key = (record["task"], int(record["episode_id"]))
            record["split"] = "validation" if key in validation_keys else "train"
            output.append(record)

    output.sort(key=lambda item: (item["task"], int(item["episode_id"])))
    validate_episode_splits(output)
    return output


def validate_episode_splits(records: Sequence[Mapping[str, Any]]) -> None:
    """Ensure each episode appears exactly once and no split overlaps exist."""
    seen: Dict[Tuple[str, int], str] = {}

    for record in records:
        split = str(record.get("split"))
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid split={split!r} in record={record}.")

        key = (str(record["task"]), int(record["episode_id"]))
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"Episode {key} appears more than once: previous={previous}, current={split}."
            )
        seen[key] = split


def build_manifest(
    rollout_root: str,
    policy_setup: str,
    records: Sequence[Mapping[str, Any]],
    split_seed: int,
    validation_count_per_task: int,
) -> Dict[str, Any]:
    validate_episode_splits(records)
    return {
        "manifest_version": MANIFEST_VERSION,
        "rollout_root": osp.abspath(rollout_root),
        "policy_setup": str(policy_setup),
        "split_seed": int(split_seed),
        "validation_count_per_task": int(validation_count_per_task),
        "episodes": [dict(record) for record in records],
    }


def save_manifest(manifest: Mapping[str, Any], output_path: str) -> None:
    output_path = osp.abspath(output_path)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dict(manifest), file, ensure_ascii=False, indent=2)
