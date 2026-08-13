# wx:Dynamic gate v2

"""
Query-level dataset for PiZero dynamic-gate training.

The manifest stores episode-level paths and split metadata only. This dataset
opens each episode's trajectory.npz and metadata.json, aligns fields by query
position, and exposes one sample per policy query.
"""

from __future__ import annotations

import json
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


MANIFEST_VERSION = "dynamic_gate_manifest_v1"

REQUIRED_ARRAY_KEYS = (
    "images",
    "policy_proprios",
    "raw_action_chunks",
    "raw_action_lengths",
    "executed_action_chunks",
    "executed_action_lengths",
    "query_indices",
    "instruction_changed",
)

REQUIRED_METADATA_KEYS = (
    "task",
    "episode_id",
    "instructions",
    "final_success",
    "num_policy_queries",
)


def load_dynamic_gate_manifest(path: str) -> Dict[str, Any]:
    """Load and minimally validate a dynamic-gate manifest."""
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest_version={version!r}; expected {MANIFEST_VERSION!r}."
        )

    if "rollout_root" not in manifest:
        raise KeyError("Manifest is missing rollout_root.")
    if "policy_setup" not in manifest:
        raise KeyError("Manifest is missing policy_setup.")
    if not isinstance(manifest.get("episodes"), list):
        raise TypeError("Manifest episodes must be a list.")

    return manifest


def _resolve_episode_dir(rollout_root: str, episode: Mapping[str, Any]) -> str:
    if "episode_dir" in episode and osp.isabs(str(episode["episode_dir"])):
        return str(episode["episode_dir"])

    relpath = episode.get("episode_relpath", episode.get("episode_dir"))
    if relpath is None:
        raise KeyError("Episode record is missing episode_relpath/episode_dir.")

    return osp.abspath(osp.join(rollout_root, str(relpath)))


def _pad_action_batch(
    samples: Sequence[Mapping[str, Any]],
    value_key: str,
    length_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.as_tensor(
        [int(sample[length_key]) for sample in samples],
        dtype=torch.long,
    )

    arrays = [np.asarray(sample[value_key], dtype=np.float32) for sample in samples]
    if not arrays:
        return (
            torch.zeros((0, 0, 0), dtype=torch.float32),
            lengths,
        )

    action_dims = {int(array.shape[-1]) for array in arrays}
    if len(action_dims) != 1:
        raise ValueError(f"Inconsistent action dimensions for {value_key}: {action_dims}")

    max_length = max(int(array.shape[0]) for array in arrays)
    action_dim = action_dims.pop()
    padded = np.zeros((len(arrays), max_length, action_dim), dtype=np.float32)

    for index, array in enumerate(arrays):
        if array.ndim != 2:
            raise ValueError(
                f"{value_key} must be [T, A], got shape={array.shape} at batch index {index}."
            )
        padded[index, : array.shape[0], :] = array

    return torch.from_numpy(padded), lengths


class RolloutQueryDataset(Dataset):
    """Expand episode records from a manifest into query-level samples."""

    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = "train",
        require_success: Optional[bool] = None,
        filter_instruction_changed: bool = False,
        max_cached_episodes: int = 2,
    ) -> None:
        super().__init__()

        self.manifest_path = osp.abspath(manifest_path)
        self.manifest = load_dynamic_gate_manifest(self.manifest_path)
        self.rollout_root = osp.abspath(self.manifest["rollout_root"])
        self.policy_setup = str(self.manifest["policy_setup"])
        self.split = None if split is None else str(split)
        self.require_success = require_success
        self.filter_instruction_changed = bool(filter_instruction_changed)
        self.max_cached_episodes = max(0, int(max_cached_episodes))

        self.episodes: List[Dict[str, Any]] = []
        for raw_episode in self.manifest["episodes"]:
            episode = dict(raw_episode)

            if self.split is not None and episode.get("split") != self.split:
                continue
            if (
                self.require_success is not None
                and bool(episode.get("final_success")) != bool(self.require_success)
            ):
                continue

            episode["episode_dir"] = _resolve_episode_dir(self.rollout_root, episode)
            self.episodes.append(episode)

        self._cache: "OrderedDict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]]" = OrderedDict()
        self._query_index: List[Tuple[int, int]] = []
        self._build_query_index()

        if not self._query_index:
            raise RuntimeError(
                "No query samples matched the requested manifest filters: "
                f"split={self.split!r}, require_success={self.require_success!r}."
            )

    def _load_episode(
        self,
        episode_dir: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if episode_dir in self._cache:
            arrays, metadata = self._cache.pop(episode_dir)
            self._cache[episode_dir] = (arrays, metadata)
            return arrays, metadata

        trajectory_path = osp.join(episode_dir, "trajectory.npz")
        metadata_path = osp.join(episode_dir, "metadata.json")

        if not osp.isfile(trajectory_path):
            raise FileNotFoundError(trajectory_path)
        if not osp.isfile(metadata_path):
            raise FileNotFoundError(metadata_path)

        with np.load(trajectory_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        for key in REQUIRED_ARRAY_KEYS:
            if key not in arrays:
                raise KeyError(f"{episode_dir}: trajectory.npz is missing {key}.")
        for key in REQUIRED_METADATA_KEYS:
            if key not in metadata:
                raise KeyError(f"{episode_dir}: metadata.json is missing {key}.")

        num_queries = int(metadata["num_policy_queries"])
        if len(metadata["instructions"]) != num_queries:
            raise ValueError(
                f"{episode_dir}: len(instructions)={len(metadata['instructions'])} "
                f"!= num_policy_queries={num_queries}."
            )

        for key in REQUIRED_ARRAY_KEYS:
            if arrays[key].shape[0] != num_queries:
                raise ValueError(
                    f"{episode_dir}: {key}.shape[0]={arrays[key].shape[0]} "
                    f"!= num_policy_queries={num_queries}."
                )

        if self.max_cached_episodes > 0:
            self._cache[episode_dir] = (arrays, metadata)
            while len(self._cache) > self.max_cached_episodes:
                self._cache.popitem(last=False)

        return arrays, metadata

    def _build_query_index(self) -> None:
        for episode_position, episode in enumerate(self.episodes):
            num_queries = int(episode["num_policy_queries"])

            if not self.filter_instruction_changed:
                self._query_index.extend(
                    (episode_position, query_position)
                    for query_position in range(num_queries)
                )
                continue

            arrays, _ = self._load_episode(episode["episode_dir"])
            changed = arrays["instruction_changed"].astype(bool)
            for query_position in range(num_queries):
                if not bool(changed[query_position]):
                    self._query_index.append((episode_position, query_position))

    def __len__(self) -> int:
        return len(self._query_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode_position, query_position = self._query_index[index]
        episode = self.episodes[episode_position]
        arrays, metadata = self._load_episode(episode["episode_dir"])

        query_index = int(arrays["query_indices"][query_position])
        episode_id = int(metadata["episode_id"])
        task = str(metadata["task"])
        sample_id = (
            f"{task}/episode_{episode_id:06d}/query_{query_index:06d}"
        )

        raw_length = int(arrays["raw_action_lengths"][query_position])
        executed_length = int(arrays["executed_action_lengths"][query_position])

        raw_action_chunk = np.asarray(
            arrays["raw_action_chunks"][query_position, :raw_length],
            dtype=np.float32,
        ).copy()
        executed_action_chunk = np.asarray(
            arrays["executed_action_chunks"][query_position, :executed_length],
            dtype=np.float32,
        ).copy()

        return {
            "sample_id": sample_id,
            "task": task,
            "episode_id": episode_id,
            "query_index": query_index,
            "episode_seed": metadata.get("seed"),
            "episode_dir": episode["episode_dir"],
            "image": np.asarray(arrays["images"][query_position]).copy(),
            "instruction": str(metadata["instructions"][query_position]),
            "proprio": np.asarray(
                arrays["policy_proprios"][query_position],
                dtype=np.float32,
            ).copy(),
            "raw_action_chunk": raw_action_chunk,
            "raw_action_length": raw_length,
            "executed_action_chunk": executed_action_chunk,
            "executed_action_length": executed_length,
            "final_success": bool(metadata["final_success"]),
            "instruction_changed": bool(
                arrays["instruction_changed"][query_position]
            ),
            "policy_name": str(metadata.get("policy_name", "")),
            "base_checkpoint": str(metadata.get("checkpoint", "")),
        }

    def summary(self) -> Dict[str, Any]:
        task_counts: Dict[str, int] = {}
        success_episode_count = 0

        for episode in self.episodes:
            task = str(episode["task"])
            task_counts[task] = task_counts.get(task, 0) + 1
            success_episode_count += int(bool(episode["final_success"]))

        return {
            "manifest_path": self.manifest_path,
            "policy_setup": self.policy_setup,
            "split": self.split,
            "num_episodes": len(self.episodes),
            "num_queries": len(self),
            "num_success_episodes": success_episode_count,
            "num_failure_episodes": len(self.episodes) - success_episode_count,
            "task_episode_counts": task_counts,
        }


def collate_rollout_queries(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Collate query samples while keeping raw images as a Python list."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    proprios = np.stack(
        [np.asarray(sample["proprio"], dtype=np.float32) for sample in samples],
        axis=0,
    )

    raw_actions, raw_lengths = _pad_action_batch(
        samples,
        "raw_action_chunk",
        "raw_action_length",
    )
    executed_actions, executed_lengths = _pad_action_batch(
        samples,
        "executed_action_chunk",
        "executed_action_length",
    )

    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "task": [str(sample["task"]) for sample in samples],
        "episode_id": torch.as_tensor(
            [int(sample["episode_id"]) for sample in samples],
            dtype=torch.long,
        ),
        "query_index": torch.as_tensor(
            [int(sample["query_index"]) for sample in samples],
            dtype=torch.long,
        ),
        "image": [np.asarray(sample["image"]).copy() for sample in samples],
        "instruction": [str(sample["instruction"]) for sample in samples],
        "proprio": torch.from_numpy(proprios),
        "raw_action_chunk": raw_actions,
        "raw_action_length": raw_lengths,
        "executed_action_chunk": executed_actions,
        "executed_action_length": executed_lengths,
        "final_success": torch.as_tensor(
            [bool(sample["final_success"]) for sample in samples],
            dtype=torch.bool,
        ),
        "instruction_changed": torch.as_tensor(
            [bool(sample["instruction_changed"]) for sample in samples],
            dtype=torch.bool,
        ),
        "episode_dir": [str(sample["episode_dir"]) for sample in samples],
        "base_checkpoint": [str(sample["base_checkpoint"]) for sample in samples],
    }
