# wx:Dynamic gate v3

"""Persistent fixed-noise PiZero teacher-action cache."""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import shutil
import uuid
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


TEACHER_CACHE_VERSION = "fixed_noise_teacher_v1"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_path(path: str) -> str:
    return osp.abspath(osp.expanduser(str(path)))


def save_teacher_cache(
    cache_dir: str,
    *,
    sample_ids: Sequence[str],
    noise_seeds: Sequence[int],
    teacher_actions: np.ndarray,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> str:
    cache_dir = normalize_path(cache_dir)
    if osp.exists(cache_dir):
        if not overwrite:
            raise FileExistsError(
                f"Teacher cache already exists: {cache_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(cache_dir)

    sample_ids_array = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    noise_seeds_array = np.asarray(noise_seeds, dtype=np.uint64)
    actions_array = np.asarray(teacher_actions, dtype=np.float32)

    if actions_array.ndim != 3:
        raise ValueError(
            f"teacher_actions must be [N, H, A], got {actions_array.shape}."
        )
    if len(sample_ids_array) != len(noise_seeds_array):
        raise ValueError("sample_ids/noise_seeds length mismatch.")
    if len(sample_ids_array) != actions_array.shape[0]:
        raise ValueError("sample_ids/teacher_actions length mismatch.")
    if len(set(sample_ids_array.tolist())) != len(sample_ids_array):
        raise ValueError("Teacher cache sample IDs must be unique.")
    if not np.all(np.isfinite(actions_array)):
        raise ValueError("Teacher actions contain NaN or Inf.")

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "cache_version": TEACHER_CACHE_VERSION,
            "num_samples": int(actions_array.shape[0]),
            "horizon_steps": int(actions_array.shape[1]),
            "action_dim": int(actions_array.shape[2]),
        }
    )

    parent = osp.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)
    temporary_dir = f"{cache_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    os.makedirs(temporary_dir, exist_ok=False)

    try:
        np.save(osp.join(temporary_dir, "sample_ids.npy"), sample_ids_array)
        np.save(osp.join(temporary_dir, "noise_seeds.npy"), noise_seeds_array)
        np.save(osp.join(temporary_dir, "teacher_actions.npy"), actions_array)
        with open(
            osp.join(temporary_dir, "metadata.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(final_metadata, file, ensure_ascii=False, indent=2)
        os.replace(temporary_dir, cache_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return cache_dir


class TeacherActionCache:
    """Read teacher actions by stable query sample ID."""

    def __init__(self, cache_dir: str, *, mmap: bool = True) -> None:
        self.cache_dir = normalize_path(cache_dir)
        metadata_path = osp.join(self.cache_dir, "metadata.json")
        sample_ids_path = osp.join(self.cache_dir, "sample_ids.npy")
        noise_seeds_path = osp.join(self.cache_dir, "noise_seeds.npy")
        actions_path = osp.join(self.cache_dir, "teacher_actions.npy")

        for path in (
            metadata_path,
            sample_ids_path,
            noise_seeds_path,
            actions_path,
        ):
            if not osp.isfile(path):
                raise FileNotFoundError(path)

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata: Dict[str, Any] = json.load(file)

        version = self.metadata.get("cache_version")
        if version != TEACHER_CACHE_VERSION:
            raise ValueError(
                f"Unsupported teacher cache version={version!r}; "
                f"expected {TEACHER_CACHE_VERSION!r}."
            )

        mmap_mode = "r" if mmap else None
        self.sample_ids = np.load(sample_ids_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.noise_seeds = np.load(noise_seeds_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.teacher_actions = np.load(actions_path, allow_pickle=False, mmap_mode=mmap_mode)

        if self.teacher_actions.ndim != 3:
            raise ValueError(
                f"teacher_actions must be [N,H,A], got {self.teacher_actions.shape}."
            )
        if len(self.sample_ids) != len(self.noise_seeds):
            raise ValueError("Teacher cache sample IDs/noise seeds mismatch.")
        if len(self.sample_ids) != self.teacher_actions.shape[0]:
            raise ValueError("Teacher cache sample IDs/actions mismatch.")

        self._index = {
            str(sample_id): index
            for index, sample_id in enumerate(self.sample_ids.tolist())
        }
        if len(self._index) != len(self.sample_ids):
            raise ValueError("Teacher cache contains duplicate sample IDs.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def validate(
        self,
        *,
        manifest_path: Optional[str] = None,
        policy_setup: Optional[str] = None,
        split: Optional[str] = None,
        base_checkpoint: Optional[str] = None,
        flow_sampling: Optional[str] = None,
        base_noise_seed: Optional[int] = None,
    ) -> None:
        if manifest_path is not None:
            expected_hash = sha256_file(normalize_path(manifest_path))
            actual_hash = str(self.metadata.get("manifest_sha256", ""))
            if actual_hash != expected_hash:
                raise ValueError(
                    "Teacher cache manifest fingerprint mismatch: "
                    f"cache={actual_hash}, expected={expected_hash}."
                )

        expected_pairs = {
            "policy_setup": policy_setup,
            "split": split,
            "flow_sampling": flow_sampling,
            "base_noise_seed": base_noise_seed,
        }
        for key, expected in expected_pairs.items():
            if expected is None:
                continue
            actual = self.metadata.get(key)
            if str(actual) != str(expected):
                raise ValueError(
                    f"Teacher cache {key} mismatch: "
                    f"cache={actual!r}, expected={expected!r}."
                )

        if base_checkpoint is not None:
            expected = normalize_path(base_checkpoint)
            actual = normalize_path(str(self.metadata.get("base_checkpoint", "")))
            if actual != expected:
                raise ValueError(
                    "Teacher cache base checkpoint mismatch: "
                    f"cache={actual!r}, expected={expected!r}."
                )

    def require_sample_ids(self, sample_ids: Sequence[str]) -> None:
        missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in self._index]
        if missing:
            raise KeyError(
                f"Teacher cache is missing {len(missing)} sample IDs; "
                f"first missing IDs: {missing[:5]}"
            )

    def get_numpy(self, sample_ids: Sequence[str]) -> np.ndarray:
        self.require_sample_ids(sample_ids)
        indices = [self._index[str(sample_id)] for sample_id in sample_ids]
        return np.asarray(self.teacher_actions[indices], dtype=np.float32).copy()

    def get_tensor(
        self,
        sample_ids: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        array = self.get_numpy(sample_ids)
        return torch.from_numpy(array).to(device=device, dtype=dtype)

    def get_noise_seeds(self, sample_ids: Sequence[str]) -> np.ndarray:
        self.require_sample_ids(sample_ids)
        indices = [self._index[str(sample_id)] for sample_id in sample_ids]
        return np.asarray(self.noise_seeds[indices], dtype=np.uint64).copy()
