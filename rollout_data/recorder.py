# wx:Rollout collect

# rollout_data/recorder.py
# Query-level unlabeled rollout recorder for PCD / SimplerEnv.
#
# This recorder stores one sample per policy query, not one sample per env.step.
# It does not store object masks, contrast images, detector outputs, VLM scores,
# gradients, or model features.

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
import uuid
from typing import Any, Dict, List, Optional

import numpy as np


COLLECTION_VERSION = "v0_query_level_rollout"


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch / numpy / list / scalar into a numpy array without object dtype."""
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass

    return np.asarray(x)


def _ensure_uint8_image(image: Any, name: str) -> np.ndarray:
    arr = np.asarray(image).copy()

    if arr.dtype != np.uint8:
        raise ValueError(
            f"{name} must be uint8, got dtype={arr.dtype}. "
            "Collector expects the raw RGB image returned by the environment."
        )

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape [H, W, 3], got shape={arr.shape}."
        )

    return arr


def _ensure_float32_array(x: Any, name: str) -> np.ndarray:
    arr = _to_numpy(x).astype(np.float32, copy=True)

    if arr.dtype == np.dtype("O"):
        raise ValueError(f"{name} has object dtype, which is not allowed.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return arr


def _as_2d_action_chunk(x: Any, name: str) -> np.ndarray:
    """
    Convert raw action chunk into [T, A].

    Common pi0 raw_action shape is [1, horizon, action_dim].
    We remove leading batch dim if it is 1.
    """
    arr = _ensure_float32_array(x, name)

    if arr.ndim == 0:
        raise ValueError(f"{name} should not be scalar.")

    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        pass
    else:
        # Keep the first remaining dimension as temporal dimension.
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32, copy=True)


def make_executed_action_vector(action: Dict[str, Any]) -> np.ndarray:
    """
    Convert the exact action dict passed by policy into the exact vector passed to env.step.

    The order matches parallel_inference.py:
        world_vector + rot_axangle + gripper
    """
    required = ["world_vector", "rot_axangle", "gripper"]
    for key in required:
        if key not in action:
            raise KeyError(f"Action dict is missing required key: {key}")

    parts = [
        np.asarray(action["world_vector"], dtype=np.float32).reshape(-1),
        np.asarray(action["rot_axangle"], dtype=np.float32).reshape(-1),
        np.asarray(action["gripper"], dtype=np.float32).reshape(-1),
    ]
    vector = np.concatenate(parts, axis=0).astype(np.float32, copy=True)

    if vector.dtype == np.dtype("O"):
        raise ValueError("Executed action vector has object dtype.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Executed action vector contains NaN or Inf.")

    return vector


def _pad_action_chunks(chunks: List[np.ndarray], name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad variable-length chunks to [Q, max_len, dim].
    Padding values are zeros. Also returns lengths [Q].
    """
    q = len(chunks)
    lengths = np.asarray([chunk.shape[0] for chunk in chunks], dtype=np.int32)

    if q == 0:
        return (
            np.zeros((0, 0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    dims = [chunk.shape[1] for chunk in chunks]
    if len(set(dims)) != 1:
        raise ValueError(f"{name} action dims are inconsistent: {dims}")

    max_len = int(max(lengths))
    dim = int(dims[0])

    padded = np.zeros((q, max_len, dim), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        length = int(lengths[i])
        if length > 0:
            padded[i, :length, :] = chunk

    return padded, lengths


class EpisodeRolloutRecorder:
    """
    Save one episode as:
        <rollout_root>/<task>/episode_000000/trajectory.npz
        <rollout_root>/<task>/episode_000000/metadata.json

    The recorder writes to a temporary directory first and then atomically renames it
    to the final episode directory.
    """

    def __init__(
        self,
        rollout_root: str,
        task: str,
        episode_id: int,
        seed: Optional[int],
        policy_name: str,
        checkpoint: str,
        compress: bool = True,
        collection_version: str = COLLECTION_VERSION,
    ):
        self.rollout_root = str(rollout_root)
        self.task = str(task)
        self.episode_id = int(episode_id)
        self.seed = None if seed is None else int(seed)
        self.policy_name = str(policy_name)
        self.checkpoint = str(checkpoint)
        self.compress = bool(compress)
        self.collection_version = str(collection_version)

        self.images: List[np.ndarray] = []
        self.next_images: List[np.ndarray] = []

        self.policy_proprios: List[np.ndarray] = []
        self.next_policy_proprios: List[np.ndarray] = []

        self.raw_action_chunks: List[np.ndarray] = []
        self.executed_action_chunks: List[np.ndarray] = []

        self.query_indices: List[int] = []
        self.env_step_starts: List[int] = []
        self.env_step_ends: List[int] = []

        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
        self.success_at_end: List[bool] = []
        self.instruction_changed: List[bool] = []

        self.instructions: List[str] = []
        self.next_instructions: List[str] = []

        self._finalized = False
        self._final_metadata: Optional[Dict[str, Any]] = None

    @property
    def num_queries(self) -> int:
        return len(self.images)

    def add_query(
        self,
        *,
        image_t: Any,
        instruction_t: str,
        policy_proprio_t: Any,
        raw_action_chunk: Any,
        executed_action_chunk: List[np.ndarray],
        next_image: Any,
        next_policy_proprio: Any,
        next_instruction: str,
        episode_id: int,
        query_index: int,
        env_step_start: int,
        env_step_end: int,
        terminated: bool,
        truncated: bool,
        success_at_end: bool,
        instruction_changed: Optional[bool] = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add query after recorder is finalized.")

        if int(episode_id) != self.episode_id:
            raise ValueError(
                f"episode_id mismatch: recorder={self.episode_id}, add_query={episode_id}"
            )

        image_arr = _ensure_uint8_image(image_t, "image_t")
        next_image_arr = _ensure_uint8_image(next_image, "next_image")

        proprio_arr = _ensure_float32_array(policy_proprio_t, "policy_proprio_t").reshape(-1)
        next_proprio_arr = _ensure_float32_array(next_policy_proprio, "next_policy_proprio").reshape(-1)

        raw_chunk_arr = _as_2d_action_chunk(raw_action_chunk, "raw_action_chunk")

        if len(executed_action_chunk) == 0:
            exec_chunk_arr = np.zeros((0, 0), dtype=np.float32)
        else:
            exec_vectors = [
                np.asarray(v, dtype=np.float32).reshape(1, -1)
                for v in executed_action_chunk
            ]
            exec_dims = [v.shape[1] for v in exec_vectors]
            if len(set(exec_dims)) != 1:
                raise ValueError(
                    f"executed_action_chunk dims are inconsistent: {exec_dims}"
                )
            exec_chunk_arr = np.concatenate(exec_vectors, axis=0).astype(np.float32, copy=True)

        inst = str(instruction_t)
        next_inst = str(next_instruction)
        changed = (inst != next_inst) if instruction_changed is None else bool(instruction_changed)

        self.images.append(image_arr)
        self.next_images.append(next_image_arr)

        self.policy_proprios.append(proprio_arr)
        self.next_policy_proprios.append(next_proprio_arr)

        self.raw_action_chunks.append(raw_chunk_arr)
        self.executed_action_chunks.append(exec_chunk_arr)

        self.query_indices.append(int(query_index))
        self.env_step_starts.append(int(env_step_start))
        self.env_step_ends.append(int(env_step_end))

        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.success_at_end.append(bool(success_at_end))
        self.instruction_changed.append(changed)

        self.instructions.append(inst)
        self.next_instructions.append(next_inst)

    def finalize(
        self,
        *,
        final_success: bool,
        terminated: bool,
        truncated: bool,
        num_environment_steps: int,
    ) -> None:
        self._final_metadata = {
            "task": self.task,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "instructions": self.instructions,
            "next_instructions": self.next_instructions,
            "final_success": bool(final_success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "num_policy_queries": int(self.num_queries),
            "num_environment_steps": int(num_environment_steps),
            "policy_name": self.policy_name,
            "checkpoint": self.checkpoint,
            "collection_version": self.collection_version,
        }
        self._finalized = True

    def _build_arrays(self) -> Dict[str, np.ndarray]:
        q = self.num_queries

        if q == 0:
            images = np.zeros((0, 0, 0, 3), dtype=np.uint8)
            next_images = np.zeros((0, 0, 0, 3), dtype=np.uint8)
            policy_proprios = np.zeros((0, 0), dtype=np.float32)
            next_policy_proprios = np.zeros((0, 0), dtype=np.float32)
        else:
            images = np.stack(self.images, axis=0).astype(np.uint8, copy=False)
            next_images = np.stack(self.next_images, axis=0).astype(np.uint8, copy=False)

            proprio_dims = [p.shape[0] for p in self.policy_proprios]
            next_proprio_dims = [p.shape[0] for p in self.next_policy_proprios]
            if len(set(proprio_dims)) != 1:
                raise ValueError(f"policy_proprio dims are inconsistent: {proprio_dims}")
            if len(set(next_proprio_dims)) != 1:
                raise ValueError(f"next_policy_proprio dims are inconsistent: {next_proprio_dims}")

            policy_proprios = np.stack(self.policy_proprios, axis=0).astype(np.float32, copy=False)
            next_policy_proprios = np.stack(self.next_policy_proprios, axis=0).astype(np.float32, copy=False)

        raw_padded, raw_lengths = _pad_action_chunks(self.raw_action_chunks, "raw_action_chunks")
        exec_padded, exec_lengths = _pad_action_chunks(self.executed_action_chunks, "executed_action_chunks")

        arrays = {
            "images": images,
            "next_images": next_images,
            "policy_proprios": policy_proprios,
            "next_policy_proprios": next_policy_proprios,
            "raw_action_chunks": raw_padded,
            "raw_action_lengths": raw_lengths,
            "executed_action_chunks": exec_padded,
            "executed_action_lengths": exec_lengths,
            "query_indices": np.asarray(self.query_indices, dtype=np.int32),
            "env_step_starts": np.asarray(self.env_step_starts, dtype=np.int32),
            "env_step_ends": np.asarray(self.env_step_ends, dtype=np.int32),
            "terminated": np.asarray(self.terminated, dtype=np.bool_),
            "truncated": np.asarray(self.truncated, dtype=np.bool_),
            "success_at_end": np.asarray(self.success_at_end, dtype=np.bool_),
            "instruction_changed": np.asarray(self.instruction_changed, dtype=np.bool_),
        }

        for key, arr in arrays.items():
            if arr.dtype == np.dtype("O"):
                raise ValueError(f"{key} has object dtype, which is not allowed.")

        return arrays

    def save(self) -> str:
        if not self._finalized or self._final_metadata is None:
            raise RuntimeError("Call finalize(...) before save().")

        task_dir = osp.join(self.rollout_root, self.task)
        final_dir = osp.join(task_dir, f"episode_{self.episode_id:06d}")

        if osp.exists(final_dir):
            raise FileExistsError(
                f"Rollout episode directory already exists: {final_dir}. "
                "Use a new --rollout-root or remove the existing directory."
            )

        os.makedirs(task_dir, exist_ok=True)

        tmp_dir = osp.join(
            task_dir,
            f".episode_{self.episode_id:06d}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
        )

        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir, exist_ok=False)

        try:
            arrays = self._build_arrays()

            npz_path = osp.join(tmp_dir, "trajectory.npz")
            if self.compress:
                np.savez_compressed(npz_path, **arrays)
            else:
                np.savez(npz_path, **arrays)

            meta_path = osp.join(tmp_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._final_metadata, f, ensure_ascii=False, indent=2)

            # Atomic directory publish on the same filesystem.
            os.rename(tmp_dir, final_dir)

        except Exception:
            if osp.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return final_dir