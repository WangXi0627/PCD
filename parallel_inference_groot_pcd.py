# wx: GR00T-N1.6 + PCD-style grounded_sam_tracking runner
# File: PCD/parallel_inference_groot_pcd.py

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

# wx:GR00T random feature mask
import hashlib
import re
# wx:GR00T random feature mask

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
    _RobustAsyncVectorEnv,
)
from gr00t.policy.server_client import PolicyClient

# ---------------------------------------------------------------------
# Original PCD modules
# ---------------------------------------------------------------------
# This runner is intended to be executed from PCD_ROOT, with PCD_ROOT in PYTHONPATH.
# It reuses:
#   1. contrast_utils.contrast_image_generator.ContrastImageGenerator
#   2. contrast_policies.kde_contrast_decoding.ContrastDecoding
# rather than reimplementing random masks or KDE manually.
try:
    from contrast_utils.contrast_image_generator import ContrastImageGenerator
    from contrast_policies.kde_contrast_decoding import ContrastDecoding
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import original PCD modules. Please run this script inside PCD_ROOT "
        "and make sure PCD_ROOT is in PYTHONPATH. Original error: " + repr(e)
    )

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------
# Basic utils
# ---------------------------------------------------------------------


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, str):
        return v.lower() in ["1", "true", "yes", "y", "on"]
    return bool(v)


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# wx:GR00T random feature mask
def safe_dir_name(name: str, max_len: int = 160) -> str:
    """
    Convert a user-provided experiment name into a safe directory name.
    If the name is too long, keep a short prefix and append a hash.
    """
    name = str(name).strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.=+\-]", "_", name)

    if len(name) <= max_len:
        return name

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[:max_len]}--{digest}"

def safe_relative_path(name: str, max_part_len: int = 120) -> Path:
    """
    Convert a user-provided run_name into a safe relative path.

    Example:
        "a/b/c" -> Path("a") / "b" / "c"

    Each path component is sanitized independently.
    Absolute paths and empty components are ignored.
    """
    parts = []
    for part in str(name).replace("\\", "/").split("/"):
        part = part.strip()
        if part in ("", ".", ".."):
            continue
        parts.append(safe_dir_name(part, max_len=max_part_len))

    if len(parts) == 0:
        return Path("default_run")

    return Path(*parts)

def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
# wx:GR00T random feature mask

def pcd_task_to_groot_env_name(task: str) -> str:
    """
    Allow both:
        google_robot_pick_coke_can
        simpler_env_google/google_robot_pick_coke_can

    PCD usually uses task names like google_robot_pick_coke_can.
    GR00T official runner uses gym env names like:
        simpler_env_google/google_robot_pick_coke_can
        simpler_env_widowx/widowx_spoon_on_towel
    """
    if "/" in task:
        return task

    if task.startswith("google_robot"):
        return f"simpler_env_google/{task}"

    if task.startswith("widowx"):
        return f"simpler_env_widowx/{task}"

    raise ValueError(
        f"Cannot infer GR00T env_name from task='{task}'. "
        "Please pass full env name, e.g. simpler_env_google/google_robot_pick_coke_can"
    )


def safe_json_dump(obj: Any, path: str | Path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def as_bool_success(x: Any) -> bool:
    """GR00T/SimplerEnv infos may return bool/list/np.ndarray/int."""
    if isinstance(x, list):
        return bool(np.any(x))
    if isinstance(x, np.ndarray):
        return bool(np.any(x))
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if x is None:
        return False
    return bool(x)


def get_first_video_key(observations: Dict[str, Any], video_key: str = "") -> str:
    if video_key:
        if video_key in observations:
            return video_key
        if not video_key.startswith("video."):
            flat = f"video.{video_key}"
            if flat in observations:
                return flat
        raise KeyError(
            f"Cannot find requested pcd_video_key='{video_key}'. "
            f"Available keys: {list(observations.keys())}"
        )

    video_keys = [k for k in observations.keys() if k.startswith("video.")]
    if not video_keys:
        raise KeyError(f"No video.* key found in observations: {list(observations.keys())}")
    # Usually video.image_0.
    return video_keys[0]


def get_language_batch(observations: Dict[str, Any], batch_size: int) -> List[str]:
    candidate_keys = [
        "annotation.human.action.task_description",
        "task",
    ]

    for key in candidate_keys:
        if key not in observations:
            continue

        value = observations[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, str):
            return [value for _ in range(batch_size)]
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if len(value) == batch_size:
                return [str(v) for v in value]
            if len(value) == 1:
                return [str(value[0]) for _ in range(batch_size)]

    raise KeyError(
        "Cannot find language instruction in observations. Expected key "
        "'annotation.human.action.task_description' or 'task'. "
        f"Available keys: {list(observations.keys())}"
    )


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.size > 0 and np.nanmax(arr) <= 1.5:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def restore_image_dtype(image_uint8: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference)
    if ref.dtype == np.uint8:
        return image_uint8.astype(np.uint8)

    if np.issubdtype(ref.dtype, np.floating):
        ref_max = float(np.nanmax(ref)) if ref.size else 255.0
        if ref_max <= 1.5:
            return (image_uint8.astype(np.float32) / 255.0).astype(ref.dtype)
        return image_uint8.astype(ref.dtype)

    return image_uint8.astype(ref.dtype)


def maybe_resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    if cv2 is None:
        raise RuntimeError(
            f"Contrast image shape {image.shape} != reference shape {reference.shape}, "
            "and cv2 is not available for resizing."
        )
    h, w = reference.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------
# Feature-mask options hook
# ---------------------------------------------------------------------


def build_policy_options(args: argparse.Namespace, step_idx: int) -> Optional[Dict[str, Any]]:
    """
    Normally returns None for original-PCD-style image contrast.
    Reserved for later feature-level GR00T server experiments.
    """
    if not str2bool(args.feature_mask_enable):
        return None

    return {
        "feature_mask": {
            "enable": True,
            "target": args.feature_mask_target,
            "mode": args.feature_mask_mode,
            "keep_ratio": float(args.feature_mask_keep_ratio),
            "seed": int(args.feature_mask_seed),
            "rescale": str2bool(args.feature_mask_rescale),
            # wx:GR00T random feature mask
            "verbose": str2bool(args.feature_mask_verbose),
            "print_candidates": str2bool(args.feature_mask_print_candidates),
            # wx:GR00T random feature mask
        },
        "sample_seed": int(args.feature_mask_seed) + int(step_idx),
    }


def policy_get_action(
    policy: PolicyClient,
    observations: Dict[str, Any],
    options: Optional[dict],
):
    """PolicyClient supports get_action(observation, options)."""
    if options is None:
        return policy.get_action(observations)

    try:
        return policy.get_action(observations, options=options)
    except TypeError:
        return policy.get_action(observations)


# ---------------------------------------------------------------------
# GR00T official observation adapter for original PCD ContrastImageGenerator
# ---------------------------------------------------------------------


class GrootObsContrastImageGenerator(ContrastImageGenerator):
    """
    Thin adapter around PCD's original ContrastImageGenerator.

    Original PCD expects raw ManiSkill/SimplerEnv obs:
        obs["image"][camera_name]["rgb"]

    GR00T official runner exposes vectorized policy observation:
        observations["video.image_0"] with shape (B,T,H,W,C)

    This subclass only overrides _get_camera_images so that the original
    ContrastImageGenerator.generate(...) can still run on a pseudo-observation.
    All object parsing, grounded_sam_tracking predictor, mask composition,
    robot exclusion, and inpainting are handled by the original PCD class.
    """

    def __init__(
        self,
        by: str = "grounded_sam_tracking",
        inpaint_mode: str = "lama",
        get_all_parts: bool = False,
        camera_name: str = "groot_camera",
    ):
        super().__init__(
            env=None,
            camera_name=camera_name,
            by=by,
            inpaint_mode=inpaint_mode,
            get_all_parts=get_all_parts,
        )

        if by != "grounded_sam_tracking":
            raise ValueError(
                "This GR00T+PCD runner is designed to reproduce PCD's "
                f"grounded_sam_tracking path, but got by={by}."
            )

    def _get_camera_images(self, obs):
        return obs["image"][self.camera_name]


def make_pseudo_pcd_obs(image: np.ndarray, camera_name: str = "groot_camera") -> Dict[str, Any]:
    return {
        "image": {
            camera_name: {
                "rgb": image,
            }
        }
    }


class VectorizedPCDContrastGenerator:
    """
    One original PCD ContrastImageGenerator per vector-env slot.

    This is closer to PCD's original serial runner than sharing one generator
    across all envs, because grounded_sam_tracking can carry state. In the
    official GR00T runner we may use vectorized envs, so every env index owns
    its own PCD contrast generator and is reset when that env finishes.
    """

    def __init__(
        self,
        n_envs: int,
        by: str = "grounded_sam_tracking",
        inpaint_mode: str = "lama",
        get_all_parts: bool = False,
        video_key: str = "",
        fallback_on_error: bool = False,
        debug_save_dir: str | Path | None = None,
        debug_image_interval: int = 0,
    ):
        self.n_envs = int(n_envs)
        self.by = by
        self.inpaint_mode = inpaint_mode
        self.get_all_parts = bool(get_all_parts)
        self.video_key = video_key
        self.camera_name = "groot_camera"
        self.fallback_on_error = bool(fallback_on_error)
        self.debug_save_dir = Path(debug_save_dir) if debug_save_dir else None
        self.debug_image_interval = int(debug_image_interval)
        self.num_failures = 0

        if self.debug_save_dir is not None:
            mkdir(self.debug_save_dir)

        self.generators = [
            GrootObsContrastImageGenerator(
                by=self.by,
                inpaint_mode=self.inpaint_mode,
                get_all_parts=self.get_all_parts,
                camera_name=self.camera_name,
            )
            for _ in range(self.n_envs)
        ]

    def reset(self):
        for env_idx in range(self.n_envs):
            self.reset_env(env_idx)

    def reset_env(self, env_idx: int):
        generator = self.generators[int(env_idx)]
        if hasattr(generator, "reset"):
            generator.reset()

    def _save_debug_pair(
        self,
        clean: np.ndarray,
        contrast: np.ndarray,
        env_idx: int,
        step_idx: int,
    ):
        if self.debug_save_dir is None:
            return
        if self.debug_image_interval <= 0:
            return
        if step_idx % self.debug_image_interval != 0:
            return
        if cv2 is None:
            return

        pair = np.concatenate([clean, contrast], axis=1)
        # cv2 expects BGR.
        pair_bgr = pair[:, :, ::-1]
        out_path = self.debug_save_dir / f"env{env_idx:02d}_step{step_idx:06d}.jpg"
        cv2.imwrite(str(out_path), pair_bgr)

    def generate(
        self,
        observations: Dict[str, Any],
        step_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate contrast observations using original PCD ContrastImageGenerator.

        Input:
            observations[video_key]: shape (B,T,H,W,C)

        Output:
            same dict, with video_key replaced by contrast video.
        """
        out = dict(observations)

        video_key = get_first_video_key(observations, self.video_key)
        video = np.asarray(observations[video_key])

        if video.ndim != 5:
            raise ValueError(f"Expected {video_key} shape (B,T,H,W,C), got {video.shape}")

        batch_size, horizon = video.shape[:2]
        if batch_size != self.n_envs:
            raise ValueError(
                f"PCD contrast generator was built with n_envs={self.n_envs}, "
                f"but observation batch size is {batch_size}."
            )

        instructions = get_language_batch(observations, batch_size)
        contrast_video = video.copy()

        for env_idx in range(batch_size):
            # GR00T official wrapper normally has T=1. If T>1, use the latest
            # policy frame for object removal and fill the history with the same
            # contrast image, which matches the model's current observation.
            ref_frame = video[env_idx, -1]
            clean_uint8 = to_uint8_image(ref_frame)
            pseudo_obs = make_pseudo_pcd_obs(clean_uint8, camera_name=self.camera_name)
            instruction = instructions[env_idx]

            try:
                contrast_uint8 = self.generators[env_idx].generate(pseudo_obs, instruction)
                contrast_uint8 = np.asarray(contrast_uint8)
                contrast_uint8 = maybe_resize_like(contrast_uint8, clean_uint8)
                contrast_uint8 = to_uint8_image(contrast_uint8)
            except Exception as e:
                self.num_failures += 1
                if not self.fallback_on_error:
                    raise RuntimeError(
                        f"PCD ContrastImageGenerator failed at env_idx={env_idx}, "
                        f"step_idx={step_idx}, instruction={instruction!r}: {repr(e)}"
                    ) from e
                contrast_uint8 = clean_uint8.copy()

            self._save_debug_pair(clean_uint8, contrast_uint8, env_idx, step_idx)

            contrast_frame = restore_image_dtype(contrast_uint8, ref_frame)
            for t in range(horizon):
                contrast_video[env_idx, t] = contrast_frame

        out[video_key] = contrast_video
        return out


# ---------------------------------------------------------------------
# GR00T action dict <-> dense array
# ---------------------------------------------------------------------


ACTION_ORDER = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def repeat_batched_observations(
    observations: Dict[str, Any],
    num_repeats: int,
) -> Dict[str, Any]:
    """
    Repeat a vectorized GR00T observation along batch dimension.

    Original B = n_envs.
    Repeated B = n_envs * num_repeats.
    """
    repeated: Dict[str, Any] = {}

    for key, value in observations.items():
        if isinstance(value, np.ndarray):
            repeated[key] = np.repeat(value, repeats=num_repeats, axis=0)
        elif isinstance(value, list):
            out = []
            for item in value:
                out.extend([item] * num_repeats)
            repeated[key] = out
        elif isinstance(value, tuple):
            out = []
            for item in value:
                out.extend([item] * num_repeats)
            repeated[key] = tuple(out)
        else:
            repeated[key] = value

    return repeated


def extract_scalar_action_component(action_dict: Dict[str, Any], key: str) -> np.ndarray:
    flat_key = f"action.{key}"

    if flat_key in action_dict:
        arr = action_dict[flat_key]
    elif key in action_dict:
        arr = action_dict[key]
    else:
        raise KeyError(
            f"Cannot find action key '{flat_key}'. Available keys: {list(action_dict.keys())}"
        )

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected action.{key} shape (B,T,1) or (B,T), got {arr.shape}")

    return arr


def action_dict_to_array(action_dict: Dict[str, Any]) -> np.ndarray:
    comps = [extract_scalar_action_component(action_dict, key) for key in ACTION_ORDER]
    return np.stack(comps, axis=-1).astype(np.float32)


def action_array_to_dict(action_arr: np.ndarray) -> Dict[str, np.ndarray]:
    action_arr = np.asarray(action_arr, dtype=np.float32)

    if action_arr.ndim != 3:
        raise ValueError(f"Expected action_arr shape (B,T,7), got {action_arr.shape}")
    if action_arr.shape[-1] != len(ACTION_ORDER):
        raise ValueError(f"Expected last dim {len(ACTION_ORDER)}, got {action_arr.shape[-1]}")

    out: Dict[str, np.ndarray] = {}
    for i, key in enumerate(ACTION_ORDER):
        out[f"action.{key}"] = action_arr[:, :, i : i + 1].astype(np.float32)
    return out


# ---------------------------------------------------------------------
# Original PCD KDE ContrastDecoding adapter
# ---------------------------------------------------------------------


def pcd_decode_action_dict_batch(
    clean_action_dict: Dict[str, Any],
    contrast_action_dict: Dict[str, Any],
    n_envs: int,
    num_repeats: int,
    decoder: ContrastDecoding,
) -> Dict[str, np.ndarray]:
    """
    Decode clean/contrast GR00T action samples with original PCD ContrastDecoding.

    Original PCD ContrastDecoding expects tensors with shape:
        (num_repeats, T, D)

    For GR00T vector envs, PolicyClient returns batch:
        (n_envs * num_repeats, T, 7)

    This adapter applies the original decoder independently for each env slot.
    """
    clean_arr = action_dict_to_array(clean_action_dict)
    contrast_arr = action_dict_to_array(contrast_action_dict)

    if clean_arr.shape != contrast_arr.shape:
        raise ValueError(
            f"clean_arr shape {clean_arr.shape} != contrast_arr shape {contrast_arr.shape}"
        )

    total_b, t, d = clean_arr.shape
    expected_b = int(n_envs) * int(num_repeats)
    if total_b != expected_b:
        raise ValueError(
            f"Expected batch {expected_b} = n_envs {n_envs} * repeats {num_repeats}, "
            f"got {total_b}"
        )

    clean_arr = clean_arr.reshape(n_envs, num_repeats, t, d)
    contrast_arr = contrast_arr.reshape(n_envs, num_repeats, t, d)

    decoded = []
    for env_idx in range(n_envs):
        clean_tensor = torch.from_numpy(clean_arr[env_idx]).float()
        contrast_tensor = torch.from_numpy(contrast_arr[env_idx]).float()

        # Original PCD class returns shape (1,T,D).
        decoded_tensor = decoder(clean_tensor, contrast_tensor)
        decoded_one = decoded_tensor[0].detach().cpu().numpy().astype(np.float32)
        decoded.append(decoded_one)

    decoded_arr = np.stack(decoded, axis=0).astype(np.float32)
    return action_array_to_dict(decoded_arr)


def get_groot_pcd_actions(
    policy: PolicyClient,
    observations: Dict[str, Any],
    args: argparse.Namespace,
    step_idx: int,
    n_envs: int,
    contrast_generator: VectorizedPCDContrastGenerator,
    decoder: ContrastDecoding,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    GR00T + original-PCD-style action selection.

    Clean branch:
        original GR00T observation repeated num_repeats times.

    Contrast branch:
        PCD ContrastImageGenerator(grounded_sam_tracking + LAMA) generates
        object-removed visual observation, then repeat num_repeats times.

    Final:
        original PCD ContrastDecoding selects action from GR00T action samples.
    """
    num_repeats = int(args.pcd_num_repeats)
    if num_repeats <= 1:
        raise ValueError("PCD requires --pcd_num_repeats > 1")

    clean_obs = repeat_batched_observations(observations, num_repeats=num_repeats)

    contrast_base_obs = contrast_generator.generate(
        observations=observations,
        step_idx=step_idx,
    )
    contrast_obs = repeat_batched_observations(contrast_base_obs, num_repeats=num_repeats)

    options = build_policy_options(args, step_idx)

    clean_action_dict, clean_info = policy_get_action(policy, clean_obs, options=options)
    contrast_action_dict, contrast_info = policy_get_action(policy, contrast_obs, options=options)

    final_actions = pcd_decode_action_dict_batch(
        clean_action_dict=clean_action_dict,
        contrast_action_dict=contrast_action_dict,
        n_envs=n_envs,
        num_repeats=num_repeats,
        decoder=decoder,
    )

    info = {
        "mode": "groot_pcd_original_grounded_sam_tracking",
        "num_repeats": num_repeats,
        "pcd_by": args.pcd_by,
        "pcd_inpaint_mode": args.pcd_inpaint_mode,
        "pcd_generator_failures": contrast_generator.num_failures,
        "clean_info": clean_info,
        "contrast_info": contrast_info,
    }
    return final_actions, info


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


class GrootPCDRunner:
    """
    GR00T + original-PCD-style runner inside PCD.

    Important:
    - Use GR00T official create_eval_env + MultiStepWrapper.
    - Reuse original PCD ContrastImageGenerator internal logic.
    - Reuse original PCD ContrastDecoding.
    - Execute final GR00T action dict by env.step(actions).
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # wx:GR00T random feature mask
        self.run_timestamp = now_timestamp()
        # wx:GR00T random feature mask

        self.env_name = pcd_task_to_groot_env_name(args.task)
        self.output_dir = self._build_output_dir()
        self.video_dir = self.output_dir / "videos"
        self.log_path = self.output_dir / "run.log"
        self.summary_path = self.output_dir / "summary.json"
        self.episodes_path = self.output_dir / "episodes.json"
        self.debug_image_dir = self.output_dir / "pcd_debug_images"

        mkdir(self.output_dir)
        if str2bool(args.save_video):
            mkdir(self.video_dir)
        if str2bool(args.pcd_debug_save_images):
            mkdir(self.debug_image_dir)

        # wx:GR00T random feature mask
        self._write_run_config()
        # wx:GR00T random feature mask

        self._log("Output directory: " + str(self.output_dir))
        self._log("Env name: " + self.env_name)
        self._log("Policy server: " + f"{args.policy_client_host}:{args.policy_client_port}")

        self.policy = PolicyClient(
            host=args.policy_client_host,
            port=int(args.policy_client_port),
            timeout_ms=int(args.timeout_ms),
            strict=False,
        )

        if not self.policy.ping():
            raise RuntimeError(
                f"Cannot connect to GR00T server at "
                f"{args.policy_client_host}:{args.policy_client_port}"
            )

        self._log("PolicyClient ping: True")
        try:
            modality = self.policy.get_modality_config()
            self._log("Modality config: " + str(modality))
        except Exception as e:
            self._log("Warning: failed to get modality config: " + repr(e))

        self.wrapper_configs = self._build_wrapper_configs()
        self.decoder = ContrastDecoding(
            alpha=float(args.pcd_alpha),
            bandwidth_factor=float(args.pcd_bandwidth_factor),
            keep_threshold=float(args.pcd_keep_threshold),
            mode="torch",
        )
        self.contrast_generator: Optional[VectorizedPCDContrastGenerator] = None

    # wx:GR00T random feature mask
    # def _build_output_dir(self) -> Path:
    #     checkpoint = self.args.checkpoint
    #     task_name = self.args.task.replace("/", "_")

    #     run_name = self.args.run_name
    #     if not run_name:
    #         pcd_tag = (
    #             f"groot_pcd_original"
    #             f"--rep={self.args.pcd_num_repeats}"
    #             f"--alpha={self.args.pcd_alpha}"
    #             f"--bw={self.args.pcd_bandwidth_factor}"
    #             f"--keep={self.args.pcd_keep_threshold}"
    #         )
    #         contrast_tag = (
    #             f"pcdby={self.args.pcd_by}"
    #             f"_inpaint={self.args.pcd_inpaint_mode}"
    #             f"_allparts={self.args.pcd_get_all_parts}"
    #         )
    #         feat_tag = (
    #             f"feat={self.args.feature_mask_mode}"
    #             f"_keep={self.args.feature_mask_keep_ratio}"
    #             if str2bool(self.args.feature_mask_enable)
    #             else "feat=none"
    #         )
    #         run_name = (
    #             f"{pcd_tag}"
    #             f"--host={self.args.policy_client_host}"
    #             f"--port={self.args.policy_client_port}"
    #             f"--n_action_steps={self.args.n_action_steps}"
    #             f"--n_envs={self.args.n_envs}"
    #             f"--{contrast_tag}"
    #             f"--{feat_tag}"
    #         )

    #     return Path(self.args.output_root) / checkpoint / run_name / task_name
    # wx:GR00T random feature mask
    def _build_output_dir(self) -> Path:
        checkpoint_name = safe_dir_name(
            Path(str(self.args.checkpoint).replace("\\", "/")).name
        )
        task_name = safe_dir_name(self.args.task.replace("/", "_"))

        user_run_name = str(self.args.run_name).strip()

        if user_run_name:
            run_path = safe_relative_path(user_run_name)
            if not bool(self.args.no_timestamp):
                run_path = run_path.parent / f"{run_path.name}_{self.run_timestamp}"
            self.run_name_resolved = str(run_path)
            return Path(self.args.output_root) / checkpoint_name / run_path / task_name

        else:
            if str2bool(self.args.pcd_enable):
                method_tag = (
                    f"groot_pcd_original"
                    f"_rep={self.args.pcd_num_repeats}"
                    f"_alpha={self.args.pcd_alpha}"
                    f"_bw={self.args.pcd_bandwidth_factor}"
                    f"_keep={self.args.pcd_keep_threshold}"
                    f"_by={self.args.pcd_by}"
                    f"_inpaint={self.args.pcd_inpaint_mode}"
                )
            else:
                method_tag = "groot_official"

            if str2bool(self.args.feature_mask_enable):
                target_tag = safe_dir_name(self.args.feature_mask_target, max_len=80)
                feat_tag = (
                    f"feat={self.args.feature_mask_mode}"
                    f"_target={target_tag}"
                    f"_keep={self.args.feature_mask_keep_ratio}"
                    f"_seed={self.args.feature_mask_seed}"
                    f"_rescale={self.args.feature_mask_rescale}"
                )
            else:
                feat_tag = "feat=none"

            run_name = (
                f"{method_tag}"
                f"_host={self.args.policy_client_host}"
                f"_port={self.args.policy_client_port}"
                f"_act={self.args.n_action_steps}"
                f"_env={self.args.n_envs}"
                f"_{feat_tag}"
            )

            run_name = safe_dir_name(run_name)

        self.run_name_resolved = run_name

        return Path(self.args.output_root) / checkpoint_name / run_name / task_name
    # wx:GR00T random feature mask

    def _build_wrapper_configs(self) -> WrapperConfigs:
        video_dir = str(self.video_dir) if str2bool(self.args.save_video) else None

        return WrapperConfigs(
            video=VideoConfig(
                video_dir=video_dir,
                steps_per_render=int(self.args.steps_per_render),
                max_episode_steps=int(self.args.max_episode_steps),
                fps=int(self.args.video_fps),
                overlay_text=str2bool(self.args.overlay_text),
                n_action_steps=int(self.args.n_action_steps),
            ),
            multistep=MultiStepConfig(
                video_delta_indices=np.array([0]),
                state_delta_indices=np.array([0]),
                n_action_steps=int(self.args.n_action_steps),
                max_episode_steps=int(self.args.max_episode_steps),
                terminate_on_success=str2bool(self.args.terminate_on_success),
            ),
        )
        
    # wx:GR00T random feature mask
    def _write_run_config(self):
        """
        Save all experiment settings before rollout starts.
        This is useful for random/adaptive feature-mask experiments.
        """
        config = {
            "result_dir": str(self.output_dir),
            "run_name": self.args.run_name,
            "run_name_resolved": getattr(self, "run_name_resolved", None),
            "run_timestamp": self.run_timestamp,
            "no_timestamp": bool(self.args.no_timestamp),
            "force": str2bool(self.args.force),

            "task": self.args.task,
            "env_name": self.env_name,
            "checkpoint": self.args.checkpoint,
            "output_root": self.args.output_root,
            "n_episodes": int(self.args.n_episodes),
            "n_envs": int(self.args.n_envs),
            "n_action_steps": int(self.args.n_action_steps),
            "max_episode_steps": int(self.args.max_episode_steps),

            "policy_server": {
                "host": self.args.policy_client_host,
                "port": int(self.args.policy_client_port),
                "timeout_ms": int(self.args.timeout_ms),
            },

            "pcd": {
                "enable": str2bool(self.args.pcd_enable),
                "num_repeats": int(self.args.pcd_num_repeats),
                "alpha": float(self.args.pcd_alpha),
                "bandwidth_factor": float(self.args.pcd_bandwidth_factor),
                "keep_threshold": float(self.args.pcd_keep_threshold),
                "by": self.args.pcd_by,
                "inpaint_mode": self.args.pcd_inpaint_mode,
                "get_all_parts": str2bool(self.args.pcd_get_all_parts),
                "fallback_on_error": str2bool(self.args.pcd_fallback_on_error),
            },

            "feature_mask": {
                "enable": str2bool(self.args.feature_mask_enable),
                "target": self.args.feature_mask_target,
                "mode": self.args.feature_mask_mode,
                "keep_ratio": float(self.args.feature_mask_keep_ratio),
                "seed": int(self.args.feature_mask_seed),
                "rescale": str2bool(self.args.feature_mask_rescale),
                "verbose": str2bool(getattr(self.args, "feature_mask_verbose", False)),
                "print_candidates": str2bool(getattr(self.args, "feature_mask_print_candidates", False)),
            },

            "video": {
                "save_video": str2bool(self.args.save_video),
                "steps_per_render": int(self.args.steps_per_render),
                "video_fps": int(self.args.video_fps),
                "overlay_text": str2bool(self.args.overlay_text),
            },
        }

        safe_json_dump(config, self.output_dir / "run_config.json")
    # wx:GR00T random feature mask

    def _log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - {msg}"
        print(line)
        mkdir(self.output_dir)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _make_env(self):
        n_envs = int(self.args.n_envs)

        env_fns = [
            partial(
                create_eval_env,
                env_idx=idx,
                env_name=self.env_name,
                total_n_envs=n_envs,
                wrapper_configs=self.wrapper_configs,
            )
            for idx in range(n_envs)
        ]

        if n_envs == 1:
            env = gym.vector.SyncVectorEnv(env_fns)
        else:
            env = _RobustAsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

        return env

    def _build_contrast_generator(self, n_envs: int) -> VectorizedPCDContrastGenerator:
        return VectorizedPCDContrastGenerator(
            n_envs=n_envs,
            by=self.args.pcd_by,
            inpaint_mode=self.args.pcd_inpaint_mode,
            get_all_parts=str2bool(self.args.pcd_get_all_parts),
            video_key=self.args.pcd_video_key,
            fallback_on_error=str2bool(self.args.pcd_fallback_on_error),
            debug_save_dir=self.debug_image_dir if str2bool(self.args.pcd_debug_save_images) else None,
            debug_image_interval=int(self.args.pcd_debug_image_interval),
        )

    def run(self) -> Dict[str, Any]:
        args = self.args
        
        # wx:GR00T random feature mask
        if self.summary_path.exists() and not str2bool(args.force):
            self._log(f"[Skip] summary.json already exists: {self.summary_path}")
            with open(self.summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # wx:GR00T random feature mask

        n_envs = int(args.n_envs)
        n_episodes = max(int(args.n_episodes), n_envs)
        max_episode_steps = int(args.max_episode_steps)

        self._log(
            f"Start GR00T+original-PCD rollout: "
            f"n_episodes={n_episodes}, n_envs={n_envs}, "
            f"n_action_steps={args.n_action_steps}, max_episode_steps={max_episode_steps}, "
            f"pcd_num_repeats={args.pcd_num_repeats}, pcd_by={args.pcd_by}, "
            f"pcd_inpaint_mode={args.pcd_inpaint_mode}"
        )

        env = self._make_env()

        episode_lengths = []
        episode_successes = []
        episode_records = []
        episode_infos = defaultdict(list)

        current_lengths = [0 for _ in range(n_envs)]
        current_successes = [False for _ in range(n_envs)]
        completed_episodes = 0

        start_time = time.time()

        observations, _ = env.reset()
        self.policy.reset()

        if str2bool(args.pcd_enable):
            self.contrast_generator = self._build_contrast_generator(n_envs)
            self.contrast_generator.reset()
        else:
            self.contrast_generator = None

        pbar = tqdm(total=n_episodes, desc="Episodes")
        step_idx = 0

        try:
            while completed_episodes < n_episodes:
                if str2bool(args.pcd_enable):
                    assert self.contrast_generator is not None
                    actions, policy_info = get_groot_pcd_actions(
                        policy=self.policy,
                        observations=observations,
                        args=args,
                        step_idx=step_idx,
                        n_envs=n_envs,
                        contrast_generator=self.contrast_generator,
                        decoder=self.decoder,
                    )
                else:
                    options = build_policy_options(args, step_idx)
                    actions, policy_info = policy_get_action(
                        self.policy,
                        observations,
                        options=options,
                    )

                next_obs, rewards, terminations, truncations, env_infos = env.step(actions)

                if str2bool(args.debug) and step_idx < int(args.debug_steps):
                    self._log(f"Step {step_idx}: action keys = {list(actions.keys())}")
                    self._log(f"Step {step_idx}: env info keys = {list(env_infos.keys())}")
                    for k, v in actions.items():
                        try:
                            self._log(f"Step {step_idx}: {k} shape = {np.asarray(v).shape}")
                        except Exception:
                            pass

                for env_idx in range(n_envs):
                    current_lengths[env_idx] += 1

                    if "success" in env_infos:
                        current_successes[env_idx] |= as_bool_success(env_infos["success"][env_idx])

                    final_info = None
                    if "final_info" in env_infos:
                        try:
                            final_info = env_infos["final_info"][env_idx]
                        except Exception:
                            final_info = None

                    if final_info is not None:
                        if isinstance(final_info, dict) and "success" in final_info:
                            current_successes[env_idx] |= as_bool_success(final_info["success"])

                    done = bool(terminations[env_idx]) or bool(truncations[env_idx])

                    if done:
                        record = {
                            "episode_index": len(episode_successes),
                            "env_index": env_idx,
                            "length": int(current_lengths[env_idx]),
                            "success": bool(current_successes[env_idx]),
                        }

                        if final_info is not None:
                            record["final_info"] = final_info

                        for key in ["task_progress", "q_score", "valid"]:
                            if key in env_infos:
                                try:
                                    val = env_infos[key][env_idx]
                                    record[key] = val
                                    episode_infos[key].append(val)
                                except Exception:
                                    pass

                        episode_lengths.append(int(current_lengths[env_idx]))
                        episode_successes.append(bool(current_successes[env_idx]))
                        episode_records.append(record)

                        completed_episodes += 1
                        pbar.update(1)

                        self._log(
                            f"Episode {len(episode_successes) - 1} finished: "
                            f"success={record['success']}, length={record['length']}"
                        )

                        current_lengths[env_idx] = 0
                        current_successes[env_idx] = False

                        # grounded_sam_tracking can be stateful. Reset the PCD
                        # generator state for this env slot when its episode ends.
                        if self.contrast_generator is not None:
                            self.contrast_generator.reset_env(env_idx)

                        if completed_episodes >= n_episodes:
                            break

                observations = next_obs
                step_idx += 1

        finally:
            pbar.close()
            try:
                env.close()
            except Exception:
                pass

        elapsed = time.time() - start_time
        success_rate = float(np.mean(episode_successes[:n_episodes])) if episode_successes else 0.0

        summary = {
            "env_name": self.env_name,
            "task": args.task,
            "checkpoint": args.checkpoint,
            # wx:GR00T random feature mask
            "run_name": args.run_name,
            "run_name_resolved": getattr(self, "run_name_resolved", None),
            "run_timestamp": self.run_timestamp,
            "no_timestamp": bool(args.no_timestamp),
            # wx:GR00T random feature mask
            "n_episodes_requested": int(args.n_episodes),
            "n_episodes_collected": int(len(episode_successes)),
            "n_envs": int(args.n_envs),
            "n_action_steps": int(args.n_action_steps),
            "max_episode_steps": int(args.max_episode_steps),
            "success_rate": success_rate,
            "episode_successes": episode_successes[:n_episodes],
            "episode_lengths": episode_lengths[:n_episodes],
            "elapsed_seconds": elapsed,
            "output_dir": str(self.output_dir),
            "video_dir": str(self.video_dir) if str2bool(args.save_video) else None,
            "pcd": {
                "enable": str2bool(args.pcd_enable),
                "num_repeats": int(args.pcd_num_repeats),
                "alpha": float(args.pcd_alpha),
                "bandwidth_factor": float(args.pcd_bandwidth_factor),
                "keep_threshold": float(args.pcd_keep_threshold),
                "decoder": "contrast_policies.kde_contrast_decoding.ContrastDecoding",
                "by": args.pcd_by,
                "inpaint_mode": args.pcd_inpaint_mode,
                "get_all_parts": str2bool(args.pcd_get_all_parts),
                "video_key": args.pcd_video_key,
                "fallback_on_error": str2bool(args.pcd_fallback_on_error),
                "generator_failures": self.contrast_generator.num_failures
                if self.contrast_generator is not None
                else 0,
            },
            "feature_mask": {
                "enable": str2bool(args.feature_mask_enable),
                "target": args.feature_mask_target,
                "mode": args.feature_mask_mode,
                "keep_ratio": float(args.feature_mask_keep_ratio),
                "seed": int(args.feature_mask_seed),
                "rescale": str2bool(args.feature_mask_rescale),
            },
        }

        safe_json_dump(summary, self.summary_path)
        safe_json_dump(episode_records, self.episodes_path)

        self._log("----------------------------------------")
        self._log("Results:")
        self._log(f"success_rate: {success_rate}")
        self._log(f"episode_successes: {episode_successes[:n_episodes]}")
        self._log(f"episode_lengths: {episode_lengths[:n_episodes]}")
        self._log(f"elapsed_seconds: {elapsed:.2f}")
        self._log(f"summary saved to: {self.summary_path}")
        if str2bool(args.save_video):
            self._log(f"videos saved to: {self.video_dir}")
        if str2bool(args.pcd_debug_save_images):
            self._log(f"PCD debug images saved to: {self.debug_image_dir}")
        self._log("----------------------------------------")

        return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GR00T + original-PCD grounded_sam_tracking runner inside PCD."
    )

    # PCD-style arguments
    parser.add_argument("--task", type=str, default="google_robot_pick_coke_can")
    parser.add_argument("--checkpoint", type=str, default="GR00T-N1.6-fractal")
    parser.add_argument("--n-trajs", "--n_episodes", dest="n_episodes", type=int, default=10)
    parser.add_argument("--output-root", type=str, default="./results")
    parser.add_argument("--run-name", type=str, default="")
    # wx:GR00T random feature mask
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--force", type=str2bool, default=False)
    # wx:GR00T random feature mask

    # GR00T official-style arguments
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument("--timeout_ms", type=int, default=60000)

    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--terminate_on_success", type=str2bool, default=True)

    # Video
    parser.add_argument("--save_video", type=str2bool, default=False)
    parser.add_argument("--steps_per_render", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--overlay_text", type=str2bool, default=True)

    # Debug
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--debug_steps", type=int, default=3)

    # GR00T + original PCD decoding
    parser.add_argument("--pcd_enable", type=str2bool, default=True)
    parser.add_argument("--pcd_num_repeats", type=int, default=24)
    parser.add_argument("--pcd_alpha", type=float, default=0.2)
    parser.add_argument("--pcd_bandwidth_factor", type=float, default=1.0)
    parser.add_argument("--pcd_keep_threshold", type=float, default=0.5)

    # Original PCD contrast image generator config
    parser.add_argument("--pcd_by", type=str, default="grounded_sam_tracking")
    parser.add_argument("--pcd_inpaint_mode", type=str, default="lama")
    parser.add_argument("--pcd_get_all_parts", type=str2bool, default=False)
    parser.add_argument("--pcd_video_key", type=str, default="")
    parser.add_argument("--pcd_fallback_on_error", type=str2bool, default=False)
    parser.add_argument("--pcd_debug_save_images", type=str2bool, default=False)
    parser.add_argument("--pcd_debug_image_interval", type=int, default=20)

    # Stage 3: feature-level mask hook
    parser.add_argument("--feature_mask_enable", type=str2bool, default=False)
    parser.add_argument("--feature_mask_target", type=str, default="image_tokens")
    parser.add_argument("--feature_mask_mode", type=str, default="dim")
    parser.add_argument("--feature_mask_keep_ratio", type=float, default=1.0)
    parser.add_argument("--feature_mask_seed", type=int, default=0)
    parser.add_argument("--feature_mask_rescale", type=str2bool, default=True)
    # wx:GR00T random feature mask
    parser.add_argument("--feature_mask_verbose", type=str2bool, default=False)
    parser.add_argument("--feature_mask_print_candidates", type=str2bool, default=False)
    # wx:GR00T random feature mask
    
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    runner = GrootPCDRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
