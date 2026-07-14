# wx:集成GR00T-N1.6
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from gr00t.policy.server_client import PolicyClient


def _cfg_attr(cfg: Any, name: str, default=None):
    """Support both ModalityConfig objects and plain dicts."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return default


def _to_bool(x: Any) -> bool:
    """
    PCD --opts may pass booleans as strings, e.g. debug True / False.
    bool("False") is True in Python, so we need explicit parsing.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["1", "true", "yes", "y", "on"]
    return bool(x)


class Gr00tClientInference:
    """
    Minimal GR00T baseline policy for PCD.

    Current stage:
    - PCD owns SimplerEnv loop.
    - GR00T runs in a separate server.
    - This client converts PCD image/proprio/instruction to GR00T sim-policy format.
    - This client converts split GR00T actions back to PCD action dict.

    Reserved for later:
    - image_mask_mode: for image-level mask.
    - feature_mask_cfg: for feature-level mask through custom GR00T server options.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 60000,
        exec_horizon: int = 1,
        debug: bool = False,
        image_mask_mode: str = "none",
        feature_mask_enable: bool = False,
        feature_mask_target: str = "image_tokens",
        feature_mask_mode: str = "dim",
        feature_mask_keep_ratio: float = 1.0,
        feature_mask_seed: int = 0,
        **kwargs,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.exec_horizon = int(exec_horizon)
        self.debug = _to_bool(debug)

        # Reserved for later image-level mask.
        self.image_mask_mode = image_mask_mode

        # Reserved for later feature-level mask.
        self.feature_mask_cfg = {
            "enable": _to_bool(feature_mask_enable),
            "target": feature_mask_target,
            "mode": feature_mask_mode,
            "keep_ratio": float(feature_mask_keep_ratio),
            "seed": int(feature_mask_seed),
        }

        self.client = PolicyClient(
            host=self.host,
            port=self.port,
            timeout_ms=self.timeout_ms,
            strict=False,
        )

        if not self.client.ping():
            raise RuntimeError(f"Cannot connect to GR00T server at {self.host}:{self.port}")

        self.modality_config = self.client.get_modality_config()

        self.video_keys = list(_cfg_attr(self.modality_config["video"], "modality_keys", []))
        self.state_keys = list(_cfg_attr(self.modality_config["state"], "modality_keys", []))
        self.action_keys = list(_cfg_attr(self.modality_config["action"], "modality_keys", []))

        if len(self.video_keys) == 0:
            raise RuntimeError(f"No video keys in GR00T modality_config: {self.modality_config}")
        if len(self.state_keys) == 0:
            raise RuntimeError(f"No state keys in GR00T modality_config: {self.modality_config}")
        if len(self.action_keys) == 0:
            raise RuntimeError(f"No action keys in GR00T modality_config: {self.modality_config}")

        self.instruction = None
        self.seed = None

        if self.debug:
            print("[GR00T-PCD] host:", self.host)
            print("[GR00T-PCD] port:", self.port)
            print("[GR00T-PCD] video_keys:", self.video_keys)
            print("[GR00T-PCD] state_keys:", self.state_keys)
            print("[GR00T-PCD] action_keys:", self.action_keys)
            print("[GR00T-PCD] modality_config:", self.modality_config)

    def reset(self, instruction: Optional[str] = None, seed: Optional[int] = None):
        self.instruction = instruction
        self.seed = seed

        # Some PolicyClient versions accept options; some may not.
        try:
            self.client.reset(options={"seed": seed})
        except TypeError:
            self.client.reset()

        return {}

    def _video_horizon(self) -> int:
        delta_indices = _cfg_attr(self.modality_config["video"], "delta_indices", [0])
        return len(delta_indices)

    def _state_horizon(self) -> int:
        delta_indices = _cfg_attr(self.modality_config["state"], "delta_indices", [0])
        return len(delta_indices)

    def _build_video(self, image: np.ndarray) -> np.ndarray:
        """
        Build video array:
            shape: (B, T, H, W, C)
            dtype: uint8
        """
        if image is None:
            raise ValueError("GR00T requires image input, but got None.")

        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H,W,C), got {image.shape}")

        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")

        t = self._video_horizon()
        video = np.stack([image] * t, axis=0)   # (T, H, W, C)
        video = video[None]                     # (1, T, H, W, C)
        return video

    def _make_btd_scalar(self, value: float) -> np.ndarray:
        """
        Build one scalar stream:
            shape: (B, T, D) = (1, state_horizon, 1)
            dtype: float32
        """
        t = self._state_horizon()
        return np.full((1, t, 1), float(value), dtype=np.float32)

    def _proprio_to_state_values(
        self,
        proprio: np.ndarray,
        env_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Convert PCD proprio to GR00T split state values.

        First-stage baseline:
        - PCD currently passes obs['agent']['eef_pos'], usually [x, y, z].
        - GR00T expects split scalar states:
            x, y, z, roll, pitch, yaw, pad, gripper
        - If roll/pitch/yaw/gripper are unavailable, we fill them with 0.

        Later:
        - If we pass full env_obs from parallel_inference.py, this function can be extended
          to read more accurate rotation and gripper state.
        """
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)

        values = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "pad": 0.0,
            "gripper": 0.0,
        }

        # Minimal compatibility with current PCD call:
        # policy.step(image, instruction, proprio=obs['agent']['eef_pos'])
        if proprio.shape[0] > 0:
            values["x"] = float(proprio[0])
        if proprio.shape[0] > 1:
            values["y"] = float(proprio[1])
        if proprio.shape[0] > 2:
            values["z"] = float(proprio[2])
        if proprio.shape[0] > 3:
            values["roll"] = float(proprio[3])
        if proprio.shape[0] > 4:
            values["pitch"] = float(proprio[4])
        if proprio.shape[0] > 5:
            values["yaw"] = float(proprio[5])
        if proprio.shape[0] > 6:
            values["gripper"] = float(proprio[6])

        # Optional future hook: use env_obs if parallel_inference.py later passes it.
        # We keep this conservative now to avoid breaking baseline.
        if env_obs is not None and isinstance(env_obs, dict):
            agent_obs = env_obs.get("agent", {})
            if isinstance(agent_obs, dict):
                # Some SimplerEnv variants may expose gripper state under different keys.
                for k in ["gripper", "gripper_pos", "gripper_position"]:
                    if k in agent_obs:
                        try:
                            g = np.asarray(agent_obs[k]).reshape(-1)
                            if g.size > 0:
                                values["gripper"] = float(g[0])
                        except Exception:
                            pass

        return values

    def _build_state_dict(
        self,
        proprio: np.ndarray,
        env_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build flat GR00T state dict:
            state.x
            state.y
            state.z
            state.roll
            state.pitch
            state.yaw
            state.pad
            state.gripper

        Each value is shaped as (B,T,1).
        """
        values = self._proprio_to_state_values(proprio, env_obs=env_obs)

        state_dict: Dict[str, np.ndarray] = {}
        for key in self.state_keys:
            state_dict[f"state.{key}"] = self._make_btd_scalar(values.get(key, 0.0))

        return state_dict

    def _build_observation(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: np.ndarray,
        env_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build flat GR00T observation.

        We provide both:
        - video.<key>
        - state.<key>
        - task
        - annotation.human.action.task_description

        Reason:
        Some GR00T server paths use Gr00tSimPolicyWrapper and accept "task",
        while the underlying Gr00tPolicy checks the original language key:
        "annotation.human.action.task_description".
        Providing both is the safest baseline-compatible choice.
        """
        obs: Dict[str, Any] = {}

        video = self._build_video(image)
        for key in self.video_keys:
            obs[f"video.{key}"] = video

        obs.update(self._build_state_dict(proprio, env_obs=env_obs))

        # SimplerEnv wrapper style.
        obs["task"] = [instruction]

        # Native GR00T language key.
        obs["annotation.human.action.task_description"] = [instruction]

        return obs

    def _build_options(self) -> Dict[str, Any]:
        """
        Reserved for stage 3 feature-level mask.

        Official GR00T server will ignore feature_mask now.
        Later, a custom feature-mask server can read this option.
        """
        return {
            "feature_mask": self.feature_mask_cfg,
        }

    def _extract_scalar_action(
        self,
        action_dict: Dict[str, Any],
        key: str,
    ) -> np.ndarray:
        """
        Extract one scalar action component from GR00T action_dict.

        Expected candidates:
            action.x
            x

        Expected shape:
            (B,T,1), but this function also tolerates (B,T).
        """
        flat_key = f"action.{key}"

        if flat_key in action_dict:
            arr = action_dict[flat_key]
        elif key in action_dict:
            arr = action_dict[key]
        else:
            raise KeyError(
                f"Cannot find action key '{flat_key}'. "
                f"Available keys: {list(action_dict.keys())}"
            )

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 3:
            # (B,T,1) -> (B,T)
            arr = arr[:, :, 0]
        elif arr.ndim == 2:
            # already (B,T)
            pass
        else:
            raise ValueError(f"Expected action.{key} shape (B,T,1) or (B,T), got {arr.shape}")

        return arr

    def _extract_action_array(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert GR00T split action dict to one action array:
            shape: (B,T,7)
            order: [x, y, z, roll, pitch, yaw, gripper]
        """
        ordered_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

        missing = [k for k in ordered_keys if f"action.{k}" not in action_dict and k not in action_dict]
        if missing:
            raise KeyError(
                f"Missing GR00T action keys: {missing}. "
                f"Available keys: {list(action_dict.keys())}"
            )

        comps = [self._extract_scalar_action(action_dict, key) for key in ordered_keys]

        # list of (B,T) -> (B,T,7)
        action_arr = np.stack(comps, axis=-1).astype(np.float32)
        return action_arr

    def _to_pcd_action_dicts(self, action_arr: np.ndarray):
        """
        Convert GR00T action chunk to PCD action dict list.

        PCD expects each step action as:
            {
                "world_vector": np.ndarray shape (3,),
                "rot_axangle": np.ndarray shape (3,),
                "gripper": np.ndarray shape (1,),
                "terminate_episode": np.ndarray shape (1,),
            }

        GR00T split action has been packed as:
            [x, y, z, roll, pitch, yaw, gripper]
        """
        if action_arr.ndim != 3:
            raise ValueError(f"Expected action_arr shape (B,T,7), got {action_arr.shape}")

        if action_arr.shape[-1] < 7:
            raise ValueError(f"Expected at least 7D action, got {action_arr.shape}")

        chunk = action_arr[0]  # (T, 7)

        action_dicts = []
        for a in chunk[: self.exec_horizon]:
            a = np.asarray(a, dtype=np.float32).reshape(-1)

            action_dicts.append(
                {
                    "world_vector": a[:3].astype(np.float32),
                    "rot_axangle": a[3:6].astype(np.float32),
                    "gripper": a[6:7].astype(np.float32),
                    "terminate_episode": np.array([0.0], dtype=np.float32),
                }
            )

        return action_dicts

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """
        PCD-compatible step.

        Current PCD call:
            raw_action, actions = policy.step(
                image,
                instruction,
                proprio=obs['agent']['eef_pos']
            )

        Optional future call:
            raw_action, actions = policy.step(
                image,
                instruction,
                proprio=obs['agent']['eef_pos'],
                env_obs=obs
            )
        """
        if proprio is None:
            raise ValueError("GR00T baseline requires proprio, but got None.")

        env_obs = kwargs.get("env_obs", None)

        obs = self._build_observation(
            image=image,
            instruction=instruction,
            proprio=proprio,
            env_obs=env_obs,
        )

        action_dict, info = self.client.get_action(
            obs,
            options=self._build_options(),
        )

        action_arr = self._extract_action_array(action_dict)
        action_dicts = self._to_pcd_action_dicts(action_arr)

        if self.debug:
            print("[GR00T-PCD] observation keys:", list(obs.keys()))
            print("[GR00T-PCD] action_dict keys:", list(action_dict.keys()))
            print("[GR00T-PCD] action_arr shape:", action_arr.shape)
            print("[GR00T-PCD] first action:", action_arr[0, 0])
            print("[GR00T-PCD] first PCD action:", action_dicts[0])

        if len(action_dicts) == 1:
            return action_arr, action_dicts[0]

        return action_arr, action_dicts