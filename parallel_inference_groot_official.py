# wx: GR00T-N1.6 official-style inference runner inside PCD
# File: PCD/parallel_inference_groot_official.py

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
    _RobustAsyncVectorEnv,
)
from gr00t.policy.server_client import PolicyClient


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
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def as_bool_success(x: Any) -> bool:
    """
    GR00T/SimplerEnv infos may return bool/list/np.ndarray/int.
    Official rollout_policy.py handles these cases similarly.
    """
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


# ---------------------------------------------------------------------
# Future hook 1: image-level mask
# ---------------------------------------------------------------------


def apply_simple_image_mask_to_observations(
    observations: Dict[str, Any],
    mode: str = "none",
    ratio: float = 0.25,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Baseline stage: keep mode='none'.

    This function is intentionally placed here for Stage 2 image-level mask.

    Expected vectorized GR00T observation:
        observations["video.image_0"]: np.ndarray, shape roughly
            (n_envs, T, H, W, C)

    Supported simple modes:
        none        : no change
        black_center: black out center patch
        random_patch: black out one random patch per batch item / time step
    """
    mode = str(mode).lower()
    if mode in ["none", "false", "off", ""]:
        return observations

    out = dict(observations)
    rng = np.random.default_rng(seed)

    for key, value in observations.items():
        if not key.startswith("video"):
            continue

        arr = np.asarray(value)
        if arr.ndim < 5:
            # Expected vector env shape is usually (B,T,H,W,C).
            # If shape is different, skip instead of breaking baseline.
            continue

        masked = arr.copy()
        b, t, h, w, c = masked.shape[:5]

        patch_h = max(1, int(h * ratio))
        patch_w = max(1, int(w * ratio))

        if mode == "black_center":
            y0 = max(0, (h - patch_h) // 2)
            x0 = max(0, (w - patch_w) // 2)
            masked[:, :, y0 : y0 + patch_h, x0 : x0 + patch_w, :] = 0

        elif mode == "random_patch":
            for bi in range(b):
                for ti in range(t):
                    y0 = int(rng.integers(0, max(1, h - patch_h + 1)))
                    x0 = int(rng.integers(0, max(1, w - patch_w + 1)))
                    masked[bi, ti, y0 : y0 + patch_h, x0 : x0 + patch_w, :] = 0

        else:
            raise ValueError(
                f"Unknown image_mask_mode='{mode}'. "
                "Supported: none, black_center, random_patch"
            )

        out[key] = masked

    return out


# ---------------------------------------------------------------------
# Future hook 2: feature-level mask options
# ---------------------------------------------------------------------


def build_policy_options(args: argparse.Namespace, step_idx: int) -> Optional[Dict[str, Any]]:
    """
    Baseline stage:
        feature_mask_enable=False, returns None.

    Stage 3:
        start a custom GR00T feature-mask server, then set:
            --feature_mask_enable True
            --feature_mask_keep_ratio 0.9
            --feature_mask_seed 0

    The official server will ignore unknown options unless customized.
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
        },
        # useful later if you want deterministic clean/masked comparisons
        "sample_seed": int(args.feature_mask_seed) + int(step_idx),
    }


def policy_get_action(policy: PolicyClient, observations: Dict[str, Any], options: Optional[dict]):
    """
    PolicyClient in GR00T supports get_action(observation, options).
    Keep a TypeError fallback for compatibility with minor API differences.
    """
    if options is None:
        return policy.get_action(observations)

    try:
        return policy.get_action(observations, options=options)
    except TypeError:
        return policy.get_action(observations)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


class GrootOfficialRunner:
    """
    GR00T official-style runner inside PCD.

    Difference from the previous PCD integration:
    - Do NOT convert GR00T action dict to PCD 7D vector.
    - Do NOT use PCD's original SimplerEnv loop.
    - Use GR00T official create_eval_env + MultiStepWrapper.
    - Directly execute env.step(actions), where actions is GR00T action dict.

    This should reproduce the official GR00T baseline much better.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.env_name = pcd_task_to_groot_env_name(args.task)
        self.output_dir = self._build_output_dir()
        self.video_dir = self.output_dir / "videos"
        self.log_path = self.output_dir / "run.log"
        self.summary_path = self.output_dir / "summary.json"
        self.episodes_path = self.output_dir / "episodes.json"

        mkdir(self.output_dir)
        if str2bool(args.save_video):
            mkdir(self.video_dir)

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

    def _build_output_dir(self) -> Path:
        checkpoint = self.args.checkpoint
        task_name = self.args.task.replace("/", "_")

        run_name = self.args.run_name
        if not run_name:
            mask_tag = self.args.image_mask_mode
            feat_tag = (
                f"feat_{self.args.feature_mask_mode}_{self.args.feature_mask_keep_ratio}"
                if str2bool(self.args.feature_mask_enable)
                else "feat_none"
            )
            run_name = (
                f"official"
                f"--host={self.args.policy_client_host}"
                f"--port={self.args.policy_client_port}"
                f"--n_action_steps={self.args.n_action_steps}"
                f"--n_envs={self.args.n_envs}"
                f"--imgmask={mask_tag}"
                f"--{feat_tag}"
            )

        return Path(self.args.output_root) / checkpoint / run_name / task_name

    def _build_wrapper_configs(self) -> WrapperConfigs:
        if str2bool(self.args.save_video):
            video_dir = str(self.video_dir)
        else:
            # Keep official behavior available, but avoid saving videos by default.
            video_dir = None

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
            # Same robust async vector env as GR00T official rollout_policy.py.
            env = _RobustAsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

        return env

    def run(self) -> Dict[str, Any]:
        args = self.args

        n_envs = int(args.n_envs)
        n_episodes = max(int(args.n_episodes), n_envs)
        max_episode_steps = int(args.max_episode_steps)

        self._log(
            f"Start GR00T official-style rollout: "
            f"n_episodes={n_episodes}, n_envs={n_envs}, "
            f"n_action_steps={args.n_action_steps}, max_episode_steps={max_episode_steps}"
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

        pbar = tqdm(total=n_episodes, desc="Episodes")

        step_idx = 0

        try:
            while completed_episodes < n_episodes:
                policy_observations = apply_simple_image_mask_to_observations(
                    observations,
                    mode=args.image_mask_mode,
                    ratio=float(args.image_mask_ratio),
                    seed=int(args.image_mask_seed) + step_idx,
                )

                options = build_policy_options(args, step_idx)

                actions, policy_info = policy_get_action(
                    self.policy,
                    policy_observations,
                    options=options,
                )

                next_obs, rewards, terminations, truncations, env_infos = env.step(actions)

                if str2bool(args.debug) and step_idx < int(args.debug_steps):
                    self._log(f"Step {step_idx}: action keys = {list(actions.keys())}")
                    self._log(f"Step {step_idx}: env info keys = {list(env_infos.keys())}")

                for env_idx in range(n_envs):
                    current_lengths[env_idx] += 1

                    # Track success from step info.
                    if "success" in env_infos:
                        current_successes[env_idx] |= as_bool_success(env_infos["success"][env_idx])

                    # Track success from final_info if available.
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
                        # Official runner has additional handling for task_progress/q_score/valid.
                        # Keep these fields if the env provides them.
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
            "image_mask": {
                "mode": args.image_mask_mode,
                "ratio": float(args.image_mask_ratio),
                "seed": int(args.image_mask_seed),
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
        self._log("----------------------------------------")

        return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GR00T official-style inference runner inside PCD."
    )

    # PCD-style arguments
    parser.add_argument("--task", type=str, default="google_robot_pick_coke_can")
    parser.add_argument("--checkpoint", type=str, default="GR00T-N1.6-fractal")
    parser.add_argument("--n-trajs", "--n_episodes", dest="n_episodes", type=int, default=10)
    parser.add_argument("--output-root", type=str, default="./results")
    parser.add_argument("--run-name", type=str, default="")

    # GR00T official-style arguments
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument("--timeout_ms", type=int, default=60000)

    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--n_envs", type=int, default=5)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--terminate_on_success", type=str2bool, default=True)

    # Video
    parser.add_argument("--save_video", type=str2bool, default=True)
    parser.add_argument("--steps_per_render", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--overlay_text", type=str2bool, default=True)

    # Debug
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--debug_steps", type=int, default=3)

    # Stage 2: image-level mask hook
    parser.add_argument("--image_mask_mode", type=str, default="none")
    parser.add_argument("--image_mask_ratio", type=float, default=0.25)
    parser.add_argument("--image_mask_seed", type=int, default=0)

    # Stage 3: feature-level mask hook
    parser.add_argument("--feature_mask_enable", type=str2bool, default=False)
    parser.add_argument("--feature_mask_target", type=str, default="image_tokens")
    parser.add_argument("--feature_mask_mode", type=str, default="dim")
    parser.add_argument("--feature_mask_keep_ratio", type=float, default=1.0)
    parser.add_argument("--feature_mask_seed", type=int, default=0)
    parser.add_argument("--feature_mask_rescale", type=str2bool, default=True)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    runner = GrootOfficialRunner(args)
    runner.run()


if __name__ == "__main__":
    main()