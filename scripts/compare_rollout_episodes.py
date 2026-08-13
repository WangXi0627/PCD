# wx:Dynamic gate v1

#!/usr/bin/env python3
"""
Compare two query-level rollout episodes.

Primary use:
    lossless closed-loop regression between the original PiZero policy and
    PiZeroDynamicGateInference with IdentityVisualGate.
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
from typing import Dict, Iterable, Tuple

import numpy as np


EXACT_ARRAY_KEYS = [
    "images",
    "next_images",
    "raw_action_lengths",
    "executed_action_lengths",
    "query_indices",
    "env_step_starts",
    "env_step_ends",
    "terminated",
    "truncated",
    "success_at_end",
    "instruction_changed",
]


FLOAT_ARRAY_KEYS = [
    "policy_proprios",
    "next_policy_proprios",
    "raw_action_chunks",
    "executed_action_chunks",
]


METADATA_EQUAL_KEYS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference-episode",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate-episode",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


def load_episode(
    episode_dir: str,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    npz_path = osp.join(
        episode_dir,
        "trajectory.npz",
    )
    metadata_path = osp.join(
        episode_dir,
        "metadata.json",
    )

    if not osp.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    if not osp.isfile(metadata_path):
        raise FileNotFoundError(metadata_path)

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        arrays = {
            key: data[key]
            for key in data.files
        }

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return arrays, metadata


def max_abs_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.size == 0 and second.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                first.astype(np.float64)
                - second.astype(np.float64)
            )
        )
    )


def compare_metadata(
    reference: Dict,
    candidate: Dict,
) -> None:
    for key in METADATA_EQUAL_KEYS:
        reference_value = reference.get(key)
        candidate_value = candidate.get(key)

        if reference_value != candidate_value:
            raise AssertionError(
                f"Metadata mismatch for {key}: "
                f"reference={reference_value!r}, "
                f"candidate={candidate_value!r}"
            )

        print(
            f"[PASS] metadata.{key} matches."
        )


def compare_exact_arrays(
    reference: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    keys: Iterable[str],
) -> None:
    for key in keys:
        if key not in reference:
            raise KeyError(
                f"Reference rollout is missing {key}."
            )

        if key not in candidate:
            raise KeyError(
                f"Candidate rollout is missing {key}."
            )

        first = reference[key]
        second = candidate[key]

        if first.shape != second.shape:
            raise AssertionError(
                f"{key} shape mismatch: "
                f"reference={first.shape}, "
                f"candidate={second.shape}"
            )

        if not np.array_equal(first, second):
            difference = max_abs_difference(
                first,
                second,
            )

            raise AssertionError(
                f"{key} is not exactly equal. "
                f"max_abs_diff={difference}"
            )

        print(
            f"[PASS] {key} is exactly equal; "
            f"shape={first.shape}."
        )


def compare_float_arrays(
    reference: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    keys: Iterable[str],
    *,
    atol: float,
    rtol: float,
) -> None:
    for key in keys:
        if key not in reference:
            raise KeyError(
                f"Reference rollout is missing {key}."
            )

        if key not in candidate:
            raise KeyError(
                f"Candidate rollout is missing {key}."
            )

        first = reference[key]
        second = candidate[key]

        if first.shape != second.shape:
            raise AssertionError(
                f"{key} shape mismatch: "
                f"reference={first.shape}, "
                f"candidate={second.shape}"
            )

        difference = max_abs_difference(
            first,
            second,
        )

        if not np.allclose(
            first,
            second,
            atol=atol,
            rtol=rtol,
        ):
            raise AssertionError(
                f"{key} differs beyond tolerance. "
                f"max_abs_diff={difference}, "
                f"atol={atol}, rtol={rtol}"
            )

        print(
            f"[PASS] {key} matches; "
            f"shape={first.shape}, "
            f"max_abs_diff={difference:.10f}."
        )


def main() -> None:
    args = parse_args()

    reference_arrays, reference_metadata = (
        load_episode(args.reference_episode)
    )
    candidate_arrays, candidate_metadata = (
        load_episode(args.candidate_episode)
    )

    print(
        "========== Identity Gate Closed-Loop Regression =========="
    )
    print(
        "reference:",
        args.reference_episode,
    )
    print(
        "candidate:",
        args.candidate_episode,
    )

    compare_metadata(
        reference_metadata,
        candidate_metadata,
    )

    compare_exact_arrays(
        reference_arrays,
        candidate_arrays,
        EXACT_ARRAY_KEYS,
    )

    compare_float_arrays(
        reference_arrays,
        candidate_arrays,
        FLOAT_ARRAY_KEYS,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(
        "========== Identity closed-loop regression passed =========="
    )


if __name__ == "__main__":
    main()