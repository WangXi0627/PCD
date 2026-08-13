# wx:Dynamic gate v2

#!/usr/bin/env python3
"""Build an episode-level manifest for dynamic-gate training."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import defaultdict


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_gating.split_utils import (  # noqa: E402
    assign_episode_splits,
    build_manifest,
    save_manifest,
    scan_rollout_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=str, required=True)
    parser.add_argument(
        "--policy-setup",
        choices=["google_robot", "widowx_bridge"],
        required=True,
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--validation-count-per-task", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = scan_rollout_episodes(
        rollout_root=args.rollout_root,
        policy_setup=args.policy_setup,
    )
    if not records:
        raise RuntimeError(
            f"No rollout episodes found for policy_setup={args.policy_setup!r} "
            f"under {args.rollout_root}."
        )

    records = assign_episode_splits(
        records,
        validation_count_per_task=args.validation_count_per_task,
        split_seed=args.split_seed,
    )

    manifest = build_manifest(
        rollout_root=args.rollout_root,
        policy_setup=args.policy_setup,
        records=records,
        split_seed=args.split_seed,
        validation_count_per_task=args.validation_count_per_task,
    )
    save_manifest(manifest, args.output)

    counts = defaultdict(lambda: defaultdict(int))
    queries = defaultdict(lambda: defaultdict(int))
    successes = defaultdict(lambda: defaultdict(int))

    for record in records:
        task = record["task"]
        split = record["split"]
        counts[task][split] += 1
        queries[task][split] += int(record["num_policy_queries"])
        successes[task][split] += int(bool(record["final_success"]))

    print("========== Dynamic Gate Manifest ==========")
    print(f"output: {osp.abspath(args.output)}")
    print(f"policy_setup: {args.policy_setup}")
    print(f"episodes: {len(records)}")
    print(f"queries: {sum(int(item['num_policy_queries']) for item in records)}")

    for task in sorted(counts):
        print(f"\n{task}")
        for split in ("train", "validation"):
            print(
                f"  {split}: episodes={counts[task][split]}, "
                f"queries={queries[task][split]}, "
                f"success_episodes={successes[task][split]}"
            )

    print("\n[OK] Manifest created.")


if __name__ == "__main__":
    main()
