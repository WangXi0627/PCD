# wx:Dynamic gate v3

"""Hierarchical task -> episode -> query balanced sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


class TaskEpisodeQueryBatchSampler(Sampler[List[int]]):
    """Sample tasks, then episodes, then queries uniformly.

    This prevents tasks with longer trajectories, or episodes with more
    policy queries, from dominating the training distribution.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 0,
        avoid_duplicates_within_batch: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if not hasattr(dataset, "episodes") or not hasattr(dataset, "_query_index"):
            raise TypeError(
                "TaskEpisodeQueryBatchSampler expects RolloutQueryDataset."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.avoid_duplicates_within_batch = bool(avoid_duplicates_within_batch)
        self.epoch = 0

        nested = defaultdict(lambda: defaultdict(list))
        for dataset_index, (episode_position, _query_position) in enumerate(
            dataset._query_index
        ):
            episode = dataset.episodes[episode_position]
            task = str(episode["task"])
            episode_key = (
                task,
                int(episode["episode_id"]),
                str(episode["episode_dir"]),
            )
            nested[task][episode_key].append(dataset_index)

        if not nested:
            raise RuntimeError("Cannot build a sampler for an empty dataset.")

        self.tasks = sorted(nested.keys())
        self.episode_keys_by_task: Dict[str, List[tuple]] = {}
        self.query_indices_by_episode: Dict[tuple, List[int]] = {}

        for task in self.tasks:
            episode_keys = sorted(nested[task].keys())
            self.episode_keys_by_task[task] = episode_keys
            for episode_key in episode_keys:
                indices = list(nested[task][episode_key])
                if not indices:
                    raise RuntimeError(f"Episode {episode_key} has no queries.")
                self.query_indices_by_episode[episode_key] = indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _draw_index(self, rng: random.Random) -> int:
        task = rng.choice(self.tasks)
        episode_key = rng.choice(self.episode_keys_by_task[task])
        return rng.choice(self.query_indices_by_episode[episode_key])

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)

        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            used = set()

            for _slot in range(self.batch_size):
                candidate = self._draw_index(rng)
                if self.avoid_duplicates_within_batch:
                    for _attempt in range(32):
                        if candidate not in used:
                            break
                        candidate = self._draw_index(rng)
                batch.append(candidate)
                used.add(candidate)

            yield batch
