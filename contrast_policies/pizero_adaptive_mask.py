# wx:Test-time adaptive mask selection
# contrast_policies/pizero_adaptive_mask.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from simpler_env.policies.pizero.pizero_model import PiZeroInference


class FixedRandomDimMask:
    """
    固定随机维度 mask。

    一个 seed + keep_ratio 对应一个固定维度子集。
    整个 evaluation 中每次 forward 都使用同一个 mask。
    """

    def __init__(
        self,
        keep_ratio: float = 1.0,
        seed: int = 0,
        rescale: bool = True,
    ):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio should be in (0, 1], got {keep_ratio}")

        self.keep_ratio = float(keep_ratio)
        self.seed = int(seed)
        self.rescale = bool(rescale)

        # key: (dim, device_type, device_index, dtype)
        self._cache: Dict[Tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}

    def _make_mask(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        rand = torch.rand(dim, generator=generator)
        mask = (rand < self.keep_ratio).to(dtype=dtype)

        if self.rescale:
            # inverted-dropout style rescale
            mask = mask / self.keep_ratio

        return mask.to(device=device)

    def get(self, h: torch.Tensor) -> torch.Tensor:
        dim = h.shape[-1]
        device = h.device
        dtype = h.dtype

        key = (dim, device.type, device.index, dtype)
        if key not in self._cache:
            self._cache[key] = self._make_mask(dim, device, dtype)

        return self._cache[key]

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0:
            return h

        mask = self.get(h)
        view_shape = [1] * h.ndim
        view_shape[-1] = h.shape[-1]
        return h * mask.view(*view_shape)


def apply_mask_to_output(output: Any, mask: FixedRandomDimMask) -> Any:
    """
    对 module/method 输出做 mask。
    支持 Tensor 或 tuple/list，其中只 mask 第一个 Tensor。
    """
    if torch.is_tensor(output):
        return mask.apply(output)

    if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
        return (mask.apply(output[0]), *output[1:])

    if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
        return [mask.apply(output[0]), *output[1:]]

    return output


def _parse_seed_list(
    seeds: Union[str, int, List[int], Tuple[int, ...], None],
    default_num_candidates: int = 10,
) -> List[int]:
    """
    支持：
      adaptive_mask_seeds = "0,1,2,3,4,5,6,7,8,9"
      adaptive_mask_seeds = [0,1,2]
      adaptive_mask_seeds = None
    """
    if seeds is None:
        return list(range(default_num_candidates))

    if isinstance(seeds, int):
        return list(range(seeds))

    if isinstance(seeds, (list, tuple)):
        return [int(x) for x in seeds]

    if isinstance(seeds, str):
        seeds = seeds.strip()
        if not seeds:
            return list(range(default_num_candidates))

        # 支持 "0,1,2" 或 "0 1 2"
        if "," in seeds:
            parts = seeds.split(",")
        else:
            parts = seeds.split()

        return [int(p.strip()) for p in parts if p.strip()]

    raise TypeError(f"Unsupported adaptive_mask_seeds type: {type(seeds)}")


class PiZeroAdaptiveMaskInference(PiZeroInference):
    """
    π0 test-time adaptive feature mask selection。

    核心思想：
    - 冻结原始 π0；
    - 不训练 mask，不反向传播；
    - 准备 K 个候选 random masks；
    - 每个 step 分别用这些 masks 前向预测 action；
    - 根据 action self-consistency / temporal smoothness / action norm 选择一个 mask；
    - 用被选中的 mask 对应 action 执行。

    默认评分：
        选择最接近候选 action 共识 median 的 mask。
    """

    def __init__(
        self,
        adaptive_feature_mask: bool = False,
        adaptive_mask_keep_ratio: Optional[float] = None,
        adaptive_mask_seeds: Union[str, int, List[int], Tuple[int, ...], None] = None,
        adaptive_num_candidates: int = 10,
        adaptive_mask_rescale: Optional[bool] = None,
        adaptive_mask_target: Optional[str] = None,
        adaptive_include_nomask: bool = False,
        adaptive_score_mode: str = "consensus",
        adaptive_consensus_weight: float = 1.0,
        adaptive_temporal_weight: float = 0.0,
        adaptive_norm_weight: float = 0.0,
        adaptive_verbose: bool = False,

        # 兼容 properties.py 里已有的 random mask keys，避免传给 PiZeroInference 报错
        random_feature_mask: bool = False,
        mask_keep_ratio: float = 1.0,
        mask_seed: int = 0,
        mask_rescale: bool = True,
        mask_target: str = "multi_modal_projector",
        mask_verbose: bool = False,

        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.adaptive_feature_mask = bool(adaptive_feature_mask)

        self.mask_keep_ratio = float(
            adaptive_mask_keep_ratio if adaptive_mask_keep_ratio is not None else mask_keep_ratio
        )
        self.mask_rescale = bool(
            adaptive_mask_rescale if adaptive_mask_rescale is not None else mask_rescale
        )
        self.mask_target = str(
            adaptive_mask_target if adaptive_mask_target is not None else mask_target
        )

        self.adaptive_include_nomask = bool(adaptive_include_nomask)
        self.adaptive_score_mode = str(adaptive_score_mode)
        self.adaptive_consensus_weight = float(adaptive_consensus_weight)
        self.adaptive_temporal_weight = float(adaptive_temporal_weight)
        self.adaptive_norm_weight = float(adaptive_norm_weight)
        self.adaptive_verbose = bool(adaptive_verbose or mask_verbose)

        self.seed_list = _parse_seed_list(
            adaptive_mask_seeds,
            default_num_candidates=int(adaptive_num_candidates),
        )

        self.candidate_masks: List[Tuple[str, Optional[FixedRandomDimMask]]] = []

        if self.adaptive_include_nomask:
            self.candidate_masks.append(("nomask", None))

        for seed in self.seed_list:
            self.candidate_masks.append(
                (
                    f"seed{seed}",
                    FixedRandomDimMask(
                        keep_ratio=self.mask_keep_ratio,
                        seed=seed,
                        rescale=self.mask_rescale,
                    ),
                )
            )

        if len(self.candidate_masks) == 0:
            raise ValueError("No candidate masks are provided.")

        self._printed_mask_apply = False
        self._prev_selected_flat: Optional[torch.Tensor] = None
        self._step_idx = 0

        if self.adaptive_verbose:
            self._print_debug_info()

    def _print_debug_info(self) -> None:
        print("\n[PiZeroAdaptiveMask] Model class:", type(self.model))
        print("[PiZeroAdaptiveMask] adaptive_feature_mask:", self.adaptive_feature_mask)
        print("[PiZeroAdaptiveMask] mask_keep_ratio:", self.mask_keep_ratio)
        print("[PiZeroAdaptiveMask] mask_rescale:", self.mask_rescale)
        print("[PiZeroAdaptiveMask] mask_target:", self.mask_target)
        print("[PiZeroAdaptiveMask] seed_list:", self.seed_list)
        print("[PiZeroAdaptiveMask] include_nomask:", self.adaptive_include_nomask)
        print("[PiZeroAdaptiveMask] score_mode:", self.adaptive_score_mode)
        print("[PiZeroAdaptiveMask] consensus_weight:", self.adaptive_consensus_weight)
        print("[PiZeroAdaptiveMask] temporal_weight:", self.adaptive_temporal_weight)
        print("[PiZeroAdaptiveMask] norm_weight:", self.adaptive_norm_weight)

        if hasattr(self.model, "named_modules"):
            print("\n[PiZeroAdaptiveMask] Candidate modules:")
            for name, module in self.model.named_modules():
                lname = name.lower()
                if any(k in lname for k in ["siglip", "embed", "projector", "joint", "action"]):
                    print("  module:", name, "=>", type(module))

    def _print_mask_apply(self, target_name: str, output: Any, label: str) -> None:
        if torch.is_tensor(output):
            shape = tuple(output.shape)
        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            shape = tuple(output[0].shape)
        else:
            shape = f"non_tensor_output_type={type(output)}"

        print(
            "[PiZeroAdaptiveMask] Applied candidate mask at "
            f"{target_name}: label={label}, shape={shape}, "
            f"keep_ratio={self.mask_keep_ratio}, rescale={self.mask_rescale}"
        )

    def _get_target_module_name_map(self) -> Dict[str, str]:
        return {
            # Visual / multimodal projection side
            "multi_modal_projector": "multi_modal_projector",
            "multi_modal_projector.linear": "multi_modal_projector.linear",

            # VLM mixture layers
            "vlm_layer_0": "joint_model.mixtures.vlm.layers.0",
            "vlm_layer_4": "joint_model.mixtures.vlm.layers.4",
            "vlm_layer_8": "joint_model.mixtures.vlm.layers.8",
            "vlm_layer_12": "joint_model.mixtures.vlm.layers.12",
            "vlm_layer_17": "joint_model.mixtures.vlm.layers.17",

            # Action mixture layers
            "action_layer_0": "joint_model.mixtures.action.layers.0",
            "action_layer_3": "joint_model.mixtures.action.layers.3",
            "action_layer_6": "joint_model.mixtures.action.layers.6",
        }

    @contextmanager
    def candidate_mask_context(
        self,
        candidate_label: str,
        candidate_mask: Optional[FixedRandomDimMask],
    ):
        """
        对一个候选 mask 临时启用 hook / patch。
        candidate_mask=None 表示 no-mask baseline candidate。
        """
        if (
            (not self.adaptive_feature_mask)
            or candidate_mask is None
            or self.mask_keep_ratio >= 1.0
        ):
            yield
            return

        target_model = getattr(self.model, "_orig_mod", self.model)

        # =========================
        # Case 1: patch method
        # =========================
        if self.mask_target == "siglip_text_embedding":
            method_name = "_forward_siglip_and_text_embedding"

            if not hasattr(target_model, method_name):
                raise RuntimeError(
                    f"Cannot find method {method_name} on target_model={type(target_model)}."
                )

            old_method = getattr(target_model, method_name)

            def wrapped_method(*args, **kwargs):
                output = old_method(*args, **kwargs)
                masked_output = apply_mask_to_output(output, candidate_mask)

                if self.adaptive_verbose and not self._printed_mask_apply:
                    self._print_mask_apply(method_name, output, candidate_label)
                    self._printed_mask_apply = True

                return masked_output

            setattr(target_model, method_name, wrapped_method)

            try:
                yield
            finally:
                setattr(target_model, method_name, old_method)

            return

        # =========================
        # Case 2: hook module
        # =========================
        module_name_map = self._get_target_module_name_map()

        if self.mask_target not in module_name_map:
            raise ValueError(
                f"Unknown mask_target={self.mask_target}. "
                f"Available targets: {['siglip_text_embedding'] + list(module_name_map.keys())}"
            )

        target_name = module_name_map[self.mask_target]
        modules = dict(target_model.named_modules())
        target_module = modules.get(target_name, None)

        if target_module is None:
            raise RuntimeError(
                f"Cannot find target module: {target_name}. "
                "Please run with adaptive_verbose=True and check candidate modules."
            )

        def hook_fn(module, inputs, output):
            masked_output = apply_mask_to_output(output, candidate_mask)

            if self.adaptive_verbose and not self._printed_mask_apply:
                self._print_mask_apply(target_name, output, candidate_label)
                self._printed_mask_apply = True

            return masked_output

        handle = target_module.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    def _flatten_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        raw_actions 通常 shape 是 [1, horizon, action_dim]。
        这里直接 flatten 整个 action chunk，用于候选 mask 打分。
        """
        return raw_actions[0].float().detach().flatten()

    def _score_candidates(
        self,
        candidate_flats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        返回每个候选 mask 的 score，越大越好。

        默认：
        - consensus loss：候选 action 越接近所有候选 action 的 median 越好；
        - temporal loss：候选 action 越接近上一时刻选中的 action 越好；
        - norm loss：候选 action 幅度不要过大。
        """
        flats = torch.stack(candidate_flats, dim=0)  # [K, D]

        # robust consensus
        consensus = flats.median(dim=0).values

        losses = torch.zeros(flats.shape[0], device=flats.device, dtype=flats.dtype)

        if self.adaptive_consensus_weight > 0:
            consensus_loss = ((flats - consensus.unsqueeze(0)) ** 2).mean(dim=1)
            losses = losses + self.adaptive_consensus_weight * consensus_loss

        if (
            self.adaptive_temporal_weight > 0
            and self._prev_selected_flat is not None
            and self._prev_selected_flat.shape == flats[0].shape
        ):
            prev = self._prev_selected_flat.to(device=flats.device, dtype=flats.dtype)
            temporal_loss = ((flats - prev.unsqueeze(0)) ** 2).mean(dim=1)
            losses = losses + self.adaptive_temporal_weight * temporal_loss

        if self.adaptive_norm_weight > 0:
            norm_loss = (flats ** 2).mean(dim=1)
            losses = losses + self.adaptive_norm_weight * norm_loss

        # score 越大越好
        scores = -losses
        return scores

    @torch.no_grad()
    def step(self, image, instruction, proprio):
        """
        每个 step：
        1. 对 K 个候选 masks 分别预测 action；
        2. 用 action self-consistency 打分；
        3. 选择分数最高的 action 执行。
        """
        inputs = self.preprocess_inputs(image, instruction, proprio)

        # adaptive_feature_mask=False 时退化为原始 PiZero baseline
        if not self.adaptive_feature_mask:
            raw_actions = super().forward_actions(inputs)
            actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
            return raw_actions, actions

        candidate_raw_actions: List[torch.Tensor] = []
        candidate_flats: List[torch.Tensor] = []
        candidate_labels: List[str] = []

        for label, candidate_mask in self.candidate_masks:
            with self.candidate_mask_context(label, candidate_mask):
                raw_actions = super().forward_actions(inputs)

            candidate_raw_actions.append(raw_actions)
            candidate_flats.append(self._flatten_action(raw_actions))
            candidate_labels.append(label)

        scores = self._score_candidates(candidate_flats)
        best_idx = int(torch.argmax(scores).item())

        selected_raw_actions = candidate_raw_actions[best_idx]
        selected_flat = candidate_flats[best_idx]

        self._prev_selected_flat = selected_flat.detach().cpu()
        self._step_idx += 1

        if self.adaptive_verbose:
            score_list = [float(s.detach().cpu()) for s in scores]
            print(
                "[PiZeroAdaptiveMask] step=",
                self._step_idx,
                "selected=",
                candidate_labels[best_idx],
                "scores=",
                {
                    candidate_labels[i]: round(score_list[i], 6)
                    for i in range(len(candidate_labels))
                },
            )

        actions = self.env_adapter.postprocess(
            selected_raw_actions[0].float().cpu().numpy()
        )

        return selected_raw_actions, actions