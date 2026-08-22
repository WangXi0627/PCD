# wx:motivation-random mask
# contrast_policies/pizero_random_mask.py

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
from simpler_env.policies.pizero.pizero_model import PiZeroInference


class FixedRandomDimMask:
    """
    固定随机维度 mask。

    一个 mask_seed + keep_ratio 对应一个固定维度子集。
    整个 evaluation 中每次 forward 都使用同一个 mask，而不是每一步重新随机采样。
    """

    # wx:motivation-random mask 尺度对照实验
    VALID_MODES = {
        "mask",             # 现有实验：直接将随机维度置零
        "norm_preserve",    # 置零后恢复原特征L2范数
        "scale_only",       # 不置零，只缩放到与mask结果相同的L2范数
    }
    # wx:motivation-random mask 尺度对照实验

    def __init__(
        self,
        keep_ratio: float = 1.0,
        seed: int = 0,
        rescale: bool = True,
        # wx:motivation-random mask 尺度对照实验
        mode: str = "mask",
        eps: float = 1e-6,
        # wx:motivation-random mask 尺度对照实验
    ):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio should be in (0, 1], got {keep_ratio}")

        # wx:motivation-random mask 尺度对照实验
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mask mode: {mode}. "
                f"Available modes: {sorted(self.VALID_MODES)}"
            )
        # wx:motivation-random mask 尺度对照实验

        self.keep_ratio = float(keep_ratio)
        self.seed = int(seed)
        self.rescale = bool(rescale)
        # wx:motivation-random mask 尺度对照实验
        self.mode = str(mode)
        self.eps = float(eps)
        # wx:motivation-random mask 尺度对照实验

        # key: (dim, device_type, device_index, dtype)
        self._cache: Dict[Tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}

    # wx:motivation-random mask 尺度对照实验
    # def _make_mask(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    #     """
    #     用 CPU generator 生成随机数，避免不同 CUDA generator 行为差异。
    #     然后把 mask 移到特征所在 device。
    #     """
    #     generator = torch.Generator(device="cpu")
    #     generator.manual_seed(self.seed)

    #     rand = torch.rand(dim, generator=generator)
    #     mask = (rand < self.keep_ratio).to(dtype=dtype)

    #     if self.rescale:
    #         # inverted dropout style rescale:
    #         # keep_ratio=0.7 时，被保留维度乘 1/0.7，避免整体 feature scale 下降。
    #         mask = mask / self.keep_ratio

    #     return mask.to(device=device)
    # wx:motivation-random mask 尺度对照实验
    def _make_mask(
        self,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        rand = torch.rand(dim, generator=generator)
        mask = (rand < self.keep_ratio).to(dtype=dtype)

        return mask.to(device=device)
    # wx:motivation-random mask 尺度对照实验

    def get(self, h: torch.Tensor) -> torch.Tensor:
        dim = h.shape[-1]
        device = h.device
        dtype = h.dtype

        key = (dim, device.type, device.index, dtype)
        if key not in self._cache:
            self._cache[key] = self._make_mask(dim, device, dtype)

        return self._cache[key]

    # wx:motivation-random mask 尺度对照实验
    # def apply(self, h: torch.Tensor) -> torch.Tensor:
    #     if self.keep_ratio >= 1.0:
    #         return h

    #     mask = self.get(h)
    #     view_shape = [1] * h.ndim
    #     view_shape[-1] = h.shape[-1]
    #     return h * mask.view(*view_shape)
    # wx:motivation-random mask 尺度对照实验
    def apply(self, h: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0:
            return h

        mask = self.get(h)

        view_shape = [1] * h.ndim
        view_shape[-1] = h.shape[-1]
        mask = mask.view(*view_shape)

        # 现有的随机维度置零结果
        masked = h * mask

        # -----------------------------------------------------
        # Mode 1: 原始实验
        # -----------------------------------------------------
        if self.mode == "mask":
            if self.rescale:
                # inverted-dropout式缩放，保持随机mask的期望，
                # 但不是严格的L2范数保持。
                return masked / self.keep_ratio

            return masked

        # 使用float32计算范数，避免bf16/fp16下数值误差
        h_float = h.float()
        masked_float = masked.float()

        original_norm = torch.linalg.vector_norm(
            h_float,
            ord=2,
            dim=-1,
            keepdim=True,
        )

        masked_norm = torch.linalg.vector_norm(
            masked_float,
            ord=2,
            dim=-1,
            keepdim=True,
        )

        # -----------------------------------------------------
        # Mode 2: 删除相同维度，但恢复原始L2范数
        # -----------------------------------------------------
        if self.mode == "norm_preserve":
            scale = original_norm / masked_norm.clamp_min(self.eps)

            # 极端情况下如果所有维度都被mask，避免异常放大
            scale = torch.where(
                masked_norm > self.eps,
                scale,
                torch.ones_like(scale),
            )

            return masked * scale.to(dtype=h.dtype)

        # -----------------------------------------------------
        # Mode 3: 不删除维度，只产生与masked完全相同的范数
        # -----------------------------------------------------
        if self.mode == "scale_only":
            scale = masked_norm / original_norm.clamp_min(self.eps)

            scale = torch.where(
                original_norm > self.eps,
                scale,
                torch.ones_like(scale),
            )

            return h * scale.to(dtype=h.dtype)

        raise RuntimeError(f"Unhandled mask mode: {self.mode}")
    # wx:motivation-random mask 尺度对照实验


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


class PiZeroRandomMaskInference(PiZeroInference):
    """
    π0 动机实验 wrapper：
    - 冻结原始 π0；
    - 不用 contrast image；
    - 在内部 feature 上施加固定随机维度 mask；
    - 每个 mask_seed 对应图上的一个点。
    """

    def __init__(
        self,
        random_feature_mask: bool = False,
        mask_keep_ratio: float = 1.0,
        mask_seed: int = 0,
        mask_rescale: bool = True,
        mask_mode: str = "mask", # wx:motivation-random mask 尺度对照实验
        mask_target: str = "siglip_text_embedding",
        mask_verbose: bool = False,
        *args,
        **kwargs,
    ):
        # 这里会构造原始 PiZeroInference，包括 self.model / self.env_adapter 等。
        super().__init__(*args, **kwargs)

        self.random_feature_mask = bool(random_feature_mask)
        self.mask_keep_ratio = float(mask_keep_ratio)
        self.mask_seed = int(mask_seed)
        self.mask_rescale = bool(mask_rescale)
        self.mask_target = str(mask_target)
        self.mask_verbose = bool(mask_verbose)
        self._printed_mask_apply = False
        self.mask_mode = str(mask_mode) # wx:motivation-random mask 尺度对照实验

        self.dim_mask = FixedRandomDimMask(
            keep_ratio=self.mask_keep_ratio,
            seed=self.mask_seed,
            rescale=self.mask_rescale,
            mode=self.mask_mode, # wx:motivation-random mask 尺度对照实验
        )

        if self.mask_verbose:
            self._print_debug_info()

    def _print_debug_info(self) -> None:
        print("\n[PiZeroRandomMask] Model class:", type(self.model))
        print("[PiZeroRandomMask] random_feature_mask:", self.random_feature_mask)
        print("[PiZeroRandomMask] mask_keep_ratio:", self.mask_keep_ratio)
        print("[PiZeroRandomMask] mask_seed:", self.mask_seed)
        print("[PiZeroRandomMask] mask_rescale:", self.mask_rescale)
        print("[PiZeroRandomMask] mask_target:", self.mask_target)
        print("[PiZeroRandomMask] mask_mode:", self.mask_mode) # wx:motivation-random mask 尺度对照实验

        print("\n[PiZeroRandomMask] Candidate methods:")
        for name in dir(self.model):
            lname = name.lower()
            if any(k in lname for k in ["siglip", "embed", "text", "image", "infer", "action"]):
                attr = getattr(self.model, name)
                if callable(attr):
                    print("  method:", name)

        if hasattr(self.model, "named_modules"):
            print("\n[PiZeroRandomMask] Candidate modules:")
            for name, module in self.model.named_modules():
                lname = name.lower()
                if any(k in lname for k in ["siglip", "embed", "projector", "joint", "action"]):
                    print("  module:", name, "=>", type(module))
                    
    def _print_mask_apply(self, target_name: str, output: Any) -> None:
        if torch.is_tensor(output):
            print(
                "[PiZeroRandomMask] Applied random dim mask at "
                f"{target_name}: shape={tuple(output.shape)}, "
                f"keep_ratio={self.mask_keep_ratio}, "
                f"mask_seed={self.mask_seed}, "
                f"rescale={self.mask_rescale}"
            )
        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            print(
                "[PiZeroRandomMask] Applied random dim mask at "
                f"{target_name}: shape={tuple(output[0].shape)}, "
                f"keep_ratio={self.mask_keep_ratio}, "
                f"mask_seed={self.mask_seed}, "
                f"rescale={self.mask_rescale}"
            )
        else:
            print(
                "[PiZeroRandomMask] Warning: output is not tensor-like, "
                f"target={target_name}, type={type(output)}"
            )

    @contextmanager
    def random_mask_context(self):
        """
        在一次 action inference 调用期间临时启用 random feature mask。

        支持的 mask_target:
        - siglip_text_embedding
        - multi_modal_projector
        - multi_modal_projector.linear
        - vlm_layer_0 / 4 / 8 / 12 / 17
        - action_layer_0 / 3 / 6
        """
        if (not self.random_feature_mask) or self.mask_keep_ratio >= 1.0:
            yield
            return

        # 如果模型被 torch.compile 包装，真实模型通常在 _orig_mod 里。
        target_model = getattr(self.model, "_orig_mod", self.model)

        # =========================
        # Case 1: patch method
        # =========================
        if self.mask_target == "siglip_text_embedding":
            method_name = "_forward_siglip_and_text_embedding"

            if not hasattr(target_model, method_name):
                raise RuntimeError(
                    f"Cannot find method {method_name} on target_model={type(target_model)}. "
                    "Please run with mask_verbose=True and check candidate methods."
                )

            old_method = getattr(target_model, method_name)

            def wrapped_method(*args, **kwargs):
                output = old_method(*args, **kwargs)
                masked_output = apply_mask_to_output(output, self.dim_mask)

                if self.mask_verbose and not self._printed_mask_apply:
                    self._print_mask_apply(method_name, output)
                    self._printed_mask_apply = True

                return masked_output

            if self.mask_verbose:
                print(
                    f"[PiZeroRandomMask] Patch method: {method_name} "
                    f"on target_model={type(target_model)}"
                )

            setattr(target_model, method_name, wrapped_method)

            try:
                yield
            finally:
                setattr(target_model, method_name, old_method)

            return

        # =========================
        # Case 2: hook module
        # =========================
        module_name_map = {
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
                "Please run with mask_verbose=True and check candidate modules."
            )

        if self.mask_verbose:
            print(
                f"[PiZeroRandomMask] Hook module: {target_name} "
                f"=> {type(target_module)}"
            )

        def hook_fn(module, inputs, output):
            masked_output = apply_mask_to_output(output, self.dim_mask)

            if self.mask_verbose and not self._printed_mask_apply:
                self._print_mask_apply(target_name, output)
                self._printed_mask_apply = True

            return masked_output

        handle = target_module.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()
        
    @torch.no_grad()
    def step(self, image, instruction, proprio):
        """
        Random feature mask motivation experiment.

        Keep the same action-generation path as the original PiZero baseline:
        PiZeroInference.forward_actions() -> self.model.infer_action(...)
        """
        inputs = self.preprocess_inputs(image, instruction, proprio)

        with self.random_mask_context():
            raw_actions = super().forward_actions(inputs)

        actions = self.env_adapter.postprocess(
            raw_actions[0].float().cpu().numpy()
        )

        return raw_actions, actions