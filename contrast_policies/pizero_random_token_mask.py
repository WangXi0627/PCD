# wx:motivation-random token mask

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
from simpler_env.policies.pizero.pizero_model import PiZeroInference


class FixedRandomTokenMask:
    def __init__(
        self,
        keep_ratio: float = 1.0,
        seed: int = 0,
        mode: str = "mask",
        eps: float = 1e-6,
    ):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(
                f"keep_ratio should be in (0, 1], got {keep_ratio}"
            )

        available_modes = {
            "mask",
            "norm_preserve",
            "scale_only",
        }
        if mode not in available_modes:
            raise ValueError(
                f"Unknown token mask mode={mode}. "
                f"Available modes: {sorted(available_modes)}"
            )

        self.keep_ratio = float(keep_ratio)
        self.seed = int(seed)
        self.mode = str(mode)
        self.eps = float(eps)

        self._cache: Dict[
            Tuple[int, str, Optional[int], torch.dtype],
            torch.Tensor,
        ] = {}

    def _make_mask(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        num_keep = int(round(num_tokens * self.keep_ratio))
        num_keep = max(1, min(num_tokens, num_keep))

        indices = torch.randperm(
            num_tokens,
            generator=generator,
        )[:num_keep]

        mask = torch.zeros(num_tokens, dtype=dtype)
        mask[indices] = 1.0

        return mask.to(device=device)

    def get(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim < 3:
            raise RuntimeError(
                "Token mask expects output shaped like [B, T, D], "
                f"but got {tuple(h.shape)}"
            )

        num_tokens = h.shape[-2]
        device = h.device
        dtype = h.dtype

        key = (
            num_tokens,
            device.type,
            device.index,
            dtype,
        )

        if key not in self._cache:
            self._cache[key] = self._make_mask(
                num_tokens,
                device,
                dtype,
            )

        return self._cache[key]

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0:
            return h

        if h.ndim != 3:
            raise RuntimeError(
                "Token mask expects [B, T, D], "
                f"but got shape={tuple(h.shape)}"
            )

        token_mask = self.get(h)

        # [T] -> [1, T, 1]
        view_shape = [1] * h.ndim
        view_shape[-2] = h.shape[-2]
        token_mask = token_mask.view(*view_shape)

        # 不进行任何补偿的 token zero-masking
        masked = h * token_mask

        if self.mode == "mask":
            return masked

        # 使用 float32 计算范数，避免 fp16/bf16 数值误差。
        # 对每个 batch 样本，联合 T 和 D 维计算 Frobenius norm。
        original_norm = torch.linalg.vector_norm(
            h.float(),
            dim=(-2, -1),
            keepdim=True,
        )

        masked_norm = torch.linalg.vector_norm(
            masked.float(),
            dim=(-2, -1),
            keepdim=True,
        )

        if self.mode == "norm_preserve":
            # 删除相同 token，但恢复到原始全局范数
            scale = original_norm / masked_norm.clamp_min(self.eps)
            return masked * scale.to(dtype=h.dtype)

        if self.mode == "scale_only":
            # 不删除任何 token，仅模拟普通 mask 带来的尺度下降
            scale = masked_norm / original_norm.clamp_min(self.eps)
            return h * scale.to(dtype=h.dtype)

        raise RuntimeError(f"Unexpected token mask mode: {self.mode}")


def apply_token_mask_to_output(
    output: Any,
    mask: FixedRandomTokenMask,
) -> Any:
    if torch.is_tensor(output):
        return mask.apply(output)

    if (
        isinstance(output, tuple)
        and len(output) > 0
        and torch.is_tensor(output[0])
    ):
        return (mask.apply(output[0]), *output[1:])

    if (
        isinstance(output, list)
        and len(output) > 0
        and torch.is_tensor(output[0])
    ):
        return [mask.apply(output[0]), *output[1:]]

    return output


class PiZeroRandomTokenMaskInference(PiZeroInference):
    """
    π0 random visual-token zero-masking wrapper.
    """

    def __init__(
        self,
        random_token_mask: bool = False,
        token_mask_keep_ratio: float = 1.0,
        token_mask_seed: int = 0,
        token_mask_mode: str = "mask",
        token_mask_eps: float = 1e-6,
        token_mask_target: str = "multi_modal_projector",
        token_mask_verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.random_token_mask = bool(random_token_mask)
        self.token_mask_keep_ratio = float(token_mask_keep_ratio)
        self.token_mask_seed = int(token_mask_seed)
        self.token_mask_mode = str(token_mask_mode)
        self.token_mask_eps = float(token_mask_eps)
        self.token_mask_target = str(token_mask_target)
        self.token_mask_verbose = bool(token_mask_verbose)

        self._printed_token_mask_apply = False

        self.token_mask = FixedRandomTokenMask(
            keep_ratio=self.token_mask_keep_ratio,
            seed=self.token_mask_seed,
            mode=self.token_mask_mode,
            eps=self.token_mask_eps,
        )

    def _print_token_mask_apply(
        self,
        target_name: str,
        output: Any,
    ) -> None:
        h = output
        if isinstance(output, (tuple, list)):
            h = output[0]

        if not torch.is_tensor(h):
            print(
                "[PiZeroRandomTokenMask] Warning: "
                f"target={target_name}, output type={type(output)}"
            )
            return

        num_tokens = h.shape[-2]
        token_mask = self.token_mask.get(h)
        num_keep = int((token_mask != 0).sum().item())

        print(
            "[PiZeroRandomTokenMask] Applied token mask at "
            f"{target_name}: shape={tuple(h.shape)}, "
            f"num_tokens={num_tokens}, "
            f"num_keep={num_keep}, "
            f"actual_keep_ratio={num_keep / num_tokens:.6f}, "
            f"seed={self.token_mask_seed}, "
            f"mode={self.token_mask_mode}"
        )

    @contextmanager
    def random_token_mask_context(self):
        if (
            not self.random_token_mask
            or self.token_mask_keep_ratio >= 1.0
        ):
            yield
            return

        target_model = getattr(
            self.model,
            "_orig_mod",
            self.model,
        )

        # 首轮实验只允许视觉 projector，防止误伤文本 token。
        module_name_map = {
            "multi_modal_projector": "multi_modal_projector",
            "multi_modal_projector.linear":
                "multi_modal_projector.linear",
        }

        if self.token_mask_target not in module_name_map:
            raise ValueError(
                f"Unknown token_mask_target={self.token_mask_target}. "
                f"Available targets: {list(module_name_map.keys())}"
            )

        target_name = module_name_map[self.token_mask_target]
        modules = dict(target_model.named_modules())
        target_module = modules.get(target_name)

        if target_module is None:
            raise RuntimeError(
                f"Cannot find target module: {target_name}"
            )

        def hook_fn(module, inputs, output):
            masked_output = apply_token_mask_to_output(
                output,
                self.token_mask,
            )

            if (
                self.token_mask_verbose
                and not self._printed_token_mask_apply
            ):
                self._print_token_mask_apply(
                    target_name,
                    output,
                )
                self._printed_token_mask_apply = True

            return masked_output

        handle = target_module.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def step(self, image, instruction, proprio):
        inputs = self.preprocess_inputs(
            image,
            instruction,
            proprio,
        )

        with self.random_token_mask_context():
            raw_actions = super().forward_actions(inputs)

        actions = self.env_adapter.postprocess(
            raw_actions[0].float().cpu().numpy()
        )

        return raw_actions, actions