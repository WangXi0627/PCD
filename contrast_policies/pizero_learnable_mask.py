# wx: # wx:Test-time learnable feature mask v1.0
# contrast_policies/pizero_learnable_mask.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpler_env.policies.pizero.pizero_model import PiZeroInference


def apply_mask_to_output(output: Any, mask_module: "LearnableDimMask", hard: bool = False) -> Any:
    """
    对 module/method 输出做 mask。
    支持 Tensor 或 tuple/list，其中只 mask 第一个 Tensor。
    """
    if torch.is_tensor(output):
        return mask_module.apply(output, hard=hard)

    if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
        return (mask_module.apply(output[0], hard=hard), *output[1:])

    if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
        return [mask_module.apply(output[0], hard=hard), *output[1:]]

    return output


class LearnableDimMask(nn.Module):
    """
    可学习 feature-dimension mask。

    对 h.shape = [B, N, D] 的特征，只学习长度为 D 的 mask。
    同一个 mask 会 broadcast 到所有 token 上。

    第一版：
    - pattern 可学习；
    - target_keep_ratio 固定；
    - 默认使用 soft mask；
    - 可选执行时 top-k hard mask。
    """

    def __init__(
        self,
        dim: int,
        target_keep_ratio: float = 0.75,
        temperature: float = 1.0,
        rescale: bool = True,
        init_to_keep_ratio: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if not (0.0 < target_keep_ratio <= 1.0):
            raise ValueError(
                f"target_keep_ratio should be in (0, 1], got {target_keep_ratio}"
            )

        self.dim = int(dim)
        self.target_keep_ratio = float(target_keep_ratio)
        self.temperature = float(temperature)
        self.rescale = bool(rescale)

        if init_to_keep_ratio:
            # 初始化时 sigmoid(logit) = target_keep_ratio。
            # 如果 rescale=True，那么 h * m / target_keep_ratio 初始约等于 h。
            p = torch.tensor(self.target_keep_ratio).clamp(1e-4, 1 - 1e-4)
            init_logit = torch.log(p / (1.0 - p)) * self.temperature
            init = torch.full((self.dim,), float(init_logit))
        else:
            init = torch.zeros(self.dim)

        if device is not None:
            init = init.to(device)

        self.mask_logits = nn.Parameter(init)

    def soft_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        logits = self.mask_logits.to(device=device)
        m = torch.sigmoid(logits / self.temperature)
        return m.to(dtype=dtype)

    def hard_topk_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        执行时可选 hard top-k mask。
        注意：这个 hard mask 不用于优化，只用于最终 forward。
        """
        logits = self.mask_logits.to(device=device)
        k = max(1, int(round(self.target_keep_ratio * self.dim)))
        topk_idx = torch.topk(logits, k=k, dim=0).indices

        m = torch.zeros_like(logits)
        m[topk_idx] = 1.0
        return m.to(dtype=dtype)

    def current_keep_ratio(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits / self.temperature).mean()

    def apply(self, h: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if self.target_keep_ratio >= 1.0:
            return h

        if hard:
            m = self.hard_topk_mask(dtype=h.dtype, device=h.device)
        else:
            m = self.soft_mask(dtype=h.dtype, device=h.device)

        if self.rescale:
            # 类似 random mask 的 inverted-dropout scaling。
            # 用 target_keep_ratio 而不是 mean(m)，更稳定。
            m = m / self.target_keep_ratio

        view_shape = [1] * h.ndim
        view_shape[-1] = h.shape[-1]
        return h * m.view(*view_shape)


class PiZeroLearnableMaskInference(PiZeroInference):
    """
    π0 test-time learnable feature mask。

    特点：
    - π0 主模型完全冻结；
    - 只学习 feature-dimension mask_logits；
    - 默认在 multi_modal_projector 输出处学习 mask；
    - 每次 policy query/action chunk 前，对当前 observation 优化几步 mask；
    - 使用 anchor loss 让 masked action 不要过度偏离原始 π0；
    - 使用 keep loss 固定平均 keep ratio。
    """

    def __init__(
        self,
        learnable_feature_mask: bool = False,
        learnable_mask_target: str = "multi_modal_projector",
        learnable_target_keep_ratio: float = 0.75,
        learnable_mask_temperature: float = 1.0,
        learnable_mask_rescale: bool = True,
        learnable_opt_steps: int = 5,
        learnable_lr: float = 0.1,
        learnable_anchor_weight: float = 1.0,
        learnable_keep_weight: float = 10.0,
        learnable_binary_weight: float = 0.0,
        learnable_l2_weight: float = 0.0,
        learnable_reset_each_episode: bool = True,
        learnable_hard_topk_eval: bool = False,
        learnable_verbose: bool = False,
        learnable_early_stop: bool = True,
        learnable_min_opt_steps: int = 2,
        learnable_loss_tol: float = 1e-4,
        learnable_patience: int = 2,

        # 兼容 properties.py 里已有 random/adaptive 参数，避免传给 PiZeroInference 报错
        random_feature_mask: bool = False,
        mask_keep_ratio: float = 1.0,
        mask_seed: int = 0,
        mask_rescale: bool = True,
        mask_target: str = "multi_modal_projector",
        mask_verbose: bool = False,

        adaptive_feature_mask: bool = False,
        adaptive_mask_keep_ratio: float = 0.9,
        adaptive_mask_seeds: str = "0,1,2,3,4,5,6,7,8,9",
        adaptive_num_candidates: int = 10,
        adaptive_mask_rescale: bool = True,
        adaptive_mask_target: str = "multi_modal_projector",
        adaptive_include_nomask: bool = False,
        adaptive_score_mode: str = "consensus",
        adaptive_consensus_weight: float = 1.0,
        adaptive_temporal_weight: float = 0.0,
        adaptive_norm_weight: float = 0.0,
        adaptive_verbose: bool = False,

        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.learnable_feature_mask = bool(learnable_feature_mask)
        self.learnable_mask_target = str(learnable_mask_target)
        self.learnable_target_keep_ratio = float(learnable_target_keep_ratio)
        self.learnable_mask_temperature = float(learnable_mask_temperature)
        self.learnable_mask_rescale = bool(learnable_mask_rescale)

        self.learnable_opt_steps = int(learnable_opt_steps)
        self.learnable_lr = float(learnable_lr)

        self.learnable_anchor_weight = float(learnable_anchor_weight)
        self.learnable_keep_weight = float(learnable_keep_weight)
        self.learnable_binary_weight = float(learnable_binary_weight)
        self.learnable_l2_weight = float(learnable_l2_weight)

        self.learnable_reset_each_episode = bool(learnable_reset_each_episode)
        self.learnable_hard_topk_eval = bool(learnable_hard_topk_eval)
        self.learnable_verbose = bool(learnable_verbose or mask_verbose)
        
        self.learnable_early_stop = bool(learnable_early_stop)
        self.learnable_min_opt_steps = int(learnable_min_opt_steps)
        self.learnable_loss_tol = float(learnable_loss_tol)
        self.learnable_patience = int(learnable_patience)

        self.mask_module: Optional[LearnableDimMask] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._printed_mask_apply = False
        self._step_idx = 0

        if self.learnable_verbose:
            self._print_debug_info()

    def reset(self, instruction, seed=None):
        super().reset(instruction, seed=seed)

        if self.learnable_reset_each_episode:
            self.mask_module = None
            self.optimizer = None
            self._step_idx = 0
            self._printed_mask_apply = False

    def _print_debug_info(self) -> None:
        print("\n[PiZeroLearnableMask] Model class:", type(self.model))
        print("[PiZeroLearnableMask] learnable_feature_mask:", self.learnable_feature_mask)
        print("[PiZeroLearnableMask] mask_target:", self.learnable_mask_target)
        print("[PiZeroLearnableMask] target_keep_ratio:", self.learnable_target_keep_ratio)
        print("[PiZeroLearnableMask] opt_steps:", self.learnable_opt_steps)
        print("[PiZeroLearnableMask] lr:", self.learnable_lr)
        print("[PiZeroLearnableMask] anchor_weight:", self.learnable_anchor_weight)
        print("[PiZeroLearnableMask] keep_weight:", self.learnable_keep_weight)
        print("[PiZeroLearnableMask] binary_weight:", self.learnable_binary_weight)
        print("[PiZeroLearnableMask] l2_weight:", self.learnable_l2_weight)
        print("[PiZeroLearnableMask] hard_topk_eval:", self.learnable_hard_topk_eval)

        target_model = getattr(self.model, "_orig_mod", self.model)
        if hasattr(target_model, "named_modules"):
            print("\n[PiZeroLearnableMask] Candidate modules:")
            for name, module in target_model.named_modules():
                lname = name.lower()
                if any(k in lname for k in ["siglip", "embed", "projector", "joint", "action"]):
                    print("  module:", name, "=>", type(module))

    def _get_target_module_name_map(self) -> Dict[str, str]:
        return {
            "multi_modal_projector": "multi_modal_projector",
            "multi_modal_projector.linear": "multi_modal_projector.linear",

            "vlm_layer_0": "joint_model.mixtures.vlm.layers.0",
            "vlm_layer_4": "joint_model.mixtures.vlm.layers.4",
            "vlm_layer_8": "joint_model.mixtures.vlm.layers.8",
            "vlm_layer_12": "joint_model.mixtures.vlm.layers.12",
            "vlm_layer_17": "joint_model.mixtures.vlm.layers.17",

            "action_layer_0": "joint_model.mixtures.action.layers.0",
            "action_layer_3": "joint_model.mixtures.action.layers.3",
            "action_layer_6": "joint_model.mixtures.action.layers.6",
        }

    def _ensure_mask_module(self, dim: int, device: torch.device) -> None:
        if self.mask_module is not None and self.mask_module.dim == dim:
            return

        self.mask_module = LearnableDimMask(
            dim=dim,
            target_keep_ratio=self.learnable_target_keep_ratio,
            temperature=self.learnable_mask_temperature,
            rescale=self.learnable_mask_rescale,
            init_to_keep_ratio=True,
            device=device,
        )

        self.optimizer = torch.optim.Adam(
            [self.mask_module.mask_logits],
            lr=self.learnable_lr,
        )

        if self.learnable_verbose:
            print(
                "[PiZeroLearnableMask] Created mask module:",
                f"dim={dim}, target_keep_ratio={self.learnable_target_keep_ratio},",
                f"init_keep={float(self.mask_module.current_keep_ratio().detach().cpu()):.4f}",
            )

    def _print_mask_apply(self, target_name: str, output: Any) -> None:
        if torch.is_tensor(output):
            shape = tuple(output.shape)
        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            shape = tuple(output[0].shape)
        else:
            shape = f"non_tensor_output_type={type(output)}"

        print(
            "[PiZeroLearnableMask] Applied learnable mask at "
            f"{target_name}: shape={shape}, "
            f"target_keep_ratio={self.learnable_target_keep_ratio}, "
            f"rescale={self.learnable_mask_rescale}"
        )

    @contextmanager
    def learnable_mask_context(self, hard: bool = False):
        """
        在一次 action inference 中启用 learnable mask。
        """
        if (
            not self.learnable_feature_mask
            or self.learnable_target_keep_ratio >= 1.0
        ):
            yield
            return

        target_model = getattr(self.model, "_orig_mod", self.model)

        # Case 1: patch method
        if self.learnable_mask_target == "siglip_text_embedding":
            method_name = "_forward_siglip_and_text_embedding"

            if not hasattr(target_model, method_name):
                raise RuntimeError(
                    f"Cannot find method {method_name} on target_model={type(target_model)}."
                )

            old_method = getattr(target_model, method_name)

            def wrapped_method(*args, **kwargs):
                output = old_method(*args, **kwargs)

                tensor = output
                if isinstance(output, (tuple, list)):
                    tensor = output[0]

                if not torch.is_tensor(tensor):
                    return output

                self._ensure_mask_module(
                    dim=tensor.shape[-1],
                    device=tensor.device,
                )

                masked_output = apply_mask_to_output(output, self.mask_module, hard=hard)

                if self.learnable_verbose and not self._printed_mask_apply:
                    self._print_mask_apply(method_name, output)
                    self._printed_mask_apply = True

                return masked_output

            setattr(target_model, method_name, wrapped_method)

            try:
                yield
            finally:
                setattr(target_model, method_name, old_method)

            return

        # Case 2: hook module
        module_name_map = self._get_target_module_name_map()

        if self.learnable_mask_target not in module_name_map:
            raise ValueError(
                f"Unknown learnable_mask_target={self.learnable_mask_target}. "
                f"Available targets: {['siglip_text_embedding'] + list(module_name_map.keys())}"
            )

        target_name = module_name_map[self.learnable_mask_target]
        modules = dict(target_model.named_modules())
        target_module = modules.get(target_name, None)

        if target_module is None:
            raise RuntimeError(
                f"Cannot find target module: {target_name}. "
                "Please run with learnable_verbose=True and check candidate modules."
            )

        def hook_fn(module, inputs, output):
            tensor = output
            if isinstance(output, (tuple, list)):
                tensor = output[0]

            if not torch.is_tensor(tensor):
                return output

            self._ensure_mask_module(
                dim=tensor.shape[-1],
                device=tensor.device,
            )

            masked_output = apply_mask_to_output(output, self.mask_module, hard=hard)

            if self.learnable_verbose and not self._printed_mask_apply:
                self._print_mask_apply(target_name, output)
                self._printed_mask_apply = True

            return masked_output

        handle = target_module.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    def _forward_actions_grad(self, inputs):
        """
        不能用 PiZeroInference.forward_actions，因为里面有 torch.inference_mode()。
        learnable mask 优化时需要保留 mask_logits 的梯度。
        """
        if self.use_naive:
            return self.model.infer_action_naive(**inputs)
        return self.model.infer_action(**inputs)

    def _get_rng_state(self):
        cpu_state = torch.get_rng_state()

        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state(self.device)

        return cpu_state, cuda_state

    def _set_rng_state(self, state) -> None:
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)

        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state, self.device)

    def _compute_mask_losses(self, masked_actions: torch.Tensor, base_actions: torch.Tensor):
        """
        masked_actions/base_actions: [B, horizon, action_dim]
        """
        anchor_loss = F.mse_loss(masked_actions.float(), base_actions.float())

        keep_loss = torch.zeros_like(anchor_loss)
        binary_loss = torch.zeros_like(anchor_loss)
        l2_loss = torch.zeros_like(anchor_loss)

        if self.mask_module is not None:
            keep_ratio = self.mask_module.current_keep_ratio()
            target = torch.tensor(
                self.learnable_target_keep_ratio,
                device=keep_ratio.device,
                dtype=keep_ratio.dtype,
            )
            keep_loss = (keep_ratio - target) ** 2

            m = torch.sigmoid(
                self.mask_module.mask_logits / self.learnable_mask_temperature
            )
            binary_loss = (m * (1.0 - m)).mean()
            l2_loss = (self.mask_module.mask_logits ** 2).mean()

        total_loss = (
            self.learnable_anchor_weight * anchor_loss
            + self.learnable_keep_weight * keep_loss
            + self.learnable_binary_weight * binary_loss
            + self.learnable_l2_weight * l2_loss
        )

        return total_loss, {
            "anchor_loss": anchor_loss.detach(),
            "keep_loss": keep_loss.detach(),
            "binary_loss": binary_loss.detach(),
            "l2_loss": l2_loss.detach(),
        }

    def _optimize_mask_for_current_inputs(self, inputs):
        """
        对当前 observation 做 test-time mask optimization。
        """
        if self.learnable_opt_steps <= 0:
            return None

        rng_state = self._get_rng_state()

        with torch.no_grad():
            self._set_rng_state(rng_state)
            base_actions = super().forward_actions(inputs).detach()

        best_loss = None
        bad_count = 0

        for opt_idx in range(self.learnable_opt_steps):
            self._set_rng_state(rng_state)

            with torch.enable_grad():
                with self.learnable_mask_context(hard=False):
                    masked_actions = self._forward_actions_grad(inputs)

                if self.optimizer is None:
                    raise RuntimeError(
                        "Mask optimizer is not initialized. "
                        "This usually means the mask hook was not triggered."
                    )

                loss, loss_dict = self._compute_mask_losses(masked_actions, base_actions)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            current_loss = float(loss.detach().cpu())

            if self.learnable_verbose:
                keep_ratio = (
                    float(self.mask_module.current_keep_ratio().detach().cpu())
                    if self.mask_module is not None
                    else -1.0
                )
                print(
                    f"[PiZeroLearnableMask] opt={opt_idx + 1}/{self.learnable_opt_steps} "
                    f"loss={current_loss:.6f} "
                    f"anchor={float(loss_dict['anchor_loss'].cpu()):.6f} "
                    f"keep={float(loss_dict['keep_loss'].cpu()):.6f} "
                    f"binary={float(loss_dict['binary_loss'].cpu()):.6f} "
                    f"keep_ratio={keep_ratio:.4f}"
                )

            # =========================
            # Early stopping
            # =========================
            if self.learnable_early_stop and (opt_idx + 1) >= self.learnable_min_opt_steps:
                if best_loss is None:
                    best_loss = current_loss
                    continue

                improvement = best_loss - current_loss

                if improvement > self.learnable_loss_tol:
                    best_loss = current_loss
                    bad_count = 0
                else:
                    bad_count += 1

                if bad_count >= self.learnable_patience:
                    if self.learnable_verbose:
                        print(
                            f"[PiZeroLearnableMask] early stop at opt={opt_idx + 1}, "
                            f"best_loss={best_loss:.6f}, current_loss={current_loss:.6f}"
                        )
                    break

        return rng_state

    def step(self, image, instruction, proprio):
        """
        每次 policy query:
        1. preprocess inputs;
        2. 对当前 observation 优化 learnable mask 几步；
        3. 用优化后的 mask 输出 action chunk；
        4. 返回给 parallel_inference.py 执行。
        """
        inputs = self.preprocess_inputs(image, instruction, proprio)

        if not self.learnable_feature_mask:
            raw_actions = super().forward_actions(inputs)
            actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
            return raw_actions, actions

        self._step_idx += 1

        rng_state = self._optimize_mask_for_current_inputs(inputs)

        with torch.no_grad():
            if rng_state is not None:
                self._set_rng_state(rng_state)
            with self.learnable_mask_context(hard=self.learnable_hard_topk_eval):
                raw_actions = super().forward_actions(inputs)

        actions = self.env_adapter.postprocess(
            raw_actions[0].float().cpu().numpy()
        )

        if self.learnable_verbose and self.mask_module is not None:
            keep_ratio = float(self.mask_module.current_keep_ratio().detach().cpu())
            print(
                f"[PiZeroLearnableMask] step={self._step_idx}, "
                f"effective_keep_ratio={keep_ratio:.4f}"
            )

        return raw_actions, actions