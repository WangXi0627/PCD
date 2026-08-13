import hydra
import os.path as osp
import torch
from omegaconf import OmegaConf

# wx:Dynamic gate v3
import hashlib
# wx:Dynamic gate v3

import sys
sys.path.append(osp.join(osp.dirname(__file__), 'open_pi_zero'))
from src.model.vla.pizero import PiZero

from .. import setup_torch_seed


def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)


class PiZeroInference:
    def __init__(self,
                 cfg_dir,
                 checkpoint_path,
                 policy_setup="widowx_bridge",
                 flow_sampling='beta',
                 use_ddp=False,
                 use_naive=False,
                 use_torch_compile=False,
                 # wx:Dynamic gate v3
                 deterministic_action_noise=False,
                 action_noise_base_seed=0,
                 evaluation_task_name=None,
                 # wx:Dynamic gate v3
                 seed=0):
        self.use_naive = use_naive
        # wx:Dynamic gate v0
        self.use_torch_compile = bool(use_torch_compile)
        # wx:Dynamic gate v0
        
        # wx:Dynamic gate v3
        self.deterministic_action_noise = bool(deterministic_action_noise)
        self.action_noise_base_seed = int(action_noise_base_seed)
        self.evaluation_task_name = (None if evaluation_task_name is None else str(evaluation_task_name))
        self._current_episode_id = None
        self._fallback_query_index = 0
        # wx:Dynamic gate v3
        
        if policy_setup == "widowx_bridge":
            cfg = OmegaConf.load(osp.join(cfg_dir, 'bridge.yaml'))
            if flow_sampling == 'beta':
                checkpoint_path = osp.join(checkpoint_path, 'bridge_beta_step19296_2024-12-26_22-30_42.pt')
            elif flow_sampling == 'uniform':
                checkpoint_path = osp.join(checkpoint_path, 'bridge_uniform_step19296_2024-12-26_22-31_42.pt')
            else:
                raise ValueError(f"Invalid flow_sampling: {flow_sampling}")
            
        elif policy_setup == "google_robot":
            cfg = OmegaConf.load(osp.join(cfg_dir, 'fractal.yaml'))
            if flow_sampling == 'beta':
                checkpoint_path = osp.join(checkpoint_path, 'fractal_beta_step29576_2024-12-29_13-10_42.pt')
            elif flow_sampling == 'uniform':
                checkpoint_path = osp.join(checkpoint_path, 'fractal_uniform_step29576_2024-12-31_22-26_42.pt')
            else:
                raise ValueError(f"Invalid flow_sampling: {flow_sampling}")
        
        cfg.flow_sampling = flow_sampling
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda')
        self.model = PiZero(cfg, use_ddp=use_ddp)
        load_checkpoint(self.model, checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        
        if use_torch_compile:
            self.model = torch.compile(self.model, mode='default')
        self.model.eval()

        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)
        self.env_adapter.reset()
        
    # wx:Dynamic gate v3
    # def reset(self, instruction, seed=None):
    #     self.env_adapter.reset()
    #     if seed is not None:
    #         setup_torch_seed(seed)
    # wx:Dynamic gate v3
    def reset(self, instruction, seed=None):
        self.env_adapter.reset()
        self._current_episode_id = (None if seed is None else int(seed))
        self._fallback_query_index = 0
        if seed is not None:
            setup_torch_seed(seed)
            
    def build_deterministic_action_noise(
        self,
        *,
        task=None,
        episode_id=None,
        query_index=None,
    ):
        """
        Build deterministic PiZero initial action noise for one online query.

        The noise key is:
            task + episode_id + query_index + action_noise_base_seed

        This allows baseline and Dynamic Gate policies to use the same
        stochastic action initialization during online comparison.
        """
        if not self.deterministic_action_noise:
            return None
        resolved_task = (self.evaluation_task_name if task is None else str(task))
        resolved_episode_id = (self._current_episode_id if episode_id is None else int(episode_id))
        resolved_query_index = (self._fallback_query_index if query_index is None else int(query_index))

        if resolved_task is None:
            raise ValueError("deterministic_action_noise requires a task name.")
        if resolved_episode_id is None:
            raise ValueError("deterministic_action_noise requires episode_id.")

        token = (
            f"{self.action_noise_base_seed}|"
            f"{resolved_task}|"
            f"{resolved_episode_id}|"
            f"{resolved_query_index}"
        )
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = (int.from_bytes(digest[:8], byteorder="little", signed=False,) % (2**63 - 1))
        target_model = getattr(self.model, "_orig_mod", self.model,)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            (
                1,
                int(target_model.horizon_steps),
                int(target_model.action_dim),
            ),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )

        return noise.to(device=self.device, dtype=self.dtype,)
    # wx:Dynamic gate v3
    
    # wx:Dynamic gate v3
    # def step(self, image, instruction, proprio, *args, **kwargs):
    #     inputs = self.preprocess_inputs(image, instruction, proprio)
    #     raw_actions = self.forward_actions(inputs)
    #     actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
    #     return raw_actions, actions
    # wx:Dynamic gate v3
    def step(self, image, instruction, proprio, *args, **kwargs):
        episode_id = kwargs.pop("episode_id", self._current_episode_id,)
        query_index = kwargs.pop("query_index", self._fallback_query_index,)
        task = kwargs.pop("task", self.evaluation_task_name,)
        inputs = self.preprocess_inputs(image, instruction, proprio)
        initial_action_noise = (self.build_deterministic_action_noise(task=task, episode_id=episode_id, query_index=query_index,))
        raw_actions = self.forward_actions(inputs, initial_action_noise=initial_action_noise,)
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        self._fallback_query_index += 1
        return raw_actions, actions
    # wx:Dynamic gate v3

    def preprocess_inputs(self, image, instruction, proprio):
        inputs = self.env_adapter.preprocess(image, instruction, proprio)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = \
            (self.model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=self.dtype))
        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(causal_mask)
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(self.dtype),
        }

        if self.use_naive:
            inputs.update({"causal_mask": causal_mask})
        else:
            inputs.update({
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
            })
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    # wx:Dynamic gate v0
    # def forward_actions(self, inputs):
    #     with torch.inference_mode():
    #         if self.use_naive:
    #             actions = self.model.infer_action_naive(**inputs)
    #         else:
    #             actions = self.model.infer_action(**inputs)
    #     return actions
    # wx:Dynamic gate v0
    def _forward_actions_impl(
        self,
        inputs,
        *,
        visual_gate=None,
        initial_action_noise=None,
        return_aux=False,
    ):
        if visual_gate is not None and self.use_torch_compile:
            raise RuntimeError(
                "Stage-0 visual gate execution requires "
                "use_torch_compile=False. Passing an external nn.Module "
                "through the compiled PiZero method is intentionally "
                "disabled until the dynamic-gate path is verified."
            )

        model_inputs = dict(inputs)
        model_inputs.update(
            {
                "visual_gate": visual_gate,
                "initial_action_noise": initial_action_noise,
                "return_aux": return_aux,
            }
        )

        if self.use_naive:
            return self.model.infer_action_naive(**model_inputs)

        return self.model.infer_action(**model_inputs)

    def forward_actions(
        self,
        inputs,
        *,
        visual_gate=None,
        initial_action_noise=None,
        return_aux=False,
    ):
        """
        Standard inference path.

        No gradients are retained.
        """
        with torch.inference_mode():
            return self._forward_actions_impl(
                inputs,
                visual_gate=visual_gate,
                initial_action_noise=initial_action_noise,
                return_aux=return_aux,
            )

    def forward_actions_with_gate_grad(
        self,
        inputs,
        *,
        visual_gate,
        initial_action_noise=None,
        return_aux=False,
    ):
        """
        Gradient-enabled gate path.

        PiZero parameters remain frozen because __init__ already calls
        model.freeze_all_weights(). Gradients can flow through the frozen
        computation graph into visual_gate parameters.
        """
        if visual_gate is None:
            raise ValueError(
                "forward_actions_with_gate_grad requires visual_gate."
            )

        if self.use_torch_compile:
            raise RuntimeError(
                "Gate gradient verification requires "
                "use_torch_compile=False."
            )

        with torch.enable_grad():
            return self._forward_actions_impl(
                inputs,
                visual_gate=visual_gate,
                initial_action_noise=initial_action_noise,
                return_aux=return_aux,
            )
    # wx:Dynamic gate v0
