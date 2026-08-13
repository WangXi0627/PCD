# wx:Dynamic gate v1
RANDOM_MASK_KEYS = [
    "random_feature_mask",
    "mask_keep_ratio",
    "mask_seed",
    "mask_rescale",
    "mask_target",
    "mask_verbose",
]

ADAPTIVE_MASK_KEYS = [
    "adaptive_feature_mask",
    "adaptive_mask_keep_ratio",
    "adaptive_mask_seeds",
    "adaptive_num_candidates",
    "adaptive_mask_rescale",
    "adaptive_mask_target",
    "adaptive_include_nomask",
    "adaptive_score_mode",
    "adaptive_consensus_weight",
    "adaptive_temporal_weight",
    "adaptive_norm_weight",
    "adaptive_verbose",
]

LEARNABLE_MASK_KEYS = [
    "learnable_feature_mask",
    "learnable_mask_target",
    "learnable_target_keep_ratio",
    "learnable_mask_temperature",
    "learnable_mask_rescale",
    "learnable_opt_steps",
    "learnable_lr",
    "learnable_anchor_weight",
    "learnable_keep_weight",
    "learnable_binary_weight",
    "learnable_l2_weight",
    "learnable_reset_each_episode",
    "learnable_hard_topk_eval",
    "learnable_verbose",
    "learnable_early_stop",
    "learnable_min_opt_steps",
    "learnable_loss_tol",
    "learnable_patience",
]

DYNAMIC_GATE_KEYS = [
    "dynamic_feature_gate",
    "dynamic_gate_mode",
    "dynamic_gate_checkpoint",
    "dynamic_gate_num_groups",
    "dynamic_gate_hidden_dim",
    "dynamic_gate_target_keep_ratio",
    "dynamic_gate_temperature",
    "dynamic_gate_rescale",
    "dynamic_gate_checkpoint_strict",
    "dynamic_gate_verbose",
    "dynamic_gate_log_every",
]

FEATURE_INTERVENTION_KEYS = {
    "random": RANDOM_MASK_KEYS,
    "adaptive": ADAPTIVE_MASK_KEYS,
    "learnable": LEARNABLE_MASK_KEYS,
    "dynamic_gate": DYNAMIC_GATE_KEYS,
}

FEATURE_INTERVENTION_SWITCHES = {
    "random": "random_feature_mask",
    "adaptive": "adaptive_feature_mask",
    "learnable": "learnable_feature_mask",
    "dynamic_gate": "dynamic_feature_gate",
}

def _get_active_feature_interventions(config):
    return [
        mode_name
        for mode_name, switch_name
        in FEATURE_INTERVENTION_SWITCHES.items()
        if bool(config.get(switch_name, False))
    ]

def _validate_feature_intervention_modes(
    config,
    *,
    contrast=False,
):
    """
    Enforce strict mutual exclusion between all PiZero feature
    intervention modes.
    """
    active_modes = _get_active_feature_interventions(
        config
    )

    if len(active_modes) > 1:
        raise ValueError(
            "PiZero feature intervention modes are mutually "
            "exclusive. Enabled modes: "
            f"{active_modes}"
        )

    if contrast and active_modes:
        raise ValueError(
            "PiZero feature intervention wrappers currently "
            "require contrast=False. "
            f"Enabled mode: {active_modes[0]}"
        )

    return active_modes

def _strip_feature_intervention_keys(
    config,
    *,
    keep_mode=None,
):
    """
    Remove configuration keys belonging to all feature intervention
    modes except keep_mode.

    keep_mode may be:
    - None
    - random
    - adaptive
    - learnable
    - dynamic_gate
    """
    if (
        keep_mode is not None
        and keep_mode not in FEATURE_INTERVENTION_KEYS
    ):
        raise ValueError(
            f"Unknown keep_mode={keep_mode!r}. "
            f"Available modes: "
            f"{list(FEATURE_INTERVENTION_KEYS)}"
        )

    cleaned = dict(config)

    for mode_name, keys in FEATURE_INTERVENTION_KEYS.items():
        if mode_name == keep_mode:
            continue

        for key in keys:
            cleaned.pop(key, None)

    return cleaned
# wx:Dynamic gate v1

def get_policy(policy, contrast, config):
    if policy == 'octo':
        if not contrast:
            from simpler_env.policies.octo.octo_model import OctoInference
            policy = OctoInference(**config)
        else:
            from .octo_contrast import OctoContrastInference
            policy = OctoContrastInference(**config)
    elif policy == 'openvla':
        if not contrast:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            policy = OpenVLAInference(**config)
        else:
            from .openvla_contrast import OpenVLAContrastInference
            policy = OpenVLAContrastInference(**config)
    # wx:Dynamic gate v1
    elif policy == "pizero":
        active_modes = _validate_feature_intervention_modes(config, contrast=contrast,)
        active_mode = (active_modes[0] if active_modes else None)
        if active_mode == "dynamic_gate":
            print("[get_policy] Using "
                  "PiZeroDynamicGateInference")
            from .pizero_dynamic_gate import PiZeroDynamicGateInference
            dynamic_config = (_strip_feature_intervention_keys(config, keep_mode="dynamic_gate",))
            policy = PiZeroDynamicGateInference(**dynamic_config)
        elif active_mode == "learnable":
            print("[get_policy] Using "
                  "PiZeroLearnableMaskInference")
            from .pizero_learnable_mask import PiZeroLearnableMaskInference
            learnable_config = (_strip_feature_intervention_keys(config, keep_mode="learnable",))
            policy = PiZeroLearnableMaskInference(**learnable_config)
        elif active_mode == "adaptive":
            print("[get_policy] Using "
                  "PiZeroAdaptiveMaskInference")
            from .pizero_adaptive_mask import PiZeroAdaptiveMaskInference
            adaptive_config = (_strip_feature_intervention_keys(config, keep_mode="adaptive",))
            policy = PiZeroAdaptiveMaskInference(**adaptive_config)
        elif active_mode == "random":
            print("[get_policy] Using "
                  "PiZeroRandomMaskInference")
            from .pizero_random_mask import PiZeroRandomMaskInference
            random_config = (_strip_feature_intervention_keys(config, keep_mode="random",))
            policy = PiZeroRandomMaskInference(**random_config)
        else:
            print("[get_policy] Using original "
                  "PiZeroInference / PiZeroContrastInference")
            base_config = (_strip_feature_intervention_keys(config, keep_mode=None,))
            if not contrast:
                from simpler_env.policies.pizero.pizero_model import PiZeroInference
                policy = PiZeroInference(**base_config)
            else:
                from .pizero_contrast import PiZeroContrastInference
                policy = PiZeroContrastInference(**base_config)
    # wx:Dynamic gate v1
    # wx:集成GR00T-N1.6
    elif policy == 'groot':
        if not contrast:
            from .groot_client import Gr00tClientInference
            policy = Gr00tClientInference(**config)
    # wx:集成GR00T-N1.6
    else:
        raise NotImplementedError()
    
    return policy
