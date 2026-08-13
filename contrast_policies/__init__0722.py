# wx:Test-time learnable feature mask v1.0
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

    # 如果你后面加了 early stop，也放进来
    "learnable_early_stop",
    "learnable_min_opt_steps",
    "learnable_loss_tol",
    "learnable_patience",
]

def _pop_keys(config, keys):
    config = dict(config)
    for key in keys:
        config.pop(key, None)
    return config

def _pop_all_mask_keys(config):
    return _pop_keys(
        config,
        RANDOM_MASK_KEYS + ADAPTIVE_MASK_KEYS + LEARNABLE_MASK_KEYS,
    )

def _pop_adaptive_and_learnable_keys(config):
    return _pop_keys(
        config,
        ADAPTIVE_MASK_KEYS + LEARNABLE_MASK_KEYS,
    )

def _pop_random_and_learnable_keys(config):
    return _pop_keys(
        config,
        RANDOM_MASK_KEYS + LEARNABLE_MASK_KEYS,
    )

def _pop_random_and_adaptive_keys(config):
    return _pop_keys(
        config,
        RANDOM_MASK_KEYS + ADAPTIVE_MASK_KEYS,
    )
# wx:Test-time learnable feature mask v1.0

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
    elif policy == 'pizero':
        # wx:Test-time learnable feature mask v1.0
        if config.get("learnable_feature_mask", False):
            print("[get_policy] Using PiZeroLearnableMaskInference")
            from .pizero_learnable_mask import PiZeroLearnableMaskInference
            learnable_config = _pop_random_and_adaptive_keys(config)
            policy = PiZeroLearnableMaskInference(**learnable_config)
        # wx:Test-time learnable feature mask v1.0
        # wx:Test-time adaptive mask selection
        elif config.get("adaptive_feature_mask", False):
            print("[get_policy] Using PiZeroAdaptiveMaskInference")
            from .pizero_adaptive_mask import PiZeroAdaptiveMaskInference
            adaptive_config = _pop_random_and_learnable_keys(config)
            policy = PiZeroAdaptiveMaskInference(**adaptive_config)
        # wx:Test-time adaptive mask selection
        # wx: motivation
        elif config.get("random_feature_mask", False):
            print("[get_policy] Using PiZeroRandomMaskInference")
            from .pizero_random_mask import PiZeroRandomMaskInference
            random_config = _pop_adaptive_and_learnable_keys(config)
            policy = PiZeroRandomMaskInference(**random_config)
        else:
            print("[get_policy] Using original PiZeroInference / PiZeroContrastInference")
            config = _pop_all_mask_keys(config)
        # wx: motivation
            if not contrast:
                from simpler_env.policies.pizero.pizero_model import PiZeroInference
                policy = PiZeroInference(**config)
            else:
                from .pizero_contrast import PiZeroContrastInference
                policy = PiZeroContrastInference(**config)
    # wx:集成GR00T-N1.6
    elif policy == 'groot':
        if not contrast:
            from .groot_client import Gr00tClientInference
            policy = Gr00tClientInference(**config)
    # wx:集成GR00T-N1.6
    else:
        raise NotImplementedError()
    
    return policy
