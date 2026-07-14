# wx: motivation
def _pop_random_mask_keys(config):
    config = dict(config)
    random_mask_keys = [
        "random_feature_mask",
        "mask_keep_ratio",
        "mask_seed",
        "mask_rescale",
        "mask_target",
        "mask_verbose",
    ]
    for key in random_mask_keys:
        config.pop(key, None)
    return config
# wx: motivation

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
        # wx: motivation
        if config.get("random_feature_mask", False):
            from .pizero_random_mask import PiZeroRandomMaskInference
            policy = PiZeroRandomMaskInference(**config)
        else:
            config = _pop_random_mask_keys(config)
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
