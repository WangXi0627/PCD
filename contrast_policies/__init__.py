# wx:motivation-random token mask
# # wx:motivation-random mask
# # def get_policy(policy, contrast, config):
# # wx:motivation-random mask
# def get_policy(policy, contrast, random_mask, config):
#     if contrast and random_mask:
#         raise ValueError("contrast and random mask cannot be enabled simultaneously.")
# # wx:motivation-random mask
# wx:motivation-random token mask
def get_policy(policy, contrast, random_mask, config, random_token_mask=False,):
    if sum(
        bool(x)
        for x in [contrast, random_mask, random_token_mask]
    ) > 1:
        raise ValueError(
            "contrast, channel mask and token mask "
            "cannot be enabled simultaneously."
        )
# wx:motivation-random token mask
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
        # wx:motivation-random mask
        # if not contrast:
        #     from simpler_env.policies.pizero.pizero_model import PiZeroInference
        #     policy = PiZeroInference(**config)
        # else:
        #     from .pizero_contrast import PiZeroContrastInference
        #     policy = PiZeroContrastInference(**config)
        # wx:motivation-random mask
        if contrast:
            from .pizero_contrast import PiZeroContrastInference
            policy = PiZeroContrastInference(**config)
        elif random_mask:
            from .pizero_random_mask import PiZeroRandomMaskInference
            policy = PiZeroRandomMaskInference(**config)
        # wx:motivation-random token mask
        elif random_token_mask:
            from .pizero_random_token_mask import PiZeroRandomTokenMaskInference
            policy = PiZeroRandomTokenMaskInference(**config)
        # wx:motivation-random token mask
        else:
            from simpler_env.policies.pizero.pizero_model import PiZeroInference
            policy = PiZeroInference(**config)
        # wx:motivation-random mask
    else:
        raise NotImplementedError()
    
    return policy
