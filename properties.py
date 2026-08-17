RT1_CONFIG = dict(
    saved_model_path="pretrained/rt_1_x",
    lang_embed_model_path="https://tfhub.dev/google/universal-sentence-encoder-large/5",
    image_width=320,
    image_height=256,
    action_scale=1.0,
    policy_setup="google_robot",
)
        
OCTO_CONFIG = dict(
    model=None,
    dataset_id=None,
    model_type='octo-base',
    model_step=None,
    policy_setup='widowx_bridge',
    horizon=2,
    pred_action_horizon=4,
    exec_horizon=1,
    image_size=256,
    action_scale=1.0,
    init_rng=0,
)

OPENVLA_CONFIG = dict(
    saved_model_path='openvla-7b',
    unnorm_key=None,
    policy_setup='google_robot',
    horizon=1,
    pred_action_horizon=1,
    exec_horizon=1,
    image_size=[224, 224],
    action_scale=1.0,
)

OPEN_PIZERO_CONFIG = dict(
    # wx:复现
    # cfg_dir='open_pi_zero/config/eval',
    cfg_dir='./simpler_env/policies/pizero/open_pi_zero/config/eval',
    # wx:复现
    use_ddp=False,
    use_naive=False,
    use_torch_compile=True,
)

CONTRAST_IMAGE_CONFIG = dict(
    camera_name=None,
    by="gt",
    inpaint_mode="lama",
    color="auto",
    sigma=5,
    version=2,
    get_all_parts=False,
)

CONTRAST_OCTO_CONFIG = dict(
    alpha=0.2,
    num_repeats=24,
    bandwidth_factor=2.0,
    keep_threshold=0.5,
)

CONTRAST_OPENVLA_CONFIG = dict(
    alpha=0.2,
)

CONTRAST_OPEN_PIZERO_CONFIG = dict(
    alpha=0.2,
    num_repeats=20,
    bandwidth_factor=1.0,
    keep_threshold=0.5,
)

# wx:motivation-random mask
RANDOM_OPEN_PIZERO_CONFIG = dict(
    random_feature_mask=True,
    mask_keep_ratio=1.0,
    mask_seed=0,
    mask_rescale=False,
    mask_target="multi_modal_projector",
    mask_verbose=False,
)
# wx:motivation-random mask

# wx:motivation-random mask
# def get_policy_config(policy, checkpoint, task, opts, contrast):
def get_policy_config(policy, checkpoint, task, opts, contrast, random_mask):
    if contrast and random_mask:
        raise ValueError(
            "contrast and random mask cannot be enabled simultaneously."
        )
# wx:motivation-random mask
    if policy == 'rt1':
        config = RT1_CONFIG.copy()
        config['saved_model_path'] = checkpoint
    elif policy == 'octo':
        config = OCTO_CONFIG.copy()
        config['model_type'] = checkpoint
    elif policy == 'openvla':
        config = OPENVLA_CONFIG.copy()
        config['saved_model_path'] = checkpoint
    elif policy == 'pizero':
        config = OPEN_PIZERO_CONFIG.copy()
        config['checkpoint_path'] = checkpoint
    else:
        raise NotImplementedError()
    
    # select policy setup based on task
    if task.startswith('google_robot'):
        config['policy_setup'] = 'google_robot'
    elif task.startswith('widowx'):
        config['policy_setup'] = 'widowx_bridge'
    else:
        raise NotImplementedError

    # update config if contrast policy is used
    if contrast:
        from properties import CONTRAST_OCTO_CONFIG, CONTRAST_OPENVLA_CONFIG
        if policy == 'octo':
            config.update(CONTRAST_OCTO_CONFIG)
        elif policy == 'openvla':
            config.update(CONTRAST_OPENVLA_CONFIG)
        elif policy == 'pizero':
            config.update(CONTRAST_OPEN_PIZERO_CONFIG)
        else:
            raise NotImplementedError()

    # wx:motivation-random mask
    elif random_mask:
        if policy == 'pizero':
            config.update(RANDOM_OPEN_PIZERO_CONFIG)
        else:
            raise NotImplementedError()
    # wx:motivation-random mask
    
    # update opts
    for k, v in opts.items():
        if k in config:
            config[k] = v

    # wx:motivation-random mask
    # 动态 hook/method patch 与 torch.compile 不一定兼容；
    # 放在 opts 后面，避免被 --opts 重新设置为 True
    if random_mask:
        config["use_torch_compile"] = False
    # wx:motivation-random mask
    
    return config


def get_contrast_image_generator_config(opts):
    config = CONTRAST_IMAGE_CONFIG.copy()
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config
