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
    
    # wx:motivation
    # Random feature mask motivation experiment for pi0.
    # These keys must be here, otherwise --opts will not update them.
    random_feature_mask=False,
    mask_keep_ratio=1.0,
    mask_seed=0,
    mask_rescale=True,
    mask_target="siglip_text_embedding",
    mask_verbose=False,
    # wx:motivation
)

# wx:集成GR00T-N1.6
GROOT_CONFIG = dict(
    host="127.0.0.1",
    port=5555,
    timeout_ms=60000,
    action_key=None,
    exec_horizon=1,
    debug=False,

    # Reserved for stage 2.
    image_mask_mode="none",

    # Reserved for stage 3.
    feature_mask_enable=False,
    feature_mask_target="image_tokens",
    feature_mask_mode="dim",
    feature_mask_keep_ratio=1.0,
    feature_mask_seed=0,
)
# wx:集成GR00T-N1.6

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

def get_policy_config(policy, checkpoint, task, opts, contrast):
    # wx:集成GR00T-N1.6
    if policy == 'rt1':
        # config = RT1_CONFIG
        config = RT1_CONFIG.copy()
        config['saved_model_path'] = checkpoint
    elif policy == 'octo':
        # config = OCTO_CONFIG
        config = OCTO_CONFIG.copy()
        config['model_type'] = checkpoint
    elif policy == 'openvla':
        # config = OPENVLA_CONFIG
        config = OPENVLA_CONFIG.copy()
        config['saved_model_path'] = checkpoint
    elif policy == 'pizero':
        # config = OPEN_PIZERO_CONFIG
        config = OPEN_PIZERO_CONFIG.copy()
        config['checkpoint_path'] = checkpoint
    elif policy == 'groot':
        config = GROOT_CONFIG.copy()
        # checkpoint is only used for naming result directory in PCD.
        # The real GR00T checkpoint is loaded by the external GR00T server.
        config['checkpoint_name'] = checkpoint
    # wx:集成GR00T-N1.6
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
        # wx:集成GR00T-N1.6
        elif policy == 'groot':
            raise NotImplementedError("GR00T contrast is stage 2.")
        # wx:集成GR00T-N1.6
        else:
            raise NotImplementedError()
    
    # update opts
    for k, v in opts.items():
        if k in config:
            config[k] = v
    
    return config


def get_contrast_image_generator_config(opts):
    config = CONTRAST_IMAGE_CONFIG
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config
