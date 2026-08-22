# 复现
PCD/contrast_utils/inpainters.py: 模型加载路径**改动**
PCD/contrast_utils/mask_predictors.py: 模型加载路径**改动**
PCD/simpler_env/policies/openvla/openvla_model.py: opnevla bug修复 **改动**
PCD/simpler_env/policies/pizero/open_pi_zero/config/eval/*: pi0 模型加载路径**改动**
PCD/properties.py: pi0 cfg路径**改动**
PCD/scripts/wx-inference/default/*: octo pi0 base&contrast运行脚本**新增**

```
报错: 
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.func.launch' failed: Failed to load PTX text as a module: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid; current tracing scope: fusion; current profiling annotation: XlaModule:#hlo_module=jit__threefry_seed program_id=1#. 
解决:
```

```
报错:
"google_robot_close_drawer" "google_robot_open_drawer" "google_robot_place_apple_in_closed_top_drawer" 三个drawer相关任务无法运行
解决:
PCD/third_party/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/open_drawer_in_scene.py: ret["shader_dir"] **改动**
```

# 自定义GPU
PCD/parallel_inference.py: **改动**

# 自定义结果文件夹名
PCD/parallel_inference.py: **改动**

# 自定义是否保存gif
PCD/parallel_inference.py: **改动**

# motivation-random mask
PCD/contrast_policies/__init__.py: random mask **改动**
PCD/contrast_policies/pizero_random_mask.py: random mask **新增**
PCD/parallel_inference.py: random mask **改动**
PCD/properties.py: random mask **改动**
PCD/scripts/wx-inference/motivation/random_pizero_inference*.sh: random mask 运行脚本 **新增**
PCD/scripts/wx-inference/motivation/collect_success_rate.py: random mask 结果统计脚本 **新增**
PCD/scripts/wx-inference/motivation/random_pizero_inference_baseline.sh: baseline 对比脚本 **新增**
PCD/scripts/wx-inference/motivation/collect_success_rate.py: pip install openpyxl

# motivation-random mask 尺度对照实验
PCD/contrast_policies/pizero_random_mask.py: 加入 mask mode **改动**
PCD/properties.py: 加入 mask mode 控制参数 **改动**
PCD/scripts/wx-inference/motivation/random_pizero_inference_scale_control*.sh: random mask 尺度对照实验运行脚本 **新增**

# motivation-random token mask
PCD/contrast_policies/__init__.py: random token mask **改动**
PCD/contrast_policies/pizero_random_token_mask.py: random token mask **新增**
PCD/parallel_inference.py: random token mask **改动**
PCD/properties.py: random token mask **改动**
PCD/scripts/wx-inference/motivation/random_token_pizero_inference*.sh: random token mask 运行脚本 **新增**

# 专家数据准备
见 /data/Xixixi/Dataset/tensorflow_datasets/README-wx.md