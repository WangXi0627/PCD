# 复现
PCD/contrast_utils/inpainters.py (改动)
PCD/contrast_utils/mask_predictors.py (改动)
PCD/simpler_env/policies/openvla/openvla_model.py (改动)
PCD/simpler_env/policies/pizero/open_pi_zero/config/eval/*  (改动)
PCD/properties.py (改动)
PCD/utils.py (改动)
PCD/scripts/inference/default/* (改动)

```
# 报错：ImportError: Encountered error: `No module named 'bitsandbytes'` when loading module 'open_pi_zero.src.model.paligemma.siglip.SiglipVisionModel'
python -m pip install -U bitsandbytes
# 报错：TypeError: initialize_config_module.__init__() got an unexpected keyword argument 'version_base'
pip install -U "hydra-core==1.3.2" "omegaconf==2.3.0"
```

```
# 报错：Vulkan is incompatible with your driver. You may not use the renderer to render, however, CPU resources will be still available.
cd /usr/lib/x86_64-linux-gnu
ln -sfn libGLX_nvidia.so.570.153.02 libGLX_nvidia.so.0
ln -sfn libEGL_nvidia.so.570.153.02 libEGL_nvidia.so.0
ln -sfn libGLESv1_CM_nvidia.so.570.153.02 libGLESv1_CM_nvidia.so.1
ln -sfn libGLESv2_nvidia.so.570.153.02 libGLESv2_nvidia.so.2
ln -sfn libnvoptix.so.570.153.02 libnvoptix.so.1
# 检查
vulkaninfo --summary | grep -E "deviceName|deviceType|driverName|vendorID"
```

# 自定义GPU
PCD/parallel_inference.py (改动)

# Motivation
PCD/contrast_policies/__init__.py (改动)
PCD/contrast_policies/pizero_random_mask.py (新增)
PCD/properties.py (改动)
PCD/scripts/inference/wx_motivation/* (新增)

# 集成GR00T-N1.6
## 让 PCD 环境能 import GR00T client
```
conda activate pcd
cd Isaac-GR00T
pip install -e . --no-deps
pip install pyzmq msgpack numpy

# 验证
cd PCD
python - <<'PY'
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="127.0.0.1", port=5555, timeout_ms=60000, strict=False)
print("ping:", client.ping())
print("modality:", client.get_modality_config())
PY
```

PCD/contrast_policies/groot_client.py (新增)
PCD/contrast_policies/__init__.py (修改)
PCD/properties.py (修改)

## H100仅有一个GPU，直接返回0
PCD/parallel_inference.py (修改)

## GR00T-N1.6 official-style inference runner inside PCD
PCD/parallel_inference_groot_official.py (新增)
PCD/scripts/inference/default/baseline_groot_google_inference.sh (新增)
PCD/scripts/inference/default/baseline_groot_widowx_inference.sh (新增)
PCD/scripts/inference/default/official_groot_google_inference.sh (新增)
PCD/scripts/inference/default/official_groot_widowx_inference.sh (新增)

## GR00T-N1.6 + PCD-style grounded_sam_tracking runner
PCD/parallel_inference_groot_pcd.py (新增)
PCD/scripts/inference/default/contrast_groot_google_inference.sh (新增)
PCD/scripts/inference/default/contrast_groot_widowx_inference.sh (新增)

```
# 报错：ImportError: Failed to import original PCD modules. Please run this script inside PCD_ROOT and make sure PCD_ROOT is in PYTHONPATH. Original error: ModuleNotFoundError("No module named 'omegaconf'")
cd Isaac-GR00T
~/.local/bin/uv pip install --python gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python omegaconf==2.3.0 hydra-core==1.3.2
~/.local/bin/uv pip install --python gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python albumentations==0.5.2 opencv-python-headless scikit-image pytorch-lightning kornia webdataset easydict joblib timm
~/.local/bin/uv pip install --python gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python scikit-learn pandas
~/.local/bin/uv pip install --python gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python jax==0.4.35 jaxlib==0.4.35 
~/.local/bin/uv pip install --python gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python iopath
# 验证
cd PCD
export PYTHONPATH="/xxx/PCD:${PYTHONPATH:-}"
/xxx/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python - <<'PY'
from omegaconf import OmegaConf
from contrast_utils.contrast_image_generator import ContrastImageGenerator
from contrast_policies.kde_contrast_decoding import ContrastDecoding
print("PCD contrast imports OK")
PY
```

PCD/third_party/inpaint_anything/lama/saicinpainting/training/modules/fake_fakes.py (修改)
PCD/third_party/inpaint_anything/lama/saicinpainting/training/trainers/__init__.py (修改)

```
# 报错：RuntimeError: PCD ContrastImageGenerator failed at env_idx=0, step_idx=0, instruction='pick coke can': ModuleNotFoundError("No module named 'sam2'")
cd PCD
~/.local/bin/uv pip install --python /xxx/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python --no-deps -e third_party/grounded_sam_2
```

```
# 报错：RuntimeError: PCD ContrastImageGenerator failed at env_idx=0, step_idx=0, instruction='put the eggplant in the yellow basket': ValueError("Instruction 'put eggplant in yellow basket' does not match any template")
PCD/contrast_utils/instruction_templates.py (修改)
```

```
# 报错：RuntimeError: PCD ContrastImageGenerator failed at env_idx=0, step_idx=2301, instruction='place sprite can into bottom drawer': ValueError("'sprite can' is not in list")
PCD/contrast_utils/mask_predictors.py (修改)
```

# Test-time adaptive mask selection
PCD/contrast_policies/pizero_adaptive_mask.py (新增)
PCD/properties.py (修改)
PCD/contrast_policies/__init__.py (修改)
PCD/scripts/inference/wx_adaptive/* (新增)

```
# 报错：OSError: [Errno 36] File name too long:
PCD/parallel_inference.py (修改)
```

# GR00T random feature mask
PCD/parallel_inference_groot_pcd.py (修改)
PCD/scripts/inference/wx_motivation/GR00T/* (新增)

# Test-time learnable feature mask v1.0
PCD/contrast_policies/pizero_learnable_mask.py (新增)
PCD/properties.py (修改)
PCD/contrast_policies/__init__.py (修改)
PCD/scripts/inference/wx_learnable/v1.0/* (新增)

```
# 报错：ModuleNotFoundError: No module named 'expecttest'
python -m pip install expecttest
```

```
# 报错：RuntimeError: not allowed to set torch.backends.cudnn flags after disable_global_flags; please use flags() context manager instead
PCD/simpler_env/policies/__init__.py (修改)
```

# Rollout collect
PCD/rollout_data/__init__.py (新增)
PCD/rollout_data/recorder.py (新增)
PCD/parallel_inference.py (修改)
PCD/scripts/check_rollout_dataset.py (新增)
PCD/.gitignore (修改)
PCD/scripts/inference/wx_collect_data/* (新增)
PCD/rollouts/* (rollouts存储位置)

# Dynamic gate
## v0：把动态Gate接入PiZero，并确保可微
PCD/feature_gating/__init__.py (新增)
PCD/feature_gating/dynamic_channel_gate.py (新增)
PCD/simpler_env/policies/pizero/open_pi_zero/src/model/vla/pizero_0722.py (备份)
PCD/simpler_env/policies/pizero/open_pi_zero/src/model/vla/pizero.py (修改)
PCD/simpler_env/policies/pizero/pizero_model_0722.py (备份)
PCD/simpler_env/policies/pizero/pizero_model.py (修改)
PCD/scripts/smoke_test_pizero_dynamic_gate.py (新增)
PCD/scripts/inference/wx_dynamic_gate/v0/* (新增)

## v1：把Gate封装成可部署的闭环推理策略
PCD/contrast_policies/pizero_dynamic_gate.py (新增)
PCD/properties.py (修改)
PCD/contrast_policies/__init__0722.py (备份)
PCD/contrast_policies/__init__.py (修改)
PCD/scripts/compare_rollout_episodes.py (新增)
PCD/scripts/inference/wx_dynamic_gate/v1/* (新增)

## v2：建立离线数据、固定噪声Teacher和训练基础设施
PCD/feature_gating/checkpoint.py (新增)
PCD/feature_gating/fixed_noise_teacher.py (新增)
PCD/feature_gating/rollout_dataset.py (新增)
PCD/feature_gating/split_utils.py (新增)
PCD/feature_gating/training_step.py (新增)
PCD/scripts/build_dynamic_gate_manifest.py (新增)
PCD/scripts/check_dynamic_gate_dataset.py (新增)
PCD/scripts/smoke_test_dynamic_gate_training.py (新增)
PCD/scripts/inference/wx_dynamic_gate/v2/* (新增)
PCD/checkpoints/dynamic_gate/* (ckps存储位置)

## v3：训练无标签V1 Gate，并完成离线与闭环效果验证
PCD/feature_gating/augmentations.py (新增)
PCD/feature_gating/samplers.py (新增)
PCD/feature_gating/teacher_cache.py (新增)
PCD/feature_gating/v1_losses.py (新增)
PCD/feature_gating/v1_training_step.py (新增)
PCD/scripts/cache_fixed_noise_teacher.py (新增)
PCD/scripts/evaluate_dynamic_gate_v1_offline.py (新增)
PCD/scripts/train_dynamic_gate_v1.py (新增)
PCD/scripts/inference/wx_dynamic_gate/v3/* (新增)
PCD/properties.py (修改)
PCD/simpler_env/policies/pizero/pizero_model.py (修改)
PCD/contrast_policies/pizero_dynamic_gate.py (修改)
PCD/parallel_inference.py (修改)
PCD/teacher_cache/dynamic_gate/* (cache存储位置)

```
# 报错：TypeError: PiZeroInference.__init__() got an unexpected keyword argument 'deterministic_action_noise'
PCD/parallel_inference.py (修改)

# 报错：CUDA out of memory
PCD/scripts/train_dynamic_gate_v1.py (修改)
```

## v3.5：加入去冗余损失，重写训练与评估脚本
PCD/feature_gating/channel_redundancy.py (新增)
PCD/feature_gating/redundancy_training_step.py (新增)
PCD/scripts/build_channel_redundancy_cache.py (新增)
PCD/scripts/evaluate_dynamic_gate_redundancy_offline.py (新增)
PCD/scripts/train_dynamic_gate_redundancy.py (新增)
PCD/scripts/inference/wx_dynamic_gate/v3.5/* (新增)