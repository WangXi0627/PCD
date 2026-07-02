# 复现
/media/hwx/Xixixi/code-vla/PCD/contrast_utils/inpainters.py (改动)
/media/hwx/Xixixi/code-vla/PCD/contrast_utils/mask_predictors.py (改动)
/media/hwx/Xixixi/code-vla/PCD/simpler_env/policies/openvla/openvla_model.py (改动)
/media/hwx/Xixixi/code-vla/PCD/simpler_env/policies/pizero/open_pi_zero/config/eval/*  (改动)
/media/hwx/Xixixi/code-vla/PCD/properties.py (改动)
/media/hwx/Xixixi/code-vla/PCD/utils.py (改动)
/media/hwx/Xixixi/code-vla/PCD/scripts/inference/default/* (改动)

# 自定义GPU
/media/hwx/Xixixi/code-vla/PCD/parallel_inference.py (改动)

# Motivation
/media/hwx/Xixixi/code-vla/PCD/contrast_policies/__init__.py (改动)
/media/hwx/Xixixi/code-vla/PCD/contrast_policies/pizero_random_mask.py (新增)
/media/hwx/Xixixi/code-vla/PCD/properties.py (改动)
/media/hwx/Xixixi/code-vla/PCD/scripts/inference/wx_motivation/* (新增)

# 集成GR00T-N1.6
## 让 PCD 环境能 import GR00T client
```
conda activate pcd
cd /media/hwx/Xixixi/code-vla/Isaac-GR00T
pip install -e . --no-deps
pip install pyzmq msgpack numpy
```