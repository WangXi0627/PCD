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
python - <<'PY'
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="127.0.0.1", port=5555, timeout_ms=60000, strict=False)
print("ping:", client.ping())
print("modality:", client.get_modality_config())
PY
```