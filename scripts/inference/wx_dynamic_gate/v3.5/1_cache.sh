# 构建冗余维度缓存

mkdir -p redundancy_cache/dynamic_gate/google_robot

export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD
CUDA_VISIBLE_DEVICES=1 \
python scripts/build_channel_redundancy_cache.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir redundancy_cache/dynamic_gate/google_robot/train_mean_top16_corr065 \
  --policy-setup google_robot \
  --split train \
  --flow-sampling beta \
  --batch-size 2 \
  --base-noise-seed 0 \
  --pooling mean \
  --top-k-per-channel 16 \
  --min-abs-correlation 0.65 \
  --weight-power 2.0 \
  --correlation-device cpu

# 检查
# coverage_ratio >= 0.60; degree_max没有异常大; 最高度数没有集中到少数channel
# covered channels = 1500+; coverage = 73%+; degree median = 1或2; degree max < 30～50
python - <<'PY'
from pathlib import Path

import numpy as np

cache_dir = Path(
    "redundancy_cache/dynamic_gate/google_robot/"
    "train_mean_top16_corr065"
)

feature_dim = 2048

channel_i = np.load(cache_dir / "channel_i.npy")
channel_j = np.load(cache_dir / "channel_j.npy")
weights = np.load(cache_dir / "weights.npy")

if channel_i.shape != channel_j.shape:
    raise ValueError("channel_i and channel_j shape mismatch")

if channel_i.shape != weights.shape:
    raise ValueError("pair indices and weights shape mismatch")

degree = np.zeros(feature_dim, dtype=np.int64)
np.add.at(degree, channel_i, 1)
np.add.at(degree, channel_j, 1)

covered = degree > 0
num_covered = int(covered.sum())
coverage_ratio = num_covered / feature_dim

print("========== Redundancy Cache Diagnostics ==========")
print(f"num_pairs              : {len(channel_i)}")
print(f"num_covered_channels   : {num_covered}")
print(f"coverage_ratio         : {coverage_ratio:.4f}")
print(f"isolated_channels      : {(degree == 0).sum()}")
print(f"degree_mean            : {degree.mean():.4f}")
print(f"degree_median          : {np.median(degree):.4f}")
print(f"degree_p75             : {np.percentile(degree, 75):.4f}")
print(f"degree_p90             : {np.percentile(degree, 90):.4f}")
print(f"degree_p95             : {np.percentile(degree, 95):.4f}")
print(f"degree_max             : {degree.max()}")
print(f"weight_min             : {weights.min():.6f}")
print(f"weight_mean            : {weights.mean():.6f}")
print(f"weight_max             : {weights.max():.6f}")

top_indices = np.argsort(degree)[-20:][::-1]
print("\nTop-20 channels by degree:")
for index in top_indices:
    print(f"channel={index:4d}, degree={degree[index]:4d}")
PY