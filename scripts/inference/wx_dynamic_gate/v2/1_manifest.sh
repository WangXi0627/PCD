# 构建 manifest

mkdir -p manifests/dynamic_gate

python scripts/build_dynamic_gate_manifest.py \
  --rollout-root /media/hwx/Xixixi/code-vla/PCD/rollouts/pizero \
  --policy-setup google_robot \
  --output /media/hwx/Xixixi/code-vla/PCD/manifests/dynamic_gate/google_robot.json \
  --validation-count-per-task 4 \
  --split-seed 0

python scripts/build_dynamic_gate_manifest.py \
  --rollout-root /media/hwx/Xixixi/code-vla/PCD/rollouts/pizero \
  --policy-setup widowx_bridge \
  --output /media/hwx/Xixixi/code-vla/PCD/manifests/dynamic_gate/widowx_bridge.json \
  --validation-count-per-task 4 \
  --split-seed 0

# 检查 manifest 和 Dataset
python scripts/check_dynamic_gate_dataset.py \
  --manifest /media/hwx/Xixixi/code-vla/PCD/manifests/dynamic_gate/google_robot.json \
  --max-samples-to-read 64

python scripts/check_dynamic_gate_dataset.py \
  --manifest /media/hwx/Xixixi/code-vla/PCD/manifests/dynamic_gate/widowx_bridge.json \
  --max-samples-to-read 64