mkdir -p teacher_cache/dynamic_gate/google_robot

CUDA_VISIBLE_DEVICES=1 \
python scripts/cache_fixed_noise_teacher.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --split train \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir teacher_cache/dynamic_gate/google_robot/train \
  --policy-setup google_robot \
  --flow-sampling beta \
  --batch-size 2 \
  --base-noise-seed 0

CUDA_VISIBLE_DEVICES=1 \
python scripts/cache_fixed_noise_teacher.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --split validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir teacher_cache/dynamic_gate/google_robot/validation \
  --policy-setup google_robot \
  --flow-sampling beta \
  --batch-size 2 \
  --base-noise-seed 0

mkdir -p teacher_cache/dynamic_gate/widowx_bridge

CUDA_VISIBLE_DEVICES=1 \
python scripts/cache_fixed_noise_teacher.py \
  --manifest manifests/dynamic_gate/widowx_bridge.json \
  --split train \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir teacher_cache/dynamic_gate/widowx_bridge/train \
  --policy-setup widowx_bridge \
  --flow-sampling beta \
  --batch-size 2 \
  --base-noise-seed 0

CUDA_VISIBLE_DEVICES=1 \
python scripts/cache_fixed_noise_teacher.py \
  --manifest manifests/dynamic_gate/widowx_bridge.json \
  --split validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir teacher_cache/dynamic_gate/widowx_bridge/validation \
  --policy-setup widowx_bridge \
  --flow-sampling beta \
  --batch-size 2 \
  --base-noise-seed 0