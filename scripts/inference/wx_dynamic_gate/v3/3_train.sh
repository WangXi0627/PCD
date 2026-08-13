mkdir -p checkpoints/dynamic_gate/v1/google_robot_r090

CUDA_VISIBLE_DEVICES=3 \
python scripts/train_dynamic_gate_v1.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --checkpoint-path "/media/hwx/Xixixi/code-vla/VLA/PCD/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/v1/google_robot_r095 \
  --policy-setup google_robot \
  --flow-sampling beta \
  --epochs 10 \
  --steps-per-epoch 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --learning-rate 1e-4 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-action-inv 1.0 \
  --lambda-mask-inv 0.1 \
  --lambda-budget 10.0


# CUDA out of memory

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

CUDA_VISIBLE_DEVICES=1 \
python scripts/train_dynamic_gate_v1.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/v1/google_robot_r095 \
  --policy-setup google_robot \
  --flow-sampling beta \
  --epochs 10 \
  --steps-per-epoch 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --max-validation-batches 64 \
  --learning-rate 1e-4 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-action-inv 1.0 \
  --lambda-mask-inv 0.1 \
  --lambda-budget 10.0 \
  --resume checkpoints/dynamic_gate/v1/google_robot_r095/last.pt

# 新训练脚本
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

CUDA_VISIBLE_DEVICES=1 \
python scripts/train_dynamic_gate_v1.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/v1/google_robot_r090_memory_test \
  --policy-setup google_robot \
  --flow-sampling beta \
  --epochs 3 \
  --steps-per-epoch 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --max-validation-batches 64 \
  --learning-rate 1e-4 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.90 \
  --lambda-suf 1.0 \
  --lambda-action-inv 1.0 \
  --lambda-mask-inv 0.1 \
  --lambda-budget 10.0