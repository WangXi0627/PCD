mkdir -p checkpoints/dynamic_gate/v1/google_robot_r095_dryrun

CUDA_VISIBLE_DEVICES=1 \
python scripts/train_dynamic_gate_v1.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/v1/google_robot_r095_dryrun \
  --policy-setup google_robot \
  --flow-sampling beta \
  --epochs 1 \
  --steps-per-epoch 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --validation-batch-size 1 \
  --max-validation-batches 4 \
  --learning-rate 1e-4 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-action-inv 1.0 \
  --lambda-mask-inv 0.1 \
  --lambda-budget 10.0