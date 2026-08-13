# redundancy 权重标定
# C1-a：λred = 1e-4
# C1-b：λred = 3e-4
# C1-c：λred = 1e-3

export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD

CUDA_VISIBLE_DEVICES=2 \
python scripts/train_dynamic_gate_redundancy.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --redundancy-cache redundancy_cache/dynamic_gate/google_robot/train_mean_top16_corr065 \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/channel_red/google/c1a_red3e-4_dryrun \
  --policy-setup google_robot \
  --flow-sampling beta \
  --epochs 1 \
  --steps-per-epoch 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --max-validation-batches 32 \
  --learning-rate 1e-4 \
  --num-groups 2048 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-redundancy 3e-4 \
  --lambda-action-inv 0.0 \
  --lambda-mask-inv 0.0 \
  --lambda-budget 10.0