# c0 最小动作保持基线
export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD

CUDA_VISIBLE_DEVICES=1 \
python scripts/train_dynamic_gate_redundancy.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --train-teacher-cache teacher_cache/dynamic_gate/google_robot/train \
  --validation-teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-dir checkpoints/dynamic_gate/channel_red/google/c0_suf_budget \
  --policy-setup google_robot \
  --epochs 10 \
  --steps-per-epoch 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --max-validation-batches 64 \
  --learning-rate 1e-4 \
  --num-groups 2048 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-redundancy 0.0 \
  --lambda-action-inv 0.0 \
  --lambda-mask-inv 0.0 \
  --lambda-budget 10.0

# c1 加入维度级去冗余
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
  --output-dir checkpoints/dynamic_gate/channel_red/google/c1_suf_budget_red \
  --policy-setup google_robot \
  --epochs 10 \
  --steps-per-epoch 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --validation-batch-size 1 \
  --max-validation-batches 64 \
  --learning-rate 1e-4 \
  --num-groups 2048 \
  --hidden-dim 512 \
  --target-keep-ratio 0.95 \
  --lambda-suf 1.0 \
  --lambda-redundancy 1e-4 \
  --lambda-action-inv 0.0 \
  --lambda-mask-inv 0.0 \
  --lambda-budget 10.0

# c2 加入弱 action invariance
--lambda-redundancy 1e-4 \
--lambda-action-inv 0.1 \
--lambda-mask-inv 0.0

# c3 最后加入弱 mask invariance
--lambda-action-inv 0.1 \
--lambda-mask-inv 0.01
