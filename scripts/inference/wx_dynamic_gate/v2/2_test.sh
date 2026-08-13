export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

mkdir -p "${PCD_ROOT}/checkpoints/dynamic_gate/stage2"

CUDA_VISIBLE_DEVICES=1 \
python scripts/smoke_test_dynamic_gate_training.py \
  --manifest "${PCD_ROOT}/manifests/dynamic_gate/google_robot.json" \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-checkpoint "${PCD_ROOT}/checkpoints/dynamic_gate/stage2/google_robot_smoke.pt" \
  --policy-setup google_robot \
  --flow-sampling beta \
  --split train \
  --num-samples 2 \
  --steps 50 \
  --learning-rate 1e-3 \
  --base-noise-seed 0 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.75 \
  --temperature 1.0

CUDA_VISIBLE_DEVICES=1 \
python scripts/smoke_test_dynamic_gate_training.py \
  --manifest "${PCD_ROOT}/manifests/dynamic_gate/widowx_bridge.json" \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --output-checkpoint "${PCD_ROOT}/checkpoints/dynamic_gate/stage2/widowx_bridge_smoke.pt" \
  --policy-setup widowx_bridge \
  --flow-sampling beta \
  --split train \
  --num-samples 2 \
  --steps 50 \
  --learning-rate 1e-3 \
  --base-noise-seed 0 \
  --num-groups 64 \
  --hidden-dim 512 \
  --target-keep-ratio 0.75 \
  --temperature 1.0