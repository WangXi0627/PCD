#!/usr/bin/env bash

set -euo pipefail

# Stage 3B: deterministic new-seed online comparison.
# Example checkpoint-selection run:
#   EPISODE_START=20 N_TRAJS=4 bash scripts/inference/stage3/evaluate_v1_checkpoint.sh
# Final test:
#   EPISODE_START=24 N_TRAJS=16 bash scripts/inference/stage3/evaluate_v1_checkpoint.sh

export PCD_ROOT="${PCD_ROOT:-/media/hwx/Xixixi/code-vla/PCD}"
export MODEL_ROOT="${MODEL_ROOT:-/media/hwx/Xixixi/code-vla/VLA/PCD}"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

GPU_ID="${GPU_ID:-1}"
TASK="${TASK:-google_robot_pick_coke_can}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${MODEL_ROOT}/open_pi_zero}"
GATE_CHECKPOINT="${GATE_CHECKPOINT:?Set GATE_CHECKPOINT to a V1 checkpoint}"

EPISODE_START="${EPISODE_START:-20}"
N_TRAJS="${N_TRAJS:-4}"
ACTION_NOISE_BASE_SEED="${ACTION_NOISE_BASE_SEED:-0}"

NUM_GROUPS="${NUM_GROUPS:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.90}"
TEMPERATURE="${TEMPERATURE:-1.0}"

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${PCD_ROOT}/results/dynamic_gate/stage3_v1_online/${TIME_TAG}"

cd "${PCD_ROOT}"

echo "============================================================"
echo "Stage 3B deterministic online evaluation"
echo "TASK                   : ${TASK}"
echo "GATE_CHECKPOINT        : ${GATE_CHECKPOINT}"
echo "EPISODE_START          : ${EPISODE_START}"
echo "N_TRAJS                : ${N_TRAJS}"
echo "ACTION_NOISE_BASE_SEED : ${ACTION_NOISE_BASE_SEED}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python parallel_inference.py \
  --num-gpus 1 \
  --result-root "${RESULT_ROOT}/baseline" \
  --n-trajs "${N_TRAJS}" \
  --episode-start "${EPISODE_START}" \
  --policy pizero \
  --checkpoint "${BASE_CHECKPOINT}" \
  --task "${TASK}" \
  --run-name baseline_fixed_noise \
  --no-timestamp \
  --no-save-gif \
  --opts \
    use_torch_compile False \
    random_feature_mask False \
    adaptive_feature_mask False \
    learnable_feature_mask False \
    dynamic_feature_gate False \
    deterministic_action_noise True \
    action_noise_base_seed "${ACTION_NOISE_BASE_SEED}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python parallel_inference.py \
  --num-gpus 1 \
  --result-root "${RESULT_ROOT}/dynamic_gate" \
  --n-trajs "${N_TRAJS}" \
  --episode-start "${EPISODE_START}" \
  --policy pizero \
  --checkpoint "${BASE_CHECKPOINT}" \
  --task "${TASK}" \
  --run-name dynamic_gate_v1_fixed_noise \
  --no-timestamp \
  --no-save-gif \
  --opts \
    use_torch_compile False \
    random_feature_mask False \
    adaptive_feature_mask False \
    learnable_feature_mask False \
    dynamic_feature_gate True \
    dynamic_gate_mode dynamic \
    dynamic_gate_checkpoint "${GATE_CHECKPOINT}" \
    dynamic_gate_num_groups "${NUM_GROUPS}" \
    dynamic_gate_hidden_dim "${HIDDEN_DIM}" \
    dynamic_gate_target_keep_ratio "${TARGET_KEEP_RATIO}" \
    dynamic_gate_temperature "${TEMPERATURE}" \
    dynamic_gate_rescale False \
    dynamic_gate_verbose False \
    deterministic_action_noise True \
    action_noise_base_seed "${ACTION_NOISE_BASE_SEED}"

echo "[OK] Results saved under: ${RESULT_ROOT}"
