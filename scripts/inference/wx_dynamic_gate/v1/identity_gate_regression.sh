#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Stage 1: Identity Gate lossless closed-loop regression
#
# Run:
# 1. Original PiZero baseline
# 2. PiZeroDynamicGateInference + IdentityVisualGate
# 3. Compare the two query-level rollouts
# ============================================================


export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

export PCD_ROOT="${PCD_ROOT:-/media/hwx/Xixixi/code-vla/PCD}"
export MODEL_ROOT="${MODEL_ROOT:-/media/hwx/Xixixi/code-vla/VLA/PCD}"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1


GPU_ID="${GPU_ID:-1}"
TASK="${TASK:-google_robot_pick_coke_can}"
CHECKPOINT="${CHECKPOINT:-${MODEL_ROOT}/open_pi_zero}"

N_TRAJS="${N_TRAJS:-1}"
NUM_GPUS=1

TIME_TAG="$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="${PCD_ROOT}/results/stage1/identity_regression/${TIME_TAG}"

BASELINE_ROLLOUT_ROOT="${PCD_ROOT}/rollouts/stage1_identity/${TIME_TAG}/baseline"
IDENTITY_ROLLOUT_ROOT="${PCD_ROOT}/rollouts/stage1_identity/${TIME_TAG}/identity"


cd "${PCD_ROOT}"


echo "============================================================"
echo "Stage 1 Identity Gate regression"
echo "============================================================"
echo "PCD_ROOT               : ${PCD_ROOT}"
echo "MODEL_ROOT             : ${MODEL_ROOT}"
echo "CHECKPOINT             : ${CHECKPOINT}"
echo "GPU_ID                 : ${GPU_ID}"
echo "TASK                   : ${TASK}"
echo "BASELINE_ROLLOUT_ROOT  : ${BASELINE_ROLLOUT_ROOT}"
echo "IDENTITY_ROLLOUT_ROOT  : ${IDENTITY_ROLLOUT_ROOT}"
echo "============================================================"


echo
echo "[STEP 1/3] Running original PiZero baseline..."

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python parallel_inference.py \
    --num-gpus "${NUM_GPUS}" \
    --result-root "${RESULT_ROOT}/baseline" \
    --n-trajs "${N_TRAJS}" \
    --policy pizero \
    --checkpoint "${CHECKPOINT}" \
    --task "${TASK}" \
    --run-name baseline \
    --no-timestamp \
    --collect-rollouts \
    --rollout-root "${BASELINE_ROLLOUT_ROOT}" \
    --no-save-gif \
    --opts \
        use_torch_compile False \
        use_naive False \
        random_feature_mask False \
        adaptive_feature_mask False \
        learnable_feature_mask False \
        dynamic_feature_gate False


echo
echo "[STEP 2/3] Running PiZero dynamic wrapper with Identity gate..."

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python parallel_inference.py \
    --num-gpus "${NUM_GPUS}" \
    --result-root "${RESULT_ROOT}/identity" \
    --n-trajs "${N_TRAJS}" \
    --policy pizero \
    --checkpoint "${CHECKPOINT}" \
    --task "${TASK}" \
    --run-name identity_gate \
    --no-timestamp \
    --collect-rollouts \
    --rollout-root "${IDENTITY_ROLLOUT_ROOT}" \
    --no-save-gif \
    --opts \
        use_torch_compile False \
        use_naive False \
        random_feature_mask False \
        adaptive_feature_mask False \
        learnable_feature_mask False \
        dynamic_feature_gate True \
        dynamic_gate_mode identity \
        dynamic_gate_verbose True \
        dynamic_gate_log_every 1


BASELINE_EPISODE="${BASELINE_ROLLOUT_ROOT}/${TASK}/episode_000000"
IDENTITY_EPISODE="${IDENTITY_ROLLOUT_ROOT}/${TASK}/episode_000000"


echo
echo "[STEP 3/3] Comparing baseline and Identity-gate rollouts..."

python scripts/compare_rollout_episodes.py \
    --reference-episode "${BASELINE_EPISODE}" \
    --candidate-episode "${IDENTITY_EPISODE}" \
    --atol 1e-5 \
    --rtol 1e-5


echo
echo "============================================================"
echo "Identity Gate closed-loop regression passed."
echo "Baseline episode: ${BASELINE_EPISODE}"
echo "Identity episode: ${IDENTITY_EPISODE}"
echo "============================================================"