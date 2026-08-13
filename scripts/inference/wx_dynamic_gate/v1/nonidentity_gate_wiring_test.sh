#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Stage 1: non-identity DynamicChannelGate wiring test
#
# This is not a performance evaluation.
#
# The gate is still untrained. Because its final layer is initialized
# with zero weights, its initial output is approximately the configured
# target keep ratio for every query.
# ============================================================


export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

export PCD_ROOT="${PCD_ROOT:-/media/hwx/Xixixi/code-vla/PCD}"
export MODEL_ROOT="${MODEL_ROOT:-/media/hwx/Xixixi/code-vla/VLA/PCD}"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1


GPU_ID="${GPU_ID:-1}"
TASK="${TASK:-google_robot_pick_coke_can}"
CHECKPOINT="${CHECKPOINT:-${MODEL_ROOT}/open_pi_zero}"

TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.99}"
NUM_GROUPS="${NUM_GROUPS:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"

N_TRAJS=1
NUM_GPUS=1

TIME_TAG="$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="${PCD_ROOT}/results/stage1/nonidentity_wiring/${TIME_TAG}"
ROLLOUT_ROOT="${PCD_ROOT}/rollouts/stage1_nonidentity/${TIME_TAG}"


cd "${PCD_ROOT}"


echo "============================================================"
echo "Stage 1 non-identity Dynamic Gate wiring test"
echo "============================================================"
echo "PCD_ROOT          : ${PCD_ROOT}"
echo "MODEL_ROOT        : ${MODEL_ROOT}"
echo "CHECKPOINT        : ${CHECKPOINT}"
echo "GPU_ID            : ${GPU_ID}"
echo "TASK              : ${TASK}"
echo "TARGET_KEEP_RATIO : ${TARGET_KEEP_RATIO}"
echo "NUM_GROUPS        : ${NUM_GROUPS}"
echo "HIDDEN_DIM        : ${HIDDEN_DIM}"
echo "ROLLOUT_ROOT      : ${ROLLOUT_ROOT}"
echo "============================================================"


CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python parallel_inference.py \
    --num-gpus "${NUM_GPUS}" \
    --result-root "${RESULT_ROOT}" \
    --n-trajs "${N_TRAJS}" \
    --policy pizero \
    --checkpoint "${CHECKPOINT}" \
    --task "${TASK}" \
    --run-name dynamic_gate_wiring \
    --no-timestamp \
    --collect-rollouts \
    --rollout-root "${ROLLOUT_ROOT}" \
    --no-save-gif \
    --opts \
        use_torch_compile False \
        use_naive False \
        random_feature_mask False \
        adaptive_feature_mask False \
        learnable_feature_mask False \
        dynamic_feature_gate True \
        dynamic_gate_mode dynamic \
        dynamic_gate_num_groups "${NUM_GROUPS}" \
        dynamic_gate_hidden_dim "${HIDDEN_DIM}" \
        dynamic_gate_target_keep_ratio "${TARGET_KEEP_RATIO}" \
        dynamic_gate_temperature 1.0 \
        dynamic_gate_rescale False \
        dynamic_gate_verbose True \
        dynamic_gate_log_every 1


echo
echo "[CHECK] Validating the newly saved rollout..."

python scripts/check_rollout_dataset.py \
    --root "${ROLLOUT_ROOT}" \
    --max-episodes 1


EPISODE_DIR="${ROLLOUT_ROOT}/${TASK}/episode_000000"

if [[ ! -f "${EPISODE_DIR}/trajectory.npz" ]]; then
    echo "[ERROR] trajectory.npz was not created: ${EPISODE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${EPISODE_DIR}/metadata.json" ]]; then
    echo "[ERROR] metadata.json was not created: ${EPISODE_DIR}" >&2
    exit 1
fi


echo
echo "============================================================"
echo "Non-identity Dynamic Gate wiring test completed."
echo "Episode directory: ${EPISODE_DIR}"
echo
echo "This only validates closed-loop wiring."
echo "The gate has not been trained and success rate is not yet a"
echo "method-performance result."
echo "============================================================"