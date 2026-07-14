#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T + original PCD grounded_sam_tracking WidowX runner
# wx: PCD + GR00T-N1.6-bridge + original ContrastImageGenerator + ContrastDecoding
# ==============================

# ---------- paths ----------
PCD_ROOT="/data/Xixixi/VLA/PCD"
GROOT_ROOT="/data/Xixixi/VLA/Isaac-GR00T"
export MODEL_ROOT="/data/Xixixi/Model/PCD"
PYTHON="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
RUNNER="${PCD_ROOT}/parallel_inference_groot_pcd.py"

export PYTHONPATH="${PCD_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/third_party/grounded_sam_2:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# ---------- GR00T server ----------
HOST="127.0.0.1"
PORT="5555"
TIMEOUT_MS="60000"

# ---------- eval config ----------
CHECKPOINT="GR00T-N1.6-bridge"
N_EPISODES=100
N_ENVS=1
N_ACTION_STEPS=4
SAVE_VIDEO=False

# ---------- original PCD config ----------
PCD_ENABLE=True
PCD_NUM_REPEATS=24
PCD_ALPHA=0.2
PCD_BANDWIDTH_FACTOR=1.0
PCD_KEEP_THRESHOLD=0.5

PCD_BY="grounded_sam_tracking"
PCD_INPAINT_MODE="lama"
PCD_GET_ALL_PARTS=False
PCD_FALLBACK_ON_ERROR=False
PCD_DEBUG_SAVE_IMAGES=False
PCD_DEBUG_IMAGE_INTERVAL=20

OUTPUT_ROOT="${PCD_ROOT}/results"
FORCE=0
CUDA_ID=0

# ---------- WidowX / Bridge tasks ----------
TASKS=(
  "widowx_spoon_on_towel"
  "widowx_carrot_on_plate"
  "widowx_stack_cube"
  "widowx_put_eggplant_in_basket"
)

declare -A TASK_HORIZON=(
  ["widowx_spoon_on_towel"]=60
  ["widowx_carrot_on_plate"]=60
  ["widowx_stack_cube"]=60
  ["widowx_put_eggplant_in_basket"]=120
)

RUN_NAME="groot_pcd_original_widowx_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}_rep${PCD_NUM_REPEATS}_alpha${PCD_ALPHA}_by${PCD_BY}_inpaint${PCD_INPAINT_MODE}_video${SAVE_VIDEO}"

cd "${PCD_ROOT}"

echo "========================================"
echo "PCD_ROOT=${PCD_ROOT}"
echo "GROOT_ROOT=${GROOT_ROOT}"
echo "PYTHON=${PYTHON}"
echo "RUNNER=${RUNNER}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ENVS=${N_ENVS}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "PCD_NUM_REPEATS=${PCD_NUM_REPEATS}"
echo "PCD_ALPHA=${PCD_ALPHA}"
echo "PCD_BY=${PCD_BY}"
echo "PCD_INPAINT_MODE=${PCD_INPAINT_MODE}"
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "FORCE=${FORCE}"
echo "CUDA_ID=${CUDA_ID}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "========================================"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[Error] PYTHON not found or not executable: ${PYTHON}"
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "[Error] Runner not found: ${RUNNER}"
  exit 1
fi

echo
echo "========================================"
echo "[Check] Testing GR00T server connection"
echo "========================================"

"${PYTHON}" - <<PY
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="${HOST}", port=int("${PORT}"), timeout_ms=int("${TIMEOUT_MS}"), strict=False)
ok = client.ping()
print("ping:", ok)
if not ok:
    raise SystemExit("Cannot connect to GR00T server.")
print("modality:", client.get_modality_config())
PY

echo "[Check] GR00T server is reachable."

for TASK in "${TASKS[@]}"; do
  if [[ -z "${TASK_HORIZON[$TASK]+x}" ]]; then
    echo "[Error] No horizon configured for task: ${TASK}"
    exit 1
  fi

  MAX_EPISODE_STEPS="${TASK_HORIZON[$TASK]}"
  SUMMARY_PATH="${OUTPUT_ROOT}/${CHECKPOINT}/${RUN_NAME}/${TASK}/summary.json"

  echo
  echo "========================================"
  echo "[Task] ${TASK}"
  echo "Horizon: ${MAX_EPISODE_STEPS}"
  echo "Summary path: ${SUMMARY_PATH}"
  echo "========================================"

  if [[ "${FORCE}" == "0" && -f "${SUMMARY_PATH}" ]]; then
    echo "[Skip] summary.json already exists: ${SUMMARY_PATH}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON}" "${RUNNER}" \
    --task "${TASK}" \
    --checkpoint "${CHECKPOINT}" \
    --n-trajs "${N_EPISODES}" \
    --policy_client_host "${HOST}" \
    --policy_client_port "${PORT}" \
    --timeout_ms "${TIMEOUT_MS}" \
    --max_episode_steps "${MAX_EPISODE_STEPS}" \
    --n_action_steps "${N_ACTION_STEPS}" \
    --n_envs "${N_ENVS}" \
    --save_video "${SAVE_VIDEO}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --pcd_enable "${PCD_ENABLE}" \
    --pcd_num_repeats "${PCD_NUM_REPEATS}" \
    --pcd_alpha "${PCD_ALPHA}" \
    --pcd_bandwidth_factor "${PCD_BANDWIDTH_FACTOR}" \
    --pcd_keep_threshold "${PCD_KEEP_THRESHOLD}" \
    --pcd_by "${PCD_BY}" \
    --pcd_inpaint_mode "${PCD_INPAINT_MODE}" \
    --pcd_get_all_parts "${PCD_GET_ALL_PARTS}" \
    --pcd_fallback_on_error "${PCD_FALLBACK_ON_ERROR}" \
    --pcd_debug_save_images "${PCD_DEBUG_SAVE_IMAGES}" \
    --pcd_debug_image_interval "${PCD_DEBUG_IMAGE_INTERVAL}"

  echo "[Done] ${TASK}"
done