#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T official-style WidowX tasks runner
# wx: official protocol, horizon=300
# ==============================

# ---------- paths ----------
PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
GROOT_ROOT="/media/hwx/Xixixi/code-vla/Isaac-GR00T"

PYTHON="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
RUNNER="${PCD_ROOT}/parallel_inference_groot_official.py"

# ---------- GR00T server ----------
HOST="127.0.0.1"
PORT="5555"
TIMEOUT_MS="60000"

# ---------- eval config ----------
CHECKPOINT="GR00T-N1.6-bridge"
N_EPISODES=200
N_ENVS=5
N_ACTION_STEPS=4
MAX_EPISODE_STEPS=300

SAVE_VIDEO=False

OUTPUT_ROOT="${PCD_ROOT}/results"

FORCE=0

TASKS=(
  "widowx_spoon_on_towel"
  "widowx_carrot_on_plate"
  "widowx_stack_cube"
  "widowx_put_eggplant_in_basket"
)

RUN_NAME="official_widowx_all_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}_horizon${MAX_EPISODE_STEPS}_video${SAVE_VIDEO}"

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
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "FORCE=${FORCE}"
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

client = PolicyClient(
    host="${HOST}",
    port=int("${PORT}"),
    timeout_ms=int("${TIMEOUT_MS}"),
    strict=False,
)

ok = client.ping()
print("ping:", ok)
if not ok:
    raise SystemExit("Cannot connect to GR00T server.")

print("modality:", client.get_modality_config())
PY

echo "[Check] GR00T server is reachable."

for TASK in "${TASKS[@]}"; do
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

  CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${RUNNER}" \
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
    --run-name "${RUN_NAME}"

  echo "[Done] ${TASK}"
done