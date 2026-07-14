#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T official-style Google tasks runner
# wx: PCD + GR00T-N1.6 baseline
# ==============================

# ---------- paths ----------
PCD_ROOT="/data/Xixixi/VLA/PCD"
GROOT_ROOT="/data/Xixixi/VLA/Isaac-GR00T"

PYTHON="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
RUNNER="${PCD_ROOT}/parallel_inference_groot_official.py"

# ---------- GR00T server ----------
HOST="127.0.0.1"
PORT="5555"
TIMEOUT_MS="60000"

# ---------- eval config ----------
CHECKPOINT="GR00T-N1.6-fractal"
N_EPISODES=200
N_ENVS=5
N_ACTION_STEPS=1
MAX_EPISODE_STEPS=300

# 视频很占空间。baseline 大规模跑建议先 false。
SAVE_VIDEO=False

# 输出目录
OUTPUT_ROOT="${PCD_ROOT}/results"

# 统一 run name，方便后面汇总
RUN_NAME="official_google_all_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}_video${SAVE_VIDEO}"

# 是否强制重跑：0=已存在 summary.json 就跳过；1=强制重跑
FORCE=0

# ---------- Google Robot benchmark tasks ----------
TASKS=(
  "google_robot_pick_coke_can"
  "google_robot_pick_object"
  "google_robot_move_near"
  "google_robot_open_drawer"
  "google_robot_close_drawer"
  "google_robot_place_in_closed_drawer"
)

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
echo "RUN_NAME=${RUN_NAME}"
echo "========================================"

# ---------- check server ----------
echo "[Check] Testing GR00T server connection..."
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

# ---------- run tasks ----------
for TASK in "${TASKS[@]}"; do
  SUMMARY_PATH="${OUTPUT_ROOT}/${CHECKPOINT}/${RUN_NAME}/${TASK}/summary.json"

  echo
  echo "========================================"
  echo "[Task] ${TASK}"
  echo "Summary path: ${SUMMARY_PATH}"
  echo "========================================"

  if [[ "${FORCE}" == "0" && -f "${SUMMARY_PATH}" ]]; then
    echo "[Skip] summary.json already exists: ${SUMMARY_PATH}"
    continue
  fi

  "${PYTHON}" "${RUNNER}" \
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