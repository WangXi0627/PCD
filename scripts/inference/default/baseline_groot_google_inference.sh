#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T official-style Google tasks runner
# wx: PCD + GR00T-N1.6 baseline
#
# This script runs GR00T with the official GR00T action-dict path,
# but uses PCD-compatible task horizons.
# ==============================

# ---------- paths ----------
PCD_ROOT="/data/Xixixi/VLA/PCD"
GROOT_ROOT="/data/Xixixi/VLA/Isaac-GR00T"

PYTHON="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
RUNNER="${PCD_ROOT}/parallel_inference_groot_official.py"

# ---------- GR00T server ----------
HOST="127.0.0.1"
PORT="4444"
TIMEOUT_MS="60000"

# ---------- eval config ----------
CHECKPOINT="GR00T-N1.6-fractal"
N_EPISODES=100
N_ENVS=5
N_ACTION_STEPS=1

# baseline 大规模跑建议 False，视频很占空间。
SAVE_VIDEO=False

# 输出目录
OUTPUT_ROOT="${PCD_ROOT}/results"

# 是否强制重跑：0=已存在 summary.json 就跳过；1=强制重跑
FORCE=0

# ---------- Google Robot tasks ----------
# 这里默认使用 PCD 仓库中常用的 5 个 Google Robot 任务。
# 这些 horizon 是你刚刚用 simpler_env.make(task) 测出来的。
TASKS=(
  "google_robot_pick_coke_can"
  "google_robot_move_near"
  "google_robot_open_drawer"
  "google_robot_close_drawer"
  "google_robot_place_in_closed_drawer"
)

# 如果后面确认 google_robot_pick_object 在 PCD 中可用，并且测出了 horizon，
# 再把它加入 TASKS，并在 TASK_HORIZON 里补上对应步数。
# TASKS+=("google_robot_pick_object")

declare -A TASK_HORIZON=(
  ["google_robot_pick_coke_can"]=80
  ["google_robot_move_near"]=80
  ["google_robot_open_drawer"]=116
  ["google_robot_close_drawer"]=116
  ["google_robot_place_in_closed_drawer"]=200
  # ["google_robot_pick_object"]=80
)

# 统一 run name，方便后面汇总。
RUN_NAME="official_google_pcd_horizon_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}_video${SAVE_VIDEO}"

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
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "FORCE=${FORCE}"
echo "========================================"

# ---------- basic checks ----------
if [[ ! -x "${PYTHON}" ]]; then
  echo "[Error] PYTHON not found or not executable: ${PYTHON}"
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "[Error] Runner not found: ${RUNNER}"
  exit 1
fi

# ---------- check server ----------
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

# ---------- run tasks ----------
for TASK in "${TASKS[@]}"; do
  if [[ -z "${TASK_HORIZON[$TASK]+x}" ]]; then
    echo "[Error] No horizon configured for task: ${TASK}"
    echo "Please add it to TASK_HORIZON."
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