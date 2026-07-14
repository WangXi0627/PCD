#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T official-style WidowX tasks runner
# wx: PCD + GR00T-N1.6-bridge baseline
#
# This script runs GR00T with the official GR00T action-dict path,
# but uses PCD-compatible task horizons.
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
N_EPISODES=300
N_ENVS=5
N_ACTION_STEPS=4

# baseline 大规模跑建议 False，视频很占空间。
SAVE_VIDEO=False

# 输出目录
OUTPUT_ROOT="${PCD_ROOT}/results"

# 是否强制重跑：0=已存在 summary.json 就跳过；1=强制重跑
FORCE=0

# ---------- WidowX / Bridge tasks ----------
# 这 4 个是 PCD 里常用的 WidowX 任务。
TASKS=(
  "widowx_spoon_on_towel"
  "widowx_carrot_on_plate"
  "widowx_stack_cube"
  "widowx_put_eggplant_in_basket"
)

# PCD-compatible horizon。
# 如果你不确定具体 horizon，可以先用下面脚本末尾的检测命令测一遍。
# 这里先给一个保守设置：统一 80。
# 如果某个任务在 PCD 中实际 horizon 更长，再改对应值。
declare -A TASK_HORIZON=(
  ["widowx_spoon_on_towel"]=60
  ["widowx_carrot_on_plate"]=60
  ["widowx_stack_cube"]=60
  ["widowx_put_eggplant_in_basket"]=120
)

# 统一 run name，方便后面汇总。
RUN_NAME="official_widowx_pcd_horizon_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}_video${SAVE_VIDEO}"

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