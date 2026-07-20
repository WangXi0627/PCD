#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GR00T-N1.6 random feature mask WidowX / Bridge runner
# wx: GR00T server-side random feature dim mask
# ==============================

# ---------- paths ----------
PCD_ROOT="/data/Xixixi/VLA/PCD"
GROOT_ROOT="/data/Xixixi/VLA/Isaac-GR00T"

# random feature mask 不依赖 PCD pretrained 权重；
# 但保留 MODEL_ROOT 不影响，后面如果混合 PCD 也方便。
export MODEL_ROOT="/data/Xixixi/Model/PCD"

PYTHON="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
RUNNER="${PCD_ROOT}/parallel_inference_groot_pcd.py"

# GROOT_ROOT 放前面，确保使用你修改过的本地 gr00t/policy 代码。
export PYTHONPATH="${GROOT_ROOT}:${PCD_ROOT}:${PCD_ROOT}/third_party/grounded_sam_2:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# ---------- GR00T server ----------
# 必须和你启动 bridge server 的端口一致。
HOST="127.0.0.1"
PORT="5555"
TIMEOUT_MS="60000"

# ---------- eval config ----------
CHECKPOINT="GR00T-N1.6-bridge"
N_EPISODES=300

# random feature mask 不加载 grounded_sam_tracking，可以并行。
# 如果先小测，可以改成 1。
N_ENVS=5
N_ACTION_STEPS=4
SAVE_VIDEO=False

# ---------- PCD disabled ----------
# 这里只测 random feature mask，不启用 PCD。
PCD_ENABLE=False

# pcd_enable=False 时这些参数不会进入 PCD 逻辑；
# 保留只是为了兼容 runner 参数。
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

# ---------- random feature mask config ----------
FEATURE_MASK_ENABLE=True
FEATURE_MASK_RESCALE=True

# 正式跑建议 False，避免 server 端刷屏。
# 调试 target 是否生效时可以改成 True。
FEATURE_MASK_VERBOSE=False
FEATURE_MASK_PRINT_CANDIDATES=False

# 更接近 PiZero multi_modal_projector / action_layer 的 GR00T target。
# 第一轮建议先保留 1-2 个。
FEATURE_MASK_TARGETS=(
  "backbone.model.mlp1"
  # "action_head.model.transformer_blocks.16"
  # "backbone.model.language_model.model.layers.8"
  # "backbone.model.language_model.model.layers.12"
  # "action_head.model.transformer_blocks.8"
  # "action_head.model.transformer_blocks.24"
)

# keep_ratio=1.0 是 no-mask sanity check。
# 如果已有 baseline，可以先不跑 1.0。
FEATURE_MASK_KEEP_RATIOS=(
  # "1.0"
  # "0.9"
  # "0.8"
  "0.7"
)

FEATURE_MASK_SEEDS=(
  "0"
  "1"
  "2"
)

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

sanitize_name() {
  local name="$1"
  name="${name// /_}"
  name="$(echo "${name}" | sed 's/[^A-Za-z0-9_.=+\-]/_/g')"

  local max_len=100
  if [[ ${#name} -le ${max_len} ]]; then
    echo "${name}"
  else
    local prefix="${name:0:${max_len}}"
    local digest
    digest="$(echo -n "${name}" | md5sum | awk '{print $1}' | cut -c1-8)"
    echo "${prefix}--${digest}"
  fi
}

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
echo "PCD_ENABLE=${PCD_ENABLE}"
echo "FEATURE_MASK_ENABLE=${FEATURE_MASK_ENABLE}"
echo "FEATURE_MASK_RESCALE=${FEATURE_MASK_RESCALE}"
echo "FEATURE_MASK_VERBOSE=${FEATURE_MASK_VERBOSE}"
echo "FEATURE_MASK_PRINT_CANDIDATES=${FEATURE_MASK_PRINT_CANDIDATES}"
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
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

for TARGET in "${FEATURE_MASK_TARGETS[@]}"; do
  TARGET_TAG="$(sanitize_name "${TARGET}")"

  for KEEP_RATIO in "${FEATURE_MASK_KEEP_RATIOS[@]}"; do
    KEEP_TAG="$(echo "${KEEP_RATIO}" | sed 's/\./p/g')"

    for SEED in "${FEATURE_MASK_SEEDS[@]}"; do
      RUN_NAME="groot_random_feature_mask/widowx_ep${N_EPISODES}_env${N_ENVS}_act${N_ACTION_STEPS}/target=${TARGET_TAG}/keep=${KEEP_TAG}/seed=${SEED}_rescale=${FEATURE_MASK_RESCALE}_video=${SAVE_VIDEO}"

      echo
      echo "########################################"
      echo "[Experiment]"
      echo "TARGET=${TARGET}"
      echo "KEEP_RATIO=${KEEP_RATIO}"
      echo "SEED=${SEED}"
      echo "RUN_NAME=${RUN_NAME}"
      echo "########################################"

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
          --no-timestamp \
          --force "${FORCE}" \
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
          --pcd_debug_image_interval "${PCD_DEBUG_IMAGE_INTERVAL}" \
          --feature_mask_enable "${FEATURE_MASK_ENABLE}" \
          --feature_mask_target "${TARGET}" \
          --feature_mask_keep_ratio "${KEEP_RATIO}" \
          --feature_mask_seed "${SEED}" \
          --feature_mask_rescale "${FEATURE_MASK_RESCALE}" \
          --feature_mask_verbose "${FEATURE_MASK_VERBOSE}" \
          --feature_mask_print_candidates "${FEATURE_MASK_PRINT_CANDIDATES}"

        echo "[Done] ${TASK}"
      done
    done
  done
done

echo
echo "========================================"
echo "[All Done] GR00T random feature mask WidowX experiments finished."
echo "========================================"