#!/usr/bin/env bash
# Evaluate Dynamic Gate V1 against the PiZero baseline on every task in a suite.
#
# Suites:
#   SUITE=google_robot   -> 5 Google Robot tasks
#   SUITE=widowx_bridge  -> 4 WidowX tasks
#   SUITE=all            -> both suites; requires two gate checkpoints
#
# Final-test default:
#   environment seeds 24-39 (EPISODE_START=24, N_TRAJS=16)
#
# Examples:
#   GATE_CHECKPOINT=/path/to/google_robot_best.pt \
#   SUITE=google_robot GPU_ID=1 \
#   bash scripts/inference/stage3/evaluate_v1_all_tasks.sh
#
#   SUITE=all \
#   GOOGLE_GATE_CHECKPOINT=/path/to/google_robot_best.pt \
#   WIDOWX_GATE_CHECKPOINT=/path/to/widowx_best.pt \
#   GPU_ID=1 \
#   bash scripts/inference/stage3/evaluate_v1_all_tasks.sh

set -euo pipefail


# ---------------------------------------------------------------------------
# Paths and common evaluation settings
# ---------------------------------------------------------------------------

export PCD_ROOT="${PCD_ROOT:-/media/hwx/Xixixi/code-vla/PCD}"
export MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing open_pi_zero}"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

GPU_ID="${GPU_ID:-1}"
SUITE="${SUITE:-google_robot}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-${MODEL_ROOT}/open_pi_zero}"

EPISODE_START="${EPISODE_START:-24}"
N_TRAJS="${N_TRAJS:-16}"
ACTION_NOISE_BASE_SEED="${ACTION_NOISE_BASE_SEED:-0}"

NUM_GROUPS="${NUM_GROUPS:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"

GOOGLE_TARGET_KEEP_RATIO="${GOOGLE_TARGET_KEEP_RATIO:-0.90}"
WIDOWX_TARGET_KEEP_RATIO="${WIDOWX_TARGET_KEEP_RATIO:-0.90}"

# For one-suite runs, GATE_CHECKPOINT is the convenient generic variable.
GATE_CHECKPOINT="${GATE_CHECKPOINT:-}"
GOOGLE_GATE_CHECKPOINT="${GOOGLE_GATE_CHECKPOINT:-${GATE_CHECKPOINT}}"
WIDOWX_GATE_CHECKPOINT="${WIDOWX_GATE_CHECKPOINT:-${GATE_CHECKPOINT}}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
SAVE_GIF="${SAVE_GIF:-0}"

TIME_TAG="${TIME_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-final_seed${EPISODE_START}_to_$((EPISODE_START + N_TRAJS - 1))_${TIME_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-${PCD_ROOT}/results/v3_online/${RUN_NAME}}"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
FAILURE_LOG="${RESULT_ROOT}/failures.log"

if [[ ! -d "${PCD_ROOT}" ]]; then
    echo "[ERROR] PCD_ROOT does not exist: ${PCD_ROOT}" >&2
    exit 1
fi

if [[ ! -e "${BASE_CHECKPOINT}" ]]; then
    echo "[ERROR] BASE_CHECKPOINT does not exist: ${BASE_CHECKPOINT}" >&2
    exit 1
fi

if ! [[ "${EPISODE_START}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] EPISODE_START must be a non-negative integer." >&2
    exit 1
fi

if ! [[ "${N_TRAJS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] N_TRAJS must be a positive integer." >&2
    exit 1
fi

case "${SUITE}" in
    google_robot)
        if [[ -z "${GOOGLE_GATE_CHECKPOINT}" && "${RUN_DYNAMIC}" == "1" ]]; then
            echo "[ERROR] Set GATE_CHECKPOINT or GOOGLE_GATE_CHECKPOINT." >&2
            exit 1
        fi
        ;;
    widowx_bridge)
        if [[ -z "${WIDOWX_GATE_CHECKPOINT}" && "${RUN_DYNAMIC}" == "1" ]]; then
            echo "[ERROR] Set GATE_CHECKPOINT or WIDOWX_GATE_CHECKPOINT." >&2
            exit 1
        fi
        ;;
    all)
        if [[ "${RUN_DYNAMIC}" == "1" ]]; then
            if [[ -z "${GOOGLE_GATE_CHECKPOINT}" ]]; then
                echo "[ERROR] SUITE=all requires GOOGLE_GATE_CHECKPOINT." >&2
                exit 1
            fi
            if [[ -z "${WIDOWX_GATE_CHECKPOINT}" ]]; then
                echo "[ERROR] SUITE=all requires WIDOWX_GATE_CHECKPOINT." >&2
                exit 1
            fi
        fi
        ;;
    *)
        echo "[ERROR] Unknown SUITE=${SUITE}. Use google_robot, widowx_bridge, or all." >&2
        exit 1
        ;;
esac

if [[ "${RUN_DYNAMIC}" == "1" ]]; then
    if [[ -n "${GOOGLE_GATE_CHECKPOINT}" && ! -f "${GOOGLE_GATE_CHECKPOINT}" ]]; then
        echo "[ERROR] Google gate checkpoint not found: ${GOOGLE_GATE_CHECKPOINT}" >&2
        exit 1
    fi
    if [[ -n "${WIDOWX_GATE_CHECKPOINT}" && ! -f "${WIDOWX_GATE_CHECKPOINT}" ]]; then
        echo "[ERROR] WidowX gate checkpoint not found: ${WIDOWX_GATE_CHECKPOINT}" >&2
        exit 1
    fi
fi

mkdir -p "${RESULT_ROOT}"
: > "${FAILURE_LOG}"

cat > "${RESULT_ROOT}/run_config.txt" <<EOF
PCD_ROOT=${PCD_ROOT}
MODEL_ROOT=${MODEL_ROOT}
BASE_CHECKPOINT=${BASE_CHECKPOINT}
SUITE=${SUITE}
GPU_ID=${GPU_ID}
EPISODE_START=${EPISODE_START}
N_TRAJS=${N_TRAJS}
ACTION_NOISE_BASE_SEED=${ACTION_NOISE_BASE_SEED}
RUN_BASELINE=${RUN_BASELINE}
RUN_DYNAMIC=${RUN_DYNAMIC}
NUM_GROUPS=${NUM_GROUPS}
HIDDEN_DIM=${HIDDEN_DIM}
TEMPERATURE=${TEMPERATURE}
GOOGLE_TARGET_KEEP_RATIO=${GOOGLE_TARGET_KEEP_RATIO}
WIDOWX_TARGET_KEEP_RATIO=${WIDOWX_TARGET_KEEP_RATIO}
GOOGLE_GATE_CHECKPOINT=${GOOGLE_GATE_CHECKPOINT}
WIDOWX_GATE_CHECKPOINT=${WIDOWX_GATE_CHECKPOINT}
RESULT_ROOT=${RESULT_ROOT}
EOF

printf "suite,task,method,success_rate,num_episodes,episode_start,episode_end,log_file\n" > "${SUMMARY_CSV}"

MODEL_NAME="$(basename "${BASE_CHECKPOINT%/}")"

GOOGLE_TASKS=(
    "google_robot_close_drawer"
    "google_robot_move_near"
    "google_robot_open_drawer"
    "google_robot_pick_coke_can"
    "google_robot_place_apple_in_closed_top_drawer"
)

WIDOWX_TASKS=(
    "widowx_carrot_on_plate"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_stack_cube"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_python_eval() {
    local method="$1"
    local suite_name="$2"
    local task="$3"
    local gate_checkpoint="$4"
    local target_keep_ratio="$5"

    local method_root="${RESULT_ROOT}/${suite_name}/${method}"
    local run_name
    local -a extra_args

    mkdir -p "${method_root}"

    if [[ "${SAVE_GIF}" == "1" ]]; then
        extra_args=()
    else
        extra_args=(--no-save-gif)
    fi

    if [[ "${method}" == "baseline" ]]; then
        run_name="baseline_fixed_noise"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${PCD_ROOT}/parallel_inference.py" \
            --num-gpus 1 \
            --result-root "${method_root}" \
            --n-trajs "${N_TRAJS}" \
            --episode-start "${EPISODE_START}" \
            --policy pizero \
            --checkpoint "${BASE_CHECKPOINT}" \
            --task "${task}" \
            --run-name "${run_name}" \
            --no-timestamp \
            "${extra_args[@]}" \
            --opts \
                use_torch_compile False \
                random_feature_mask False \
                adaptive_feature_mask False \
                learnable_feature_mask False \
                dynamic_feature_gate False \
                deterministic_action_noise True \
                action_noise_base_seed "${ACTION_NOISE_BASE_SEED}"

    elif [[ "${method}" == "dynamic_gate" ]]; then
        run_name="dynamic_gate_v1_fixed_noise"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${PCD_ROOT}/parallel_inference.py" \
            --num-gpus 1 \
            --result-root "${method_root}" \
            --n-trajs "${N_TRAJS}" \
            --episode-start "${EPISODE_START}" \
            --policy pizero \
            --checkpoint "${BASE_CHECKPOINT}" \
            --task "${task}" \
            --run-name "${run_name}" \
            --no-timestamp \
            "${extra_args[@]}" \
            --opts \
                use_torch_compile False \
                random_feature_mask False \
                adaptive_feature_mask False \
                learnable_feature_mask False \
                dynamic_feature_gate True \
                dynamic_gate_mode dynamic \
                dynamic_gate_checkpoint "${gate_checkpoint}" \
                dynamic_gate_num_groups "${NUM_GROUPS}" \
                dynamic_gate_hidden_dim "${HIDDEN_DIM}" \
                dynamic_gate_target_keep_ratio "${target_keep_ratio}" \
                dynamic_gate_temperature "${TEMPERATURE}" \
                dynamic_gate_rescale False \
                dynamic_gate_verbose False \
                deterministic_action_noise True \
                action_noise_base_seed "${ACTION_NOISE_BASE_SEED}"
    else
        echo "[ERROR] Unknown method: ${method}" >&2
        return 2
    fi
}

find_result_log() {
    local method="$1"
    local suite_name="$2"
    local task="$3"
    local run_name

    if [[ "${method}" == "baseline" ]]; then
        run_name="baseline_fixed_noise"
    else
        run_name="dynamic_gate_v1_fixed_noise"
    fi

    local task_dir="${RESULT_ROOT}/${suite_name}/${method}/${MODEL_NAME}/${run_name}/${task}"
    find "${task_dir}" \
        -maxdepth 1 \
        -type f \
        -name "000_success_*.log" \
        -print 2>/dev/null \
        | sort \
        | tail -n 1
}

append_summary() {
    local suite_name="$1"
    local task="$2"
    local method="$3"

    local log_file
    log_file="$(find_result_log "${method}" "${suite_name}" "${task}")"

    if [[ -z "${log_file}" ]]; then
        echo "[ERROR] Cannot find success log for ${suite_name}/${task}/${method}." >&2
        echo "${suite_name},${task},${method},missing_log" >> "${FAILURE_LOG}"
        return 1
    fi

    local filename
    filename="$(basename "${log_file}")"

    local success_rate
    success_rate="${filename#000_success_}"
    success_rate="${success_rate%.log}"

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${suite_name}" \
        "${task}" \
        "${method}" \
        "${success_rate}" \
        "${N_TRAJS}" \
        "${EPISODE_START}" \
        "$((EPISODE_START + N_TRAJS - 1))" \
        "${log_file}" \
        >> "${SUMMARY_CSV}"

    echo "[RESULT] suite=${suite_name} task=${task} method=${method} success=${success_rate}"
}

run_one_task() {
    local suite_name="$1"
    local task="$2"
    local gate_checkpoint="$3"
    local target_keep_ratio="$4"

    echo
    echo "================================================================"
    echo "Suite                  : ${suite_name}"
    echo "Task                   : ${task}"
    echo "Episodes               : ${EPISODE_START}-$((EPISODE_START + N_TRAJS - 1))"
    echo "Action noise base seed : ${ACTION_NOISE_BASE_SEED}"
    echo "================================================================"

    if [[ "${RUN_BASELINE}" == "1" ]]; then
        echo "[RUN] baseline: ${task}"
        if ! run_python_eval \
            "baseline" \
            "${suite_name}" \
            "${task}" \
            "${gate_checkpoint}" \
            "${target_keep_ratio}"; then
            echo "[FAILED] baseline ${suite_name}/${task}" | tee -a "${FAILURE_LOG}"
            return 1
        fi
        append_summary "${suite_name}" "${task}" "baseline"
    fi

    if [[ "${RUN_DYNAMIC}" == "1" ]]; then
        echo "[RUN] dynamic gate: ${task}"
        if ! run_python_eval \
            "dynamic_gate" \
            "${suite_name}" \
            "${task}" \
            "${gate_checkpoint}" \
            "${target_keep_ratio}"; then
            echo "[FAILED] dynamic_gate ${suite_name}/${task}" | tee -a "${FAILURE_LOG}"
            return 1
        fi
        append_summary "${suite_name}" "${task}" "dynamic_gate"
    fi
}

run_suite() {
    local suite_name="$1"
    local gate_checkpoint="$2"
    local target_keep_ratio="$3"
    shift 3
    local tasks=("$@")

    echo
    echo "################################################################"
    echo "Running suite: ${suite_name}"
    echo "Number of tasks: ${#tasks[@]}"
    echo "Gate checkpoint: ${gate_checkpoint}"
    echo "Target keep ratio: ${target_keep_ratio}"
    echo "################################################################"

    local task
    for task in "${tasks[@]}"; do
        run_one_task \
            "${suite_name}" \
            "${task}" \
            "${gate_checkpoint}" \
            "${target_keep_ratio}"
    done
}


# ---------------------------------------------------------------------------
# Run requested suite(s)
# ---------------------------------------------------------------------------

cd "${PCD_ROOT}"

echo "============================================================"
echo "Dynamic Gate V1 final new-seed evaluation"
echo "Suite                  : ${SUITE}"
echo "Episodes               : ${EPISODE_START}-$((EPISODE_START + N_TRAJS - 1))"
echo "Baseline               : ${RUN_BASELINE}"
echo "Dynamic Gate           : ${RUN_DYNAMIC}"
echo "Result root            : ${RESULT_ROOT}"
echo "============================================================"

case "${SUITE}" in
    google_robot)
        run_suite \
            "google_robot" \
            "${GOOGLE_GATE_CHECKPOINT}" \
            "${GOOGLE_TARGET_KEEP_RATIO}" \
            "${GOOGLE_TASKS[@]}"
        ;;
    widowx_bridge)
        run_suite \
            "widowx_bridge" \
            "${WIDOWX_GATE_CHECKPOINT}" \
            "${WIDOWX_TARGET_KEEP_RATIO}" \
            "${WIDOWX_TASKS[@]}"
        ;;
    all)
        run_suite \
            "google_robot" \
            "${GOOGLE_GATE_CHECKPOINT}" \
            "${GOOGLE_TARGET_KEEP_RATIO}" \
            "${GOOGLE_TASKS[@]}"

        run_suite \
            "widowx_bridge" \
            "${WIDOWX_GATE_CHECKPOINT}" \
            "${WIDOWX_TARGET_KEEP_RATIO}" \
            "${WIDOWX_TASKS[@]}"
        ;;
esac

echo
echo "============================================================"
echo "[OK] All requested evaluations finished."
echo "Results : ${RESULT_ROOT}"
echo "Summary : ${SUMMARY_CSV}"
echo "Failures: ${FAILURE_LOG}"
echo "============================================================"

column -s, -t "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
