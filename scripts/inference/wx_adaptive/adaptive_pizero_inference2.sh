#!/usr/bin/env bash

set -e

# =========================
# Basic paths
# =========================
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PCD_ROOT="/data/Xixixi/VLA/PCD"
export MODEL_ROOT="/data/Xixixi/Model/PCD"
CHECKPOINT="${MODEL_ROOT}/open_pi_zero"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl

cd "${PCD_ROOT}"

# =========================
# Experiment settings
# =========================
num_gpus=1
gpu_id=0

# adaptive 会对每个候选 mask 都 forward 一次，速度大约是 K 倍慢
# 建议先 20 或 50，确认有效后再跑 100
n_trajs=100

policy="pizero"

tasks=(
    # "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "widowx_carrot_on_plate"
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

# 你已经验证过 multi_modal_projector 的 0.9 / 0.75 比较关键
keep_ratios=(
    # "0.9"
    "0.75"
)

# 候选 mask seed 集合；adaptive 内部会从这些候选里选
adaptive_mask_seeds="0,1,2,3,4,5,6,7,8,9"

adaptive_num_candidates=10

# 输出目录
adaptive_result_root="./results/adaptive/pizero"

echo "=============================="
echo "Running PiZero adaptive mask selection"
echo "n_trajs=${n_trajs}"
echo "tasks=${tasks[*]}"
echo "keep_ratios=${keep_ratios[*]}"
echo "adaptive_mask_seeds=${adaptive_mask_seeds}"
echo "mask_target=multi_modal_projector"
echo "use_torch_compile=False"
echo "=============================="

for task in "${tasks[@]}"; do
    for keep_ratio in "${keep_ratios[@]}"; do
        echo "[ADAPTIVE] task=${task}, keep_ratio=${keep_ratio}, seeds=${adaptive_mask_seeds}"

        CUDA_VISIBLE_DEVICES=${gpu_id} python parallel_inference.py \
            --num-gpus ${num_gpus} \
            --result-root ${adaptive_result_root} \
            --n-trajs ${n_trajs} \
            --policy ${policy} \
            --checkpoint ${CHECKPOINT} \
            --task ${task} \
            --run-name adaptive_mp_kr${keep_ratio}_K${adaptive_num_candidates} \
            --no-timestamp \
            --opts \
                adaptive_feature_mask True \
                adaptive_mask_target multi_modal_projector \
                adaptive_mask_keep_ratio ${keep_ratio} \
                adaptive_mask_seeds ${adaptive_mask_seeds} \
                adaptive_num_candidates ${adaptive_num_candidates} \
                adaptive_mask_rescale True \
                adaptive_include_nomask False \
                adaptive_score_mode consensus \
                adaptive_consensus_weight 1.0 \
                adaptive_temporal_weight 0.0 \
                adaptive_norm_weight 0.0 \
                adaptive_verbose False \
                use_torch_compile False
    done
done

echo "=============================="
echo "Done."
echo "Adaptive results: ${adaptive_result_root}"
echo "=============================="