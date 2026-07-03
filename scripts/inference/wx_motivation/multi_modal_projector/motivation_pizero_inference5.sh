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

cd "${PCD_ROOT}"

# =========================
# Experiment settings
# =========================
num_gpus=1
gpu_id=0

# 动机实验先别跑 300，先跑 50 看现象
n_trajs=50

policy="pizero"

tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "widowx_carrot_on_plate"
    # "widowx_spoon_on_towel"
    # "widowx_put_eggplant_in_basket"
    # "widowx_stack_cube"
    # "google_robot_place_apple_in_closed_top_drawer"
)

keep_ratios=(
    # "0.5"
    # "0.6"
    # "0.7"
    # "0.8"
    # "0.85"
    # "0.9"
    "0.95"
)

# 10 个随机 mask seed，对应图上的 10 个点
mask_seeds=(
    0 1 2 3 4 5 6 7 8 9
)

# 输出目录
base_result_root="./results/motivation/pi0_base"
mask_result_root="./results/motivation/pi0_random_mask"

# # =========================
# # Run baseline
# # =========================
# echo "=============================="
# echo "Running PiZero motivation baseline"
# echo "n_trajs=${n_trajs}"
# echo "use_torch_compile=False"
# echo "=============================="

# for task in "${tasks[@]}"; do
#     echo "[BASE] task=${task}"

#     CUDA_VISIBLE_DEVICES=${gpu_id} python parallel_inference.py \
#         --num-gpus ${num_gpus} \
#         --result-root ${base_result_root} \
#         --n-trajs ${n_trajs} \
#         --policy ${policy} \
#         --checkpoint ${CHECKPOINT} \
#         --task ${task} \
#         --opts use_torch_compile False
# done

# =========================
# Run random feature masks
# =========================
echo "=============================="
echo "Running PiZero random feature mask motivation experiment"
echo "n_trajs=${n_trajs}"
echo "mask_seeds=${mask_seeds[*]}"
echo "keep_ratios=${keep_ratios[*]}"
echo "=============================="

for task in "${tasks[@]}"; do
    for keep_ratio in "${keep_ratios[@]}"; do
        for seed in "${mask_seeds[@]}"; do
            echo "[MASK] task=${task}, keep_ratio=${keep_ratio}, mask_seed=${seed}"

            CUDA_VISIBLE_DEVICES=${gpu_id} python parallel_inference.py \
                --num-gpus ${num_gpus} \
                --result-root ${mask_result_root} \
                --n-trajs ${n_trajs} \
                --policy ${policy} \
                --checkpoint ${CHECKPOINT} \
                --task ${task} \
                --opts \
                    random_feature_mask True \
                    mask_keep_ratio ${keep_ratio} \
                    mask_target multi_modal_projector \
                    mask_seed ${seed} \
                    mask_rescale True \
                    mask_verbose True \
                    use_torch_compile False
        done
    done
done

echo "=============================="
echo "Done."
echo "Base results: ${base_result_root}"
echo "Mask results: ${mask_result_root}"
echo "=============================="