#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Environment
# ============================================================

export PCD_ROOT="/data/Xixixi/VLA/PCD"
export MODEL_ROOT="/data/Xixixi/Model/PCD"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

cd "${PCD_ROOT}"

# ============================================================
# Basic experiment settings
# ============================================================

gpu_id=0
num_gpus=1
n_trajs=50

policy="pizero"
checkpoint="${MODEL_ROOT}/open_pi_zero"

# 与 channel-mask 结果完全分开
result_root="./results/motivation-${n_trajs}/random_token_mask"

# ============================================================
# Random token-mask settings
# ============================================================

token_mask_target="multi_modal_projector"
num_mask_seeds=10
master_seed=20260816

# ------------------------------------------------------------
# token_mask_mode:
#
# mask:
#   随机 token 置零，不进行范数补偿。
#
# norm_preserve:
#   随机 token 置零，然后恢复每个样本的全局 Frobenius norm。
#
# scale_only:
#   不删除 token，只将完整特征缩放到普通 mask 后的全局范数。
# ------------------------------------------------------------

# 第一轮主实验只运行原始 token mask
token_mask_modes=(
    "mask"
)

# ============================================================
# Tasks
# ============================================================

tasks=(
    "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "google_robot_place_apple_in_closed_top_drawer"

    # "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    # "widowx_carrot_on_plate"
    # "widowx_stack_cube"
)

# ============================================================
# Keep ratios
# ============================================================

keep_ratios=(
    0.9
    # 0.75
    # 0.5
)

# ============================================================
# Generate reproducible mask seeds
# ============================================================

mapfile -t mask_seeds < <(
    python - "${master_seed}" "${num_mask_seeds}" <<'PY'
import random
import sys

master_seed = int(sys.argv[1])
num_mask_seeds = int(sys.argv[2])

rng = random.Random(master_seed)

for seed in rng.sample(range(1, 1_000_000), num_mask_seeds):
    print(seed)
PY
)

# ============================================================
# Print configuration
# ============================================================

echo "============================================================"
echo "PiZero random visual-token mask experiment"
echo "GPU=${gpu_id}"
echo "num_gpus=${num_gpus}"
echo "n_trajs=${n_trajs}"
echo "policy=${policy}"
echo "checkpoint=${checkpoint}"
echo "result_root=${result_root}"
echo "token_mask_target=${token_mask_target}"
echo "token_mask_modes=${token_mask_modes[*]}"
echo "master_seed=${master_seed}"
echo "num_mask_seeds=${num_mask_seeds}"
echo "mask_seeds=${mask_seeds[*]}"
echo "keep_ratios=${keep_ratios[*]}"
echo "tasks=${tasks[*]}"
echo "============================================================"

# ============================================================
# Run token-mask experiments
# ============================================================

for task in "${tasks[@]}"; do
    for token_mask_mode in "${token_mask_modes[@]}"; do
        for keep_ratio in "${keep_ratios[@]}"; do
            for mask_seed in "${mask_seeds[@]}"; do

                exp_name="mask_mode-${token_mask_mode}/target-${token_mask_target}/master_seed-${master_seed}/keep_ratio-${keep_ratio}/mask_seed-${mask_seed}"

                echo "------------------------------------------------------------"
                echo "[RANDOM TOKEN MASK]"
                echo "task=${task}"
                echo "mode=${token_mask_mode}"
                echo "target=${token_mask_target}"
                echo "master_seed=${master_seed}"
                echo "keep_ratio=${keep_ratio}"
                echo "mask_seed=${mask_seed}"
                echo "exp_name=${exp_name}"
                echo "------------------------------------------------------------"

                command=(
                    python parallel_inference.py
                    --num-gpus "${num_gpus}"
                    --result-root "${result_root}"
                    --exp_name "${exp_name}"
                    --n-trajs "${n_trajs}"
                    --policy "${policy}"
                    --checkpoint "${checkpoint}"
                    --task "${task}"
                    --no-save-gif
                    --random_token_mask
                    --opts
                    token_mask_keep_ratio "${keep_ratio}"
                    token_mask_seed "${mask_seed}"
                    token_mask_mode "${token_mask_mode}"
                    token_mask_target "${token_mask_target}"
                    token_mask_verbose False
                )

                CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}"
            done
        done
    done
done

# ============================================================
# Done
# ============================================================

echo "============================================================"
echo "Random token-mask experiment completed."
echo "Results: ${result_root}"
echo "Target: ${token_mask_target}"
echo "Modes: ${token_mask_modes[*]}"
echo "Master seed: ${master_seed}"
echo "Mask seeds: ${mask_seeds[*]}"
echo "Tasks: ${tasks[*]}"
echo "============================================================"