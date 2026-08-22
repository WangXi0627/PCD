#!/usr/bin/env bash
set -euo pipefail

export PCD_ROOT="/data/Xixixi/VLA/PCD"
export MODEL_ROOT="/data/Xixixi/Model/PCD"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

cd "${PCD_ROOT}"

gpu_id=0
num_gpus=1
n_trajs=50

result_root="./results/motivation-${n_trajs}/random_mask"

policy="pizero"
checkpoint="${MODEL_ROOT}/open_pi_zero"

# Mask设置
mask_target="multi_modal_projector"
num_mask_seeds=2        # baseline keep_ratio=1.0 为了验证确定性用2个seed
master_seed=20260817

tasks=(
    # "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_put_eggplant_in_basket"
    # "widowx_spoon_on_towel"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

keep_ratios=(
    1.0
)

# 使用 master_seed 生成可复现的不重复 mask seeds
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

echo "============================================================"
echo "PiZero random feature-mask experiment"
echo "GPU=${gpu_id}"
echo "n_trajs=${n_trajs}"
echo "mask_target=${mask_target}"
echo "master_seed=${master_seed}"
echo "mask_seeds=${mask_seeds[*]}"
echo "keep_ratios=${keep_ratios[*]}"
echo "result_root=${result_root}"
echo "============================================================"

for task in "${tasks[@]}"; do
    for keep_ratio in "${keep_ratios[@]}"; do
        for mask_seed in "${mask_seeds[@]}"; do

            exp_name="target-${mask_target}/master_seed-${master_seed}/keep_ratio-${keep_ratio}/mask_seed-${mask_seed}"

            echo "------------------------------------------------------------"
            echo "[MASK]"
            echo "task=${task}"
            echo "mask_target=${mask_target}"
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
                --random_mask
                --opts
                mask_keep_ratio "${keep_ratio}"
                mask_seed "${mask_seed}"
                mask_target "${mask_target}"
                mask_verbose False
            )

            CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}"
        done
    done
done

echo "============================================================"
echo "Done."
echo "Results: ${result_root}"
echo "mask_target: ${mask_target}"
echo "master_seed: ${master_seed}"
echo "mask_seeds: ${mask_seeds[*]}"
echo "============================================================"