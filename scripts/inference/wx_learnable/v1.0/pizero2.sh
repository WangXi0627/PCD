#!/usr/bin/env bash

set -e

# =========================
# Basic paths
# =========================
export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
CHECKPOINT="${MODEL_ROOT}/open_pi_zero"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl

cd "${PCD_ROOT}"

# =========================
# Experiment settings
# =========================
num_gpus=1
gpu_id=1

n_trajs=50
policy="pizero"

tasks=(
    # "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    "widowx_carrot_on_plate"
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

keep_ratios=(
    "0.75"
    # "0.9"
)

# =========================
# Learnable mask settings
# =========================
# 有 early stop 后，这个是“最大优化步数”
learnable_opt_steps=10
learnable_lr=0.1

learnable_early_stop=True
learnable_min_opt_steps=2
learnable_loss_tol=1e-4
learnable_patience=2

learnable_anchor_weight=0.5
learnable_keep_weight=10.0
learnable_binary_weight=0.0
learnable_l2_weight=0.0

result_root="./results/learnable/pizero"

echo "=============================="
echo "Running PiZero test-time learnable mask"
echo "n_trajs=${n_trajs}"
echo "tasks=${tasks[*]}"
echo "keep_ratios=${keep_ratios[*]}"
echo "max_opt_steps=${learnable_opt_steps}"
echo "lr=${learnable_lr}"
echo "early_stop=${learnable_early_stop}"
echo "min_opt_steps=${learnable_min_opt_steps}"
echo "loss_tol=${learnable_loss_tol}"
echo "patience=${learnable_patience}"
echo "mask_target=multi_modal_projector"
echo "=============================="

for task in "${tasks[@]}"; do
    for keep_ratio in "${keep_ratios[@]}"; do
        echo "[LEARNABLE] task=${task}, keep_ratio=${keep_ratio}"

        CUDA_VISIBLE_DEVICES=${gpu_id} python parallel_inference.py \
            --num-gpus ${num_gpus} \
            --result-root ${result_root} \
            --n-trajs ${n_trajs} \
            --policy ${policy} \
            --checkpoint ${CHECKPOINT} \
            --task ${task} \
            --run-name learnable_mp_kr${keep_ratio}_maxstep${learnable_opt_steps}_es${learnable_early_stop}_lr${learnable_lr}_aw${learnable_anchor_weight} \
            --no-timestamp \
            --opts \
                learnable_feature_mask True \
                learnable_mask_target multi_modal_projector \
                learnable_target_keep_ratio ${keep_ratio} \
                learnable_mask_temperature 1.0 \
                learnable_mask_rescale True \
                learnable_opt_steps ${learnable_opt_steps} \
                learnable_lr ${learnable_lr} \
                learnable_anchor_weight ${learnable_anchor_weight} \
                learnable_keep_weight ${learnable_keep_weight} \
                learnable_binary_weight ${learnable_binary_weight} \
                learnable_l2_weight ${learnable_l2_weight} \
                learnable_reset_each_episode True \
                learnable_hard_topk_eval False \
                learnable_early_stop ${learnable_early_stop} \
                learnable_min_opt_steps ${learnable_min_opt_steps} \
                learnable_loss_tol ${learnable_loss_tol} \
                learnable_patience ${learnable_patience} \
                learnable_verbose True \
                use_torch_compile False
    done
done

echo "=============================="
echo "Done."
echo "Results: ${result_root}"
echo "=============================="