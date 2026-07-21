#!/usr/bin/env bash

set -e

export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
CHECKPOINT="${MODEL_ROOT}/open_pi_zero"

export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl

cd "${PCD_ROOT}"

num_gpus=1
gpu_id=1
n_trajs=20

result_root="./results/collector"
rollout_root="./rollouts/pizero"

tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "google_robot_place_apple_in_closed_top_drawer"
    "widowx_carrot_on_plate"
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_stack_cube"
)

for task in "${tasks[@]}"; do
    echo "=============================="
    echo "[COLLECT] task=${task}"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=${gpu_id} python parallel_inference.py \
        --num-gpus ${num_gpus} \
        --result-root ${result_root} \
        --n-trajs ${n_trajs} \
        --policy pizero \
        --checkpoint ${CHECKPOINT} \
        --task ${task} \
        --collect-rollouts \
        --rollout-root ${rollout_root} \
        --no-save-gif \
        --opts \
            use_torch_compile False

    python scripts/check_rollout_dataset.py \
        --root "${rollout_root}/${task}" \
        --max-episodes ${n_trajs}
done

echo "Done. Rollouts saved to ${rollout_root}"