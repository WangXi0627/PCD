export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# wx:复现
export PCD_ROOT="/data/Xixixi/VLA/PCD"
export MODEL_ROOT="/data/Xixixi/Model/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd ${PCD_ROOT}
# wx:复现

num_gpus=1
n_trajs=100     # wx:复现
result_root="./results/default-100/baseline"

policies=("pizero")
checkpoints=("${MODEL_ROOT}/open_pi_zero")

tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        CUDA_VISIBLE_DEVICES=0 python parallel_inference.py \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --n-trajs $n_trajs \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task
    done
done
