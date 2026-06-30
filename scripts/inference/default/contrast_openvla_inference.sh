# wx
PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd ${PCD_ROOT}
# wx

num_gpus=1
n_trajs=300
result_root="./results/default/contrast"

# search_opts="by point_tracking,box_tracking,grounded_sam_tracking alpha 0.8"
search_opts="by grounded_sam_tracking alpha 0.8"

policies=("openvla")
checkpoints=("/media/hwx/Xixixi/code-vla/VLA/PCD/openvla-7b")

tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_carrot_on_plate"
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_stack_cube"
    "google_robot_place_apple_in_closed_top_drawer"
)

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        CUDA_VISIBLE_DEVICES=3 python parallel_inference.py \
            --contrast \
            --n-trajs $n_trajs \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task \
            --search-opts $search_opts
    done
done

