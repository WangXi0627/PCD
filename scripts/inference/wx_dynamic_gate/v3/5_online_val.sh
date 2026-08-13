# seed 20-23
export PCD_ROOT=/media/hwx/Xixixi/code-vla/PCD
export MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD

SUITE=google_robot \
GATE_CHECKPOINT="${PCD_ROOT}/checkpoints/dynamic_gate/v1/google_robot_r095/best_total_under_budget.pt" \
EPISODE_START=20 \
N_TRAJS=4 \
GPU_ID=1 \
RUN_NAME=google_best_total_selection \
bash scripts/inference/wx_dynamic_gate/v3/evaluate_v1_all_tasks.sh

SUITE=google_robot \
GATE_CHECKPOINT="${PCD_ROOT}/checkpoints/dynamic_gate/v1/google_robot_r095/best_action_under_budget.pt" \
EPISODE_START=20 \
N_TRAJS=4 \
GPU_ID=1 \
RUN_NAME=google_best_action_selection \
bash scripts/inference/wx_dynamic_gate/v3/evaluate_v1_all_tasks.sh

# 建议选择规则：
# 闭环平均成功率更高；
# 成功率相同时，优先离线sufficiency_loss较低者；
# 如果结果非常接近，优先best_total_under_budget.pt。

export PCD_ROOT=/media/hwx/Xixixi/code-vla/PCD
export MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD

SUITE=google_robot \
GATE_CHECKPOINT="${PCD_ROOT}/checkpoints/dynamic_gate/v1/google_robot_r095/best_total_under_budget.pt" \
GPU_ID=1 \
EPISODE_START=40 \
N_TRAJS=50 \
GOOGLE_TARGET_KEEP_RATIO=0.95 \
bash scripts/inference/wx_dynamic_gate/v3/evaluate_v1_all_tasks.sh


export PCD_ROOT=/media/hwx/Xixixi/code-vla/PCD
export MODEL_ROOT=/media/hwx/Xixixi/code-vla/VLA/PCD

SUITE=widowx_bridge \
GATE_CHECKPOINT="${PCD_ROOT}/checkpoints/dynamic_gate/v1/widowx_bridge_r090/best_total_under_budget.pt" \
GPU_ID=1 \
EPISODE_START=24 \
N_TRAJS=16 \
WIDOWX_TARGET_KEEP_RATIO=0.90 \
bash scripts/inference/wx_dynamic_gate/v3/evaluate_v1_all_tasks.sh