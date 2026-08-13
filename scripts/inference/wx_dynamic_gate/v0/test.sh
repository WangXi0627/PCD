cd /media/hwx/Xixixi/code-vla/PCD

# Rollout collect
python scripts/check_rollout_dataset.py \
  --root /media/hwx/Xixixi/code-vla/PCD/rollouts/pizero \
  --max-episodes 1

# Dynamic gate v0
# 不反向传播的接口测试
export PCD_ROOT="/media/hwx/Xixixi/code-vla/PCD"
export MODEL_ROOT="/media/hwx/Xixixi/code-vla/VLA/PCD"
CHECKPOINT="${MODEL_ROOT}/open_pi_zero"
export PYTHONPATH="${PCD_ROOT}:${PCD_ROOT}/simpler_env/policies/pizero:${PCD_ROOT}/simpler_env/policies/pizero/open_pi_zero:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
python scripts/smoke_test_pizero_dynamic_gate.py \
  --checkpoint-path ${CHECKPOINT} \
  --episode-dir /media/hwx/Xixixi/code-vla/PCD/rollouts/pizero/google_robot_pick_coke_can/episode_000000 \
  --skip-grad-check

python scripts/smoke_test_pizero_dynamic_gate.py \
  --checkpoint-path ${CHECKPOINT} \
  --episode-dir /media/hwx/Xixixi/code-vla/PCD/rollouts/pizero/google_robot_pick_coke_can/episode_000000