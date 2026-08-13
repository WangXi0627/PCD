CUDA_VISIBLE_DEVICES=1 \
python scripts/evaluate_dynamic_gate_v1_offline.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --gate-checkpoint checkpoints/dynamic_gate/v1/google_robot_r090/best_total_under_budget.pt \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --policy-setup google_robot \
  --flow-sampling beta \
  --split validation \
  --batch-size 1 \
  --output-json checkpoints/dynamic_gate/v1/google_robot_r090/offline_best_total.json

CUDA_VISIBLE_DEVICES=1 \
python scripts/evaluate_dynamic_gate_v1_offline.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --gate-checkpoint checkpoints/dynamic_gate/v1/google_robot_r090/best_action_under_budget.pt \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --policy-setup google_robot \
  --flow-sampling beta \
  --split validation \
  --batch-size 1 \
  --output-json checkpoints/dynamic_gate/v1/google_robot_r090/offline_best_action.json

CUDA_VISIBLE_DEVICES=1 \
python scripts/evaluate_dynamic_gate_v1_offline.py \
  --manifest manifests/dynamic_gate/google_robot.json \
  --teacher-cache teacher_cache/dynamic_gate/google_robot/validation \
  --gate-checkpoint checkpoints/dynamic_gate/v1/google_robot_r090/last.pt \
  --checkpoint-path "${MODEL_ROOT}/open_pi_zero" \
  --policy-setup google_robot \
  --flow-sampling beta \
  --split validation \
  --batch-size 1 \
  --output-json checkpoints/dynamic_gate/v1/google_robot_r090/offline_last.json