#!/usr/bin/env bash
set -euo pipefail

uv run python flatland_ppo_training_torchrl.py \
  --num-envs 4 \
  --num-steps 100 \
  --vf-coef 0.01 \
  --ent-coef 0.01 \
  --max-grad-norm 0.2 \
  --learning-rate 2.5e-6 \
  --clip-coef 0.1 \
  --seed 1 \
  --exp-name "jiang_phase_1_3_ci_2_to_5_agents" \
  --curriculum-path "curriculums/jiang_phases_1_3_ci_2_to_5_agents_30x30.json" \
  --value-loss "l2"
