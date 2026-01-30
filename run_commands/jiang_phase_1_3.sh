#!/usr/bin/env bash
set -euo pipefail

uv run python flatland_ppo_training_torchrl.py \
  --num-envs 10 \
  --num-steps 200 \
  --vf-coef 0.01 \
  --ent-coef 0.01 \
  --max-grad-norm 0.2 \
  --learning-rate 2.5e-6 \
  --clip-coef 0.1 \
  --seed 1 \
  --exp-name "jiang_phase_1_3_7_to_10_agents" \
  --curriculum-path "curriculums/jiang_phases_1_3_7_to_10_agents_30x30.json" \
  --value-loss "l2"
