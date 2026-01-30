#!/usr/bin/env bash
set -euo pipefail

# Phase I (50 agents)
uv run python flatland_ppo_training_torchrl.py \
  --num-envs 8 \
  --vf-coef 0.01 \
  --ent-coef 0.01 \
  --max-grad-norm 0.1 \
  --learning-rate 2.5e-4 \
  --clip-coef 0.1 \
  --seed 2 \
  --exp-name "jiang_phase_I_50" \
  --curriculum-path "curriculums/jiang_phase_I_50.json" \
  --value-loss "l2"

# Phase II (50 agents, initialize from Phase I)
PHASE_I_CKPT=$(ls -t model_checkpoints/flatland-rl__jiang_phase_I_50__2__*/flatland-rl__jiang_phase_I_50__2__*.tar | head -n 1)
uv run python flatland_ppo_training_torchrl.py \
  --num-envs 8 \
  --vf-coef 0.01 \
  --ent-coef 0.01 \
  --max-grad-norm 0.1 \
  --learning-rate 2.5e-4 \
  --clip-coef 0.1 \
  --seed 2 \
  --exp-name "jiang_phase_II_50" \
  --pretrained-network-path "$PHASE_I_CKPT" \
  --curriculum-path "curriculums/jiang_phase_II_50.json" \
  --value-loss "l2"

# Phase III-50 (initialize from Phase II)
PHASE_II_CKPT=$(ls -t model_checkpoints/flatland-rl__jiang_phase_II_50__2__*/flatland-rl__jiang_phase_II_50__2__*.tar | head -n 1)
uv run python flatland_ppo_training_torchrl.py \
  --num-envs 8 \
  --vf-coef 0.01 \
  --ent-coef 0.01 \
  --max-grad-norm 0.1 \
  --learning-rate 2.5e-4 \
  --clip-coef 0.1 \
  --seed 2 \
  --exp-name "jiang_phase_III_50" \
  --pretrained-network-path "$PHASE_II_CKPT" \
  --curriculum-path "curriculums/jiang_phase_III_50.json" \
  --value-loss "l2"

