#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/carbs_sweep.py \
  --trials 10 \
  --seed 1 \
  --metric "stats/arrival_ratio" \
  --curriculum-path "curriculums/jiang_sweep_2_agents_30x30.json"
