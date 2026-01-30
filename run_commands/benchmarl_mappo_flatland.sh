#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train_benchmarl_flatland.py \
  --algo mappo \
  --seed 1
