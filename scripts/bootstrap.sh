#!/usr/bin/env bash
set -euo pipefail

uv sync --reinstall

# Build C-utils extension.
uv pip install ./flatland_cutils
