#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import torch


SMOKE_TASKS = [
    "flatland/smoke_2_agents",
]


def resolve_train_device(train_device: str) -> str:
    if train_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        return "cpu"
    if train_device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return train_device


def run_smoke(task_name: str, train_device: str):
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarl_ext/benchmarl/run.py",
        f"task={task_name}",
        "algorithm=mappo",
        "model=layers/mlp",
        "model@critic_model=layers/mlp",
        f"experiment.train_device={train_device}",
        "experiment.sampling_device=cpu",
        "experiment.loggers=[tensorboard]",
        "experiment.create_json=false",
        "experiment.checkpoint_interval=0",
        "experiment.max_n_frames=2000",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default=".")
    args = parser.parse_args()

    train_device = resolve_train_device(args.train_device)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for task_name in SMOKE_TASKS:
        run_smoke(task_name, train_device)
