#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import torch


PHASE_TASKS = [
    "flatland/phase_1",
    "flatland/phase_3",
    "flatland/phase_7",
    "flatland/phase_10",
]


def resolve_devices(train_device: str | None) -> tuple[str, str]:
    resolved_train = train_device or "cpu"

    if train_device == "auto":
        if torch.cuda.is_available():
            resolved_train = "cuda"
        elif torch.backends.mps.is_available():
            resolved_train = "mps"
        else:
            resolved_train = "cpu"

    if resolved_train == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return resolved_train, "cpu"


def run_phase(task_name, restore_file=None, train_device="cpu", sampling_device="cpu"):
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarl_ext/benchmarl/run.py",
        f"task={task_name}",
        "algorithm=ippo",
        "model=layers/flatland_treelstm",
        "model@critic_model=layers/flatland_treelstm_critic",
        f"experiment.train_device={train_device}",
        f"experiment.sampling_device={sampling_device}",
        "experiment.loggers=[tensorboard]",
        "experiment.create_json=false",
        "experiment.checkpoint_interval=120000",
        "experiment.checkpoint_at_end=true",
        "experiment.project_name=flatland",
    ]

    if restore_file:
        cmd.append(f"experiment.restore_file={restore_file}")

    subprocess.run(cmd, check=True)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if output_dir is None:
        return None
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def find_latest_experiment(output_root: Path) -> Path | None:
    candidates = [
        p
        for p in output_root.rglob("ippo_*")
        if p.is_dir() and (p / "checkpoints").exists()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--sampling-device", type=str, default="cpu")
    parser.add_argument("--output-root", type=str, default=".")
    parser.add_argument("--phase-multirun", action="store_true")
    parser.add_argument("--start-phase", type=int, default=1)
    parser.add_argument("--restore-file", type=str, default=None)
    args = parser.parse_args()

    restore_file = args.restore_file
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_device, sampling_device = resolve_devices(args.train_device)

    last_experiment = None
    for phase_idx, task_name in enumerate(PHASE_TASKS, start=1):
        if phase_idx < args.start_phase:
            continue

        if args.phase_multirun:
            phase_cmd = [
                "uv",
                "run",
                "python",
                "benchmarl_ext/benchmarl/run.py",
                f"task={task_name}",
                "algorithm=ippo",
                "model=layers/flatland_treelstm",
                "model@critic_model=layers/flatland_treelstm_critic",
                f"experiment.train_device={train_device}",
                f"experiment.sampling_device={sampling_device}",
                "experiment.loggers=[tensorboard]",
                "experiment.create_json=false",
                "experiment.checkpoint_interval=120000",
                "experiment.checkpoint_at_end=true",
                "experiment.project_name=flatland",
                "-m",
            ]
            if restore_file:
                phase_cmd.append(f"experiment.restore_file={restore_file}")
            subprocess.run(phase_cmd, check=True)
        else:
            run_phase(
                task_name,
                restore_file=restore_file,
                train_device=train_device,
                sampling_device=sampling_device,
            )

        latest_experiment = find_latest_experiment(output_root)
        if latest_experiment is None:
            raise RuntimeError(
                "Could not locate BenchMARL output folder for checkpointing."
            )
        if last_experiment is not None and latest_experiment == last_experiment:
            raise RuntimeError("No new BenchMARL output folder found after phase run.")
        last_experiment = latest_experiment

        latest_checkpoint = find_latest_checkpoint(latest_experiment)
        if latest_checkpoint is None:
            raise RuntimeError("No checkpoint found for phase run.")
        restore_file = str(latest_checkpoint)
