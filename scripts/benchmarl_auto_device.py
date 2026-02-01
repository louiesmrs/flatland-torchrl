#!/usr/bin/env python3
import argparse
import os
import subprocess

import torch


def resolve_train_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--", dest="cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_device = args.train_device
    if train_device == "auto":
        train_device = resolve_train_device()

    if not args.cmd:
        raise SystemExit(
            "Provide a benchmarl run command after '--'. Example: scripts/benchmarl_auto_device.py -- benchmarl/run.py task=flatland/phase_1"
        )

    cmd = list(args.cmd)
    cmd.append(f"experiment.train_device={train_device}")
    cmd.append("experiment.sampling_device=cpu")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
