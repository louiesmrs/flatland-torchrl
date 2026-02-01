#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

from pufferlib.sweep import Protein
from tensorboard.backend.event_processing import event_accumulator

SWEEP_CFG = {
    "method": "Protein",
    "metric": "stats/arrival_ratio",
    "metric_distribution": "raw",
    "goal": "maximize",
    "downsample": 1,
    "early_stop_quantile": 0.3,
    # Hyperparams to sweep are declared as nested dicts in the pufferlib format
    "learning_rate": {
        "distribution": "log_normal",
        "min": 2.5e-6,
        "max": 2.5e-3,
        "mean": 2.5e-5,
        "scale": "auto",
    },
    "clip_coef": {
        "distribution": "uniform",
        "min": 0.05,
        "max": 0.3,
        "mean": 0.1,
        "scale": "auto",
    },
    "vf_coef": {
        "distribution": "log_normal",
        "min": 0.01,
        "max": 1.0,
        "mean": 0.1,
        "scale": "auto",
    },
    "ent_coef": {
        "distribution": "log_normal",
        "min": 1e-4,
        "max": 1e-2,
        "mean": 1e-3,
        "scale": "auto",
    },
}

# base args that will be mutated by the sweep.suggest call
BASE_ARGS = {
    "sweep": SWEEP_CFG,
    "train": {
        "num_envs": 10,
        "num_steps": 200,
        "total_timesteps": 1000000,
    },
    "data_path": "sweep",
}


def run_training_with_suggestion(suggestion_args, use_gpu=False):
    train = suggestion_args.get("train", {})
    # map fields to CLI
    exp_name = f"puffer_{int(time.time())}"
    cmd = [
        "uv",
        "run",
        "python",
        "flatland_ppo_training_torchrl.py",
        "--exp-name",
        exp_name,
        "--num-envs",
        str(int(train.get("num_envs", 10))),
        "--num-steps",
        str(int(train.get("num_steps", 200))),
        "--seed",
        str(int(time.time()) & 0xFFFFFFFF),
        "--curriculum-path",
        "curriculums/jiang_sweep_2_agents_30x30.json",
        "--learning-rate",
        str(train.get("learning_rate", 2.5e-5)),
        "--clip-coef",
        str(train.get("clip_coef", 0.1)),
        "--vf-coef",
        str(train.get("vf_coef", 0.1)),
        "--ent-coef",
        str(train.get("ent_coef", 1e-3)),
    ]

    # append GPU flag when requested
    if use_gpu:
        cmd.append("--cuda")

    subprocess.run(cmd, check=True)
    # find produced run dir
    matches = sorted(Path("runs").glob(f"flatland-rl__{exp_name}__*"))
    if not matches:
        raise FileNotFoundError("no run dir found for exp: " + exp_name)
    run_dir = matches[-1]

    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    tag = "stats/arrival_ratio"
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag '{tag}' not found in TensorBoard logs for {run_dir}")
    scalars = ea.Scalars(tag)
    return scalars[-1].value


def main(trials=20, use_gpu=False):
    cfg = dict(SWEEP_CFG)
    # ensure the sweep config reflects the requested GPU usage
    cfg["use_gpu"] = use_gpu

    # Directly construct the Protein sweep using the provided config
    sweep = Protein(cfg, expansion_rate=1.0, use_gpu=use_gpu)

    Path("puffer_runs").mkdir(parents=True, exist_ok=True)

    for i in range(trials):
        args = json.loads(json.dumps(BASE_ARGS))  # deep copy
        suggestion, info = sweep.suggest(args)
        score = run_training_with_suggestion(suggestion, use_gpu=use_gpu)
        cost = suggestion.get("train", {}).get("total_timesteps", 1)
        sweep.observe(suggestion, score, cost)

        with open(Path("puffer_runs") / "results.jsonl", "a") as f:
            f.write(
                json.dumps({"suggestion": suggestion, "score": score, "cost": cost})
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()
    main(trials=args.trials, use_gpu=args.use_gpu)
