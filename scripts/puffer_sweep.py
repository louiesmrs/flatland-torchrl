#!/usr/bin/env python3
import argparse
import copy
import pickle
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pufferlib.sweep import Protein
from tensorboard.backend.event_processing import event_accumulator

DEFAULT_CONFIG = "scripts/flatland_sweep.yaml"


def _load_tensorboard_scalar(run_dir: Path, tag: str) -> float:
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag '{tag}' not found in TensorBoard logs for {run_dir}")
    scalars = ea.Scalars(tag)
    return scalars[-1].value


def _flatten_overrides(overrides, prefix=""):
    items = {}
    for key, value in overrides.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(_flatten_overrides(value, full_key))
        else:
            items[full_key] = value
    return items


def run_benchmarl_with_suggestion(benchmarl_cfg, use_gpu=False):
    train_device = "cuda" if use_gpu else benchmarl_cfg.get("train_device", "cpu")
    sampling_device = "cuda" if use_gpu else benchmarl_cfg.get("sampling_device", "cpu")

    loggers = ",".join(benchmarl_cfg.get("loggers", ["tensorboard"]))
    cmd = [
        "uv",
        "run",
        "python",
        "benchmarl_ext/benchmarl/run.py",
        f"task={benchmarl_cfg.get('task', 'flatland/phase_1')}",
        f"algorithm={benchmarl_cfg.get('algorithm', 'ippo')}",
        f"model={benchmarl_cfg.get('model', 'layers/flatland_treelstm')}",
        f"model@critic_model={benchmarl_cfg.get('critic_model', 'layers/flatland_treelstm_critic')}",
        f"experiment.train_device={train_device}",
        f"experiment.sampling_device={sampling_device}",
        f"experiment.checkpoint_interval={benchmarl_cfg.get('checkpoint_interval', 120000)}",
        f"experiment.checkpoint_at_end={str(benchmarl_cfg.get('checkpoint_at_end', True)).lower()}",
        f"experiment.loggers=[{loggers}]",
        f"experiment.create_json={str(benchmarl_cfg.get('create_json', False)).lower()}",
        "experiment.project_name=flatland",
        f"seed={benchmarl_cfg.get('seed', 0)}",
    ]

    overrides = _flatten_overrides(benchmarl_cfg.get("overrides", {}))
    for key, value in overrides.items():
        cmd.append(f"{key}={value}")

    subprocess.run(cmd, check=True)

    output_root = Path(".").resolve()
    experiments = sorted(output_root.rglob("ippo_*"))
    experiments = [p for p in experiments if p.is_dir()]
    if not experiments:
        raise FileNotFoundError("No BenchMARL output folder found after run.")
    run_dir = max(experiments, key=lambda p: p.stat().st_mtime)

    return _load_tensorboard_scalar(run_dir, tag="collection_info_arrival_ratio_mean")


def run_training_with_suggestion(suggestion_args, use_gpu=False):
    benchmarl_cfg = suggestion_args.get("benchmarl")
    if benchmarl_cfg:
        return run_benchmarl_with_suggestion(benchmarl_cfg, use_gpu=use_gpu)

    train = suggestion_args.get("train", {})
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
        suggestion_args.get(
            "curriculum_path", "curriculums/jiang_sweep_2_agents_30x30.json"
        ),
        "--learning-rate",
        str(train.get("learning_rate", 2.5e-5)),
        "--clip-coef",
        str(train.get("clip_coef", 0.1)),
        "--vf-coef",
        str(train.get("vf_coef", 0.1)),
        "--ent-coef",
        str(train.get("ent_coef", 1e-3)),
    ]

    if use_gpu:
        cmd.append("--cuda")

    subprocess.run(cmd, check=True)

    matches = sorted(Path("runs").glob(f"flatland-rl__{exp_name}__*"))
    if not matches:
        raise FileNotFoundError("no run dir found for exp: " + exp_name)
    run_dir = matches[-1]

    return _load_tensorboard_scalar(run_dir, tag="stats/arrival_ratio")


def sweep_from_config(config_path, use_gpu=False):
    with open(config_path) as f:
        args = yaml.safe_load(f)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_file = f"puffer_runs/sweep_observations_flatland_{timestamp}.pkl"
    print(f"Sweep observations will be saved to: {obs_file}")

    sweep_cfg = copy.deepcopy(args["sweep"])
    sweep_manager = Protein(sweep_cfg, **args.get("sweep_extra", {}))

    Path("puffer_runs").mkdir(parents=True, exist_ok=True)

    max_runs = int(args.get("max_runs", 20))
    suggest_history = []
    orig_args = copy.deepcopy(args)

    for i in range(max_runs):
        print(f"\n--- Starting sweep run {i + 1}/{max_runs} ---")

        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        suggest_start_time = time.time()
        run_args, info = sweep_manager.suggest(args)
        suggest_time = time.time() - suggest_start_time
        print(f"sweep_manager.suggest() took {suggest_time:.4f} seconds")
        suggest_history.append(
            {
                "run_args": copy.deepcopy(run_args),
                "info": info,
                "suggest_time": suggest_time,
                "run_index": i,
            }
        )

        try:
            score = run_training_with_suggestion(run_args, use_gpu=use_gpu)
        except Exception as e:
            print(f"Run {i + 1} failed: {e}")
            sweep_manager.observe(run_args, 0, 0, is_failure=True)
            continue

        cost = run_args.get("train", {}).get("total_timesteps", 1)
        sweep_manager.observe(run_args, score, cost)

        if (i + 1) % 10 == 0 or (i + 1) >= max_runs:
            print(f"\n--- Saving sweep observations to {obs_file} (run {i + 1}) ---")
            with open(obs_file, "wb") as f:
                pickle.dump(
                    {
                        "success": sweep_manager.success_observations,
                        "failure": sweep_manager.failure_observations,
                        "suggest_history": suggest_history,
                        "total_sweep_time": time.time() - start_time,
                        "args": orig_args,
                    },
                    f,
                )

    total_sweep_time = time.time() - start_time
    print(f"\n--- Total sweep time: {total_sweep_time:.2f} seconds ---")
    total_suggest_time = sum(h["suggest_time"] for h in suggest_history)
    print(f"--- Total suggest time: {total_suggest_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    sweep_from_config(args.config, use_gpu=args.use_gpu)
