import argparse
import json
import subprocess
import time
from pathlib import Path

from carbs import CARBS, CARBSParams, LogSpace, LinearSpace, Param, ObservationInParam
from tensorboard.backend.event_processing import event_accumulator


def run_training(params: dict) -> str:
    exp_name = f"carbs_{int(time.time())}"
    cmd = [
        "uv",
        "run",
        "python",
        "flatland_ppo_training_torchrl.py",
        "--exp-name",
        exp_name,
        "--num-envs",
        str(int(params["num-envs"])),
        "--num-steps",
        str(int(params["num-steps"])),
        "--vf-coef",
        str(params["vf-coef"]),
        "--ent-coef",
        str(params["ent-coef"]),
        "--max-grad-norm",
        str(params["max-grad-norm"]),
        "--learning-rate",
        str(params["learning-rate"]),
        "--clip-coef",
        str(params["clip-coef"]),
        "--seed",
        str(int(params["seed"])),
        "--curriculum-path",
        params["curriculum-path"],
    ]
    subprocess.run(cmd, check=True)
    return exp_name


def find_latest_run_dir(exp_name: str, seed: int) -> Path:
    pattern = f"flatland-rl__{exp_name}__{seed}__*"
    matches = sorted(Path("runs").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return matches[-1]


def read_tb_scalar(run_dir: Path, tag: str) -> float:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files in {run_dir}")

    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag '{tag}' not found in TensorBoard logs")
    scalars = ea.Scalars(tag)
    return scalars[-1].value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric", default="stats/arrival_ratio")
    parser.add_argument(
        "--curriculum-path",
        default="curriculums/jiang_sweep_2_agents_30x30.json",
    )
    args = parser.parse_args()

    param_spaces = [
        Param("learning-rate", LogSpace(min=2.5e-6, max=2.5e-3), search_center=2.5e-5),
        Param("clip-coef", LinearSpace(min=0.05, max=0.3), search_center=0.1),
        Param("vf-coef", LogSpace(min=0.01, max=1.0), search_center=0.1),
        Param("ent-coef", LogSpace(min=1e-4, max=1e-2), search_center=1e-3),
    ]
    carbs = CARBS(
        CARBSParams(better_direction_sign=1, resample_frequency=0),
        param_spaces,
    )

    for _ in range(args.trials):
        suggestion = carbs.suggest().suggestion
        params = {
            "learning-rate": suggestion["learning-rate"],
            "clip-coef": suggestion["clip-coef"],
            "vf-coef": suggestion["vf-coef"],
            "ent-coef": suggestion["ent-coef"],
            "max-grad-norm": 0.2,
            "num-envs": 8,
            "num-steps": 200,
            "seed": args.seed,
            "curriculum-path": args.curriculum_path,
        }

        exp_name = run_training(params)
        run_dir = find_latest_run_dir(exp_name, args.seed)
        metric_value = read_tb_scalar(run_dir, args.metric)

        carbs.observe(
            ObservationInParam(
                input=suggestion,
                output=metric_value,
                cost=1,
            )
        )

        result = {
            "suggestion": suggestion,
            "metric": metric_value,
        }
        results_path = Path("carbs_runs") / "results.jsonl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
