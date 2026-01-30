import argparse
from pathlib import Path

import torch

from benchmarl.environments.flatland.common import FlatlandTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algo", choices=["mappo", "ippo"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=10)
    args = parser.parse_args()

    task = FlatlandTask.PHASE_1_3_7_TO_10_AGENTS.get_from_yaml()
    task.config["num_agents"] = args.num_agents

    if args.algo == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
    else:
        algorithm_config = IppoConfig.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.evaluation = True
    experiment_config.evaluation_episodes = 5
    experiment_config.render = False
    experiment_config.loggers = []

    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )

    checkpoint_path = Path(args.checkpoint)
    experiment.config.restore_file = str(checkpoint_path)

    experiment.evaluate()


if __name__ == "__main__":
    main()
