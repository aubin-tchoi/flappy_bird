import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deep_rl.environments.flappy_bird import FlappyBird
from tqdm import trange

from src import infer_parameters, TreeBasedAgent, checkpoint


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Agent that plays flappy bird")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Adds verbosity on each step",
    )
    parser.add_argument(
        "--enable_inner_progress_bar",
        action="store_true",
        help="Displays progress bars within each episode",
    )
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=10,
        help="Number of experiments to average on",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000000,
        help="Maximum number of steps in an episode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    params = {"gravity": 0.05, "force_push": 0.1, "vx": 0.05}
    env = FlappyBird(**params, prob_new_bar=1, max_height_bar=0.5)

    timer = checkpoint()

    # parameters inference
    gravity, force_push, vx = infer_parameters(env)
    timer("Time spent on parameters inference")
    assert np.allclose(
        [gravity, force_push, vx],
        (
            params["gravity"],
            params["force_push"],
            params["vx"],
        ),
        rtol=1e-3,
    ), "Parameters inference failed"

    # agent definition
    agent = TreeBasedAgent(gravity, force_push, vx, max_bars=-1)
    env.reset()
    observation = env.step(0)[0]

    rewards = np.zeros(args.n_experiments)
    n_steps = np.zeros(args.n_experiments)

    for i in trange(args.n_experiments):
        step, total_reward = 0, 0
        for step in trange(
            1, args.max_steps, desc="Step", disable=not args.enable_inner_progress_bar
        ):
            action = agent.act(observation)
            observation, reward, done = env.step(action)
            if args.verbose:
                print(
                    f"action: {action}, reward: {reward}, observation: {str(observation)}"
                )

            total_reward += reward

            if args.verbose:
                print(f"Cumulated reward at step {step}: {total_reward:>3}.")
                agent.print_outcomes_stats()
            if done:
                print(f"Simulation ended after {step} steps for a total reward of {total_reward}.")
                break
        rewards[i] = total_reward
        n_steps[i] = step

    print(
        f"\n\nReward over {args.n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f} "
        f"[{rewards.min()}, {rewards.max()}]"
    )
    print(
        f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f} [{n_steps.min()}, {n_steps.max()}]"
    )

    sns.set_theme()
    plt.hist(rewards, bins=int(rewards.max()))
    plt.show()
