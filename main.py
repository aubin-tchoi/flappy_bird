import argparse
from typing import List
from multiprocessing import Pool

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
        "--is_final_run",
        action="store_true",
        help="Runs the experiments with a single value of (alpha, beta) instead of performing a cross-validation.",
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
        default=10000,
        help="Maximum number of steps in an episode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Strength of the regularization towards y close to 0.5 and vy close to 0.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Factor between 0 and 1 that balances between the two regularizing factors.",
    )

    return parser.parse_args()


def launch_multiple_experiments(
    environment: FlappyBird,
    n_experiments: int,
    alpha: float = 0.0,
    beta: float = 0.3,
    max_steps: int = 1000,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> None:
    """
    Launches multiple runs for a single set of settings to obtain averaged results.
    """
    agent = TreeBasedAgent(gravity, force_push, vx, alpha=alpha, beta=beta, max_bars=-1)
    environment.reset()
    observation = environment.step(0)[0]

    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    for exp in trange(n_experiments):
        step, total_reward = 0, 0
        for step in trange(1, max_steps, desc="Step", disable=disable_progress_bar):
            action = agent.act(observation)
            observation, reward, done = environment.step(action)
            if verbose:
                print(
                    f"action: {action}, reward: {reward}, observation: {str(observation)}"
                )

            total_reward += reward

            if verbose:
                print(f"Cumulated reward at step {step}: {total_reward:>3}.")
                agent.print_outcomes_stats()
            if done:
                if verbose:
                    print(
                        f"  Simulation ended after {step} steps for a total reward of {total_reward}."
                    )
                break
        rewards[exp] = total_reward
        n_steps[exp] = step

    print(
        f"\n\nReward over {n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f} "
        f"[{rewards.min(initial=0)}, {rewards.max(initial=0)}]"
    )
    print(
        f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f} "
        f"[{n_steps.min(initial=0)}, {n_steps.max(initial=0)}]"
    )

    sns.set_theme()
    plt.hist(rewards, bins=int(rewards.max(initial=0)))
    plt.show()


def parallel_experiment(
    environment: FlappyBird,
    agent: TreeBasedAgent,
    n_experiments: int,
    max_steps: int = 1000,
) -> np.ndarray:
    """
    Launches an experiment. Allows for easy multiprocessing as it is a pure function.
    """
    n_steps = np.zeros(n_experiments)

    for exp in range(n_experiments):
        step, total_reward = 0, 0
        observation = environment.reset()
        for step in range(1, max_steps + 1):
            action = agent.act(observation)
            observation, reward, done = environment.step(action)
            total_reward += reward
            if done:
                break
        n_steps[exp] = step

    return n_steps


def display_results(
    values: np.ndarray,
    alpha_list: List[float],
    beta_list: List[float],
    max_steps: int,
) -> None:
    """
    Displays the results obtained through the cross-validation experiment on alpha and beta.
    Operates in three steps: prints the results obtained for each set of parameters, displays them in a matrix and
    gives the best parameters.
    """
    mean_results = values.mean(axis=-1)
    for alpha_idx, alpha in enumerate(alpha_list):
        for beta_idx, beta in enumerate(beta_list):
            print(
                f"Parameters - alpha: {alpha}, beta: {beta}\n"
                f"Number of steps: {mean_results[alpha_idx, beta_idx]:.2f} "
                f"+/- {1.96 * values[alpha_idx, beta_idx, :].std():.2f} "
                f"[{values[alpha_idx, beta_idx, :].min():.2f}, {values[alpha_idx, beta_idx, :].max():.2f}]\n"
            )

    # matrix plot
    fig, ax = plt.subplots(figsize=(4 * len(alpha_list), 4 * len(beta_list)))
    # noinspection PyUnresolvedReferences
    ax.matshow(mean_results, cmap=plt.cm.Blues)
    for row in range(mean_results.shape[1]):
        for col in range(mean_results.shape[0]):
            # not the same convention between numpy and matshow (column-major)
            ax.text(row, col, f"{mean_results[col, row]:.2f}", va="center", ha="center")
    ax.set(
        ylabel=r"$\alpha$",
        xlabel=r"$\beta$",
        title=f"Mean rewards over {values.shape[-1]} experiments with at most {max_steps} steps.",
    )
    ax.set_xticks(range(len(beta_list)), beta_list)
    ax.set_yticks(range(len(alpha_list)), alpha_list)
    plt.show()

    # best parameters
    best_alpha, best_beta = np.unravel_index(mean_results.argmax(), values.shape[:-1])
    print(
        f"Best parameters - alpha: {alpha_list[best_alpha]}, beta: {beta_list[best_beta]}\n"
        f"Number of steps: {values[best_alpha, best_beta, :].mean():.2f} "
        f"+/- {1.96 * values[best_alpha, best_beta, :].std():.2f} "
        f"[{values[best_alpha, best_beta, :].min():.2f}, {values[best_alpha, best_beta, :].max():.2f}]\n"
    )


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

    if args.is_final_run:
        launch_multiple_experiments(
            env,
            args.n_experiments,
            max_steps=args.max_steps,
            verbose=args.verbose,
            disable_progress_bar=args.disable_progress_bar,
            alpha=args.alpha,
            beta=args.beta,
        )
        timer("Time spent on the experiment")
    else:
        alpha_values = [0.0, 0.2, 0.8]
        beta_values = [0.2, 0.4, 0.9]

        results = np.zeros((len(alpha_values), len(beta_values), args.n_experiments))
        all_results = []
        with Pool(processes=8) as pool:
            for alpha_i, alpha_val in enumerate(alpha_values):
                for beta_i, beta_val in enumerate(beta_values):
                    # keeping only one computation for null alpha
                    if alpha_val > 0 or beta_i == 0:
                        # one big process for each (alpha, beta) instead of n_alpha x n_beta x args.n_experiments
                        alpha_beta_result = pool.apply_async(
                            parallel_experiment,
                            (
                                env,
                                TreeBasedAgent(
                                    gravity,
                                    force_push,
                                    vx,
                                    max_bars=-1,
                                    alpha=alpha_val,
                                    beta=beta_val,
                                ),
                                args.n_experiments,
                                args.max_steps,
                            ),
                        )
                        all_results.append((alpha_i, beta_i, alpha_beta_result))

            for alpha_i, beta_i, res in all_results:
                results[alpha_i, beta_i, :] = res.get(timeout=72000)

            for alpha_i, alpha_val in enumerate(alpha_values):
                for beta_i, beta_val in enumerate(beta_values):
                    if alpha_val == 0 and beta_i > 0:
                        results[alpha_i, beta_i, :] = results[alpha_i, 0, :]

        display_results(results, alpha_values, beta_values, args.max_steps)
        timer("Time spent on parameters tuning")
