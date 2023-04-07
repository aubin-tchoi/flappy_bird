import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np


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
        "--run_cross_val",
        action="store_true",
        help="Runs a cross validation on the parameters (alpha, beta) instead of performing a regular run.",
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
        "--min_tree_depth",
        type=int,
        default=20,
        help="Minimal tree depth. Strongly affects the score function computed using the 'exact' heuristic.",
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
    parser.add_argument(
        "--heuristic",
        type=str,
        default="convex",
        choices=["convex", "geometric", "exact"],
        help="Parameter that selects the heuristic used to guide the bird towards the center with a small velocity.",
    )

    return parser.parse_args()


def display_results(
    rewards: np.ndarray,
    n_steps: np.ndarray,
    alpha_list: List[float],
    beta_list: List[float],
    max_steps: int,
) -> None:
    """
    Displays the results obtained through the cross-validation experiment on alpha and beta.
    Operates in three steps: prints the results obtained for each set of parameters, displays them in a matrix and
    gives the best parameters.
    """
    mean_rewards = rewards.mean(axis=-1)
    mean_n_steps = n_steps.mean(axis=-1)
    success_rates = (n_steps == max_steps).mean(axis=-1)
    for alpha_idx, alpha in enumerate(alpha_list):
        for beta_idx, beta in enumerate(beta_list):
            print(
                f"Parameters - alpha: {alpha}, beta: {beta}: success rate: {success_rates[alpha_idx, beta_idx]:.2f}\n"
                f"Rewards: {mean_rewards[alpha_idx, beta_idx]:.2f} "
                f"+/- {1.96 * rewards[alpha_idx, beta_idx, :].std():.2f} "
                f"[{rewards[alpha_idx, beta_idx, :].min():.2f}, {rewards[alpha_idx, beta_idx, :].max():.2f}]\n"
                f"Number of steps: {mean_n_steps[alpha_idx, beta_idx]:.2f} "
                f"+/- {1.96 * n_steps[alpha_idx, beta_idx, :].std():.2f} "
                f"[{n_steps[alpha_idx, beta_idx, :].min():.2f}, {n_steps[alpha_idx, beta_idx, :].max():.2f}]\n"
            )

    # matrix plot
    # noinspection PyArgumentEqualDefault
    fig, axes = plt.subplots(1, 3, figsize=(8 * len(beta_list), 2 * len(alpha_list)))
    # noinspection PyUnresolvedReferences
    axes[0].matshow(mean_rewards, cmap=plt.cm.Oranges)
    # noinspection PyUnresolvedReferences
    axes[1].matshow(mean_n_steps, cmap=plt.cm.Oranges)
    # noinspection PyUnresolvedReferences
    axes[2].matshow(success_rates, cmap=plt.cm.Oranges)
    for row in range(mean_n_steps.shape[1]):
        for col in range(mean_n_steps.shape[0]):
            # not the same convention between numpy and matshow (column-major)
            axes[0].text(
                row, col, f"{mean_rewards[col, row]:.2f}", va="center", ha="center"
            )
            axes[1].text(
                row, col, f"{mean_n_steps[col, row]:.2f}", va="center", ha="center"
            )
            axes[2].text(
                row,
                col,
                f"{success_rates[col, row] * 100:.2f}%",
                va="center",
                ha="center",
            )
    for ax in axes:
        ax.set(ylabel=r"$\alpha$", xlabel=r"$\beta$")
        ax.set_xticks(range(len(beta_list)), beta_list)
        ax.set_yticks(range(len(alpha_list)), alpha_list)
    plt.suptitle(
        f"{n_steps.shape[-1]} experiments with at most {max_steps} steps.",
    )
    axes[0].set(
        title=f"Mean reward",
    )
    axes[1].set(
        title=f"Mean number of steps",
    )
    axes[2].set(
        title=f"Mean success rates",
    )
    plt.show()

    # best parameters
    best_alpha, best_beta = np.unravel_index(mean_rewards.argmax(), n_steps.shape[:-1])
    print(
        f"Best parameters (in terms of rewards) - alpha: {alpha_list[best_alpha]}, beta: {beta_list[best_beta]}\n"
        f"Number of steps: {n_steps[best_alpha, best_beta, :].mean():.2f} "
        f"+/- {1.96 * n_steps[best_alpha, best_beta, :].std():.2f} "
        f"[{n_steps[best_alpha, best_beta, :].min():.2f}, {n_steps[best_alpha, best_beta, :].max():.2f}]\n"
    )
