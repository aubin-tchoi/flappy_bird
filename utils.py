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
    parser.add_argument(
        "--heuristic",
        type=str,
        default="convex",
        choices=["convex", "geometric"],
        help="Parameter that selects the heuristic used to guide the bird towards the center with a small velocity.",
    )

    return parser.parse_args()


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
    success_rates = (values == max_steps).mean(axis=-1)
    for alpha_idx, alpha in enumerate(alpha_list):
        for beta_idx, beta in enumerate(beta_list):
            print(
                f"Parameters - alpha: {alpha}, beta: {beta}: success rate: {success_rates[alpha_idx, beta_idx]:.2f}\n"
                f"Number of steps: {mean_results[alpha_idx, beta_idx]:.2f} "
                f"+/- {1.96 * values[alpha_idx, beta_idx, :].std():.2f} "
                f"[{values[alpha_idx, beta_idx, :].min():.2f}, {values[alpha_idx, beta_idx, :].max():.2f}]\n"
            )

    # matrix plot
    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * len(beta_list), 4 * len(alpha_list)))
    # noinspection PyUnresolvedReferences
    ax1.matshow(mean_results, cmap=plt.cm.Blues)
    # noinspection PyUnresolvedReferences
    ax2.matshow(success_rates, cmap=plt.cm.Blues)
    for row in range(mean_results.shape[1]):
        for col in range(mean_results.shape[0]):
            # not the same convention between numpy and matshow (column-major)
            ax1.text(row, col, f"{mean_results[col, row]:.2f}", va="center", ha="center")
            ax2.text(row, col, f"{success_rates[col, row] * 100:.2f}%", va="center", ha="center")
    ax1.set(
        ylabel=r"$\alpha$",
        xlabel=r"$\beta$",
        title=f"Mean rewards over {values.shape[-1]} experiments with at most {max_steps} steps.",
    )
    ax1.set_xticks(range(len(beta_list)), beta_list)
    ax1.set_yticks(range(len(alpha_list)), alpha_list)
    ax2.set(
        ylabel=r"$\alpha$",
        xlabel=r"$\beta$",
        title=f"Mean success rates over {values.shape[-1]} experiments with at most {max_steps} steps.",
    )
    ax2.set_xticks(range(len(beta_list)), beta_list)
    ax2.set_yticks(range(len(alpha_list)), alpha_list)
    plt.show()

    # best parameters
    best_alpha, best_beta = np.unravel_index(mean_results.argmax(), values.shape[:-1])
    print(
        f"Best parameters - alpha: {alpha_list[best_alpha]}, beta: {beta_list[best_beta]}\n"
        f"Number of steps: {values[best_alpha, best_beta, :].mean():.2f} "
        f"+/- {1.96 * values[best_alpha, best_beta, :].std():.2f} "
        f"[{values[best_alpha, best_beta, :].min():.2f}, {values[best_alpha, best_beta, :].max():.2f}]\n"
    )
