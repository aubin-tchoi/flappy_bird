from typing import List

import numpy as np

from .build_tree import predict_trajectory_success_rate
from .typing import Observation


def repr_obs(observation: Observation) -> str:
    """
    Utility function to represent an observation.
    """
    (bird_x, bird_y, bird_vy), bars = observation
    repr_bars = "\n - ".join(
        f"({bar[0]:.2f}, {bar[1]:.2f}, {bar[2]:.2f}, {'TOP' if bar[3] else 'BOTTOM'})"
        for bar in bars
    )
    return f"({bird_x:.2f}, {bird_y:.2f}, {bird_vy:.2f})\n - {repr_bars}"


def print_outcomes_stats(outcomes: np.ndarray) -> None:
    print(
        f"\nNumber of favorable outcomes:  {outcomes.sum():>5} / {outcomes.shape[0]:<5} ({outcomes.mean() * 100:.2f}%)"
    )
    print(
        f"- Probability of winning when standing still: {predict_trajectory_success_rate(outcomes, 0) * 100:.2f}%"
    )
    print(
        f"- Probability of winning when jumping:        {predict_trajectory_success_rate(outcomes, 1) * 100:.2f}%\n"
    )


def index_to_decisions(index: int, tree_depth: int) -> List[int]:
    return [index // (2**i) % (2 ** (i + 1)) // 2**i for i in range(tree_depth)]


def print_successful_decisions(outcomes: np.ndarray, max_lines: int = 10) -> None:
    successful_trajectories = outcomes.nonzero()[0]
    indices = np.random.choice(
        successful_trajectories, min(max_lines, successful_trajectories.shape[0])
    )
    print("\nA few successful trajectories:\n -- ", end="")
    print(
        "\n -- ".join(
            " - ".join(
                "JUMP" if decision else "FALL"
                for decision in index_to_decisions(
                    index, int(np.log2(outcomes.shape[0]))
                )
            )
            for index in indices
        )
    )


def print_successful_trajectories(
    bird_x: float,
    bird_y: float,
    bird_vy: float,
    gravity: float,
    force_push: float,
    vx: float,
    outcomes: np.ndarray,
    max_lines: int = 3,
) -> None:
    successful_trajectories = outcomes.nonzero()[0]
    indices = np.random.choice(
        successful_trajectories, min(max_lines, successful_trajectories.shape[0])
    )
    print("\nA few successful trajectories:")
    for index in indices:
        new_bird_x, new_bird_y, new_bird_vy = bird_x, bird_y, bird_vy
        print(f" -- ({bird_x:.2f}, {bird_y:.2f}, {bird_vy:.2f})", end="")
        decisions = index_to_decisions(index, int(np.log2(outcomes.shape[0])))
        for decision in decisions:
            new_bird_x += vx
            new_bird_vy += -gravity + force_push * decision
            new_bird_y += new_bird_vy
            print(
                f" -> {'jumps' if decision else 'falls'} to ({new_bird_x:.2f}, {new_bird_y:.2f}, {new_bird_vy:.2f})",
                end="",
            )
        print()
