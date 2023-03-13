from dataclasses import dataclass
from math import ceil, log
from typing import Tuple, List

import numpy as np

BarObs = Tuple[float, float, float, bool]
BirdObs = Tuple[float, float, float]
FlappyObs = Tuple[BirdObs, List[BarObs]]


def predict_trajectory_outcome(decisions: List[bool], outcomes: np.ndarray) -> bool:
    assert len(decisions) == int(
        log(outcomes.shape[0], 2)
    ), "Invalid number of decisions"
    return bool(outcomes[sum(2**d for d in decisions)])


def predict_trajectory_success_rate(
    outcomes: np.ndarray, *decisions: bool | int
) -> float:
    tree_portion = outcomes
    for decision in decisions:
        # taking the even indices if decision == 0 and the odd indices otherwise
        tree_portion = tree_portion[int(decision) :: 2]

    return tree_portion.sum() / outcomes.shape[0]


def print_outcomes_stats(outcomes: np.ndarray) -> None:
    print(
        f"\nNumber of favorable outcomes: {outcomes.sum()} / {outcomes.shape[0]} ({outcomes.mean() * 100:.2f}%)"
    )
    print("The two probabilities below are conditional to the current state.")
    print(
        f"- Probability of losing when standing still: {predict_trajectory_success_rate(outcomes, 0):.2f}"
    )
    print(
        f"- Probability of losing when jumping: {predict_trajectory_success_rate(outcomes, 1):.2f}\n"
    )


@dataclass
class TreeBuilder:
    bars: List[BarObs]
    gravity: float
    force_push: float
    vx: float
    # TODO: maybe replace max_bars by a max distance (== max tree depth)
    max_bars: int = 4
    n_steps_computed: int = 0
    n_steps_saved: int = 0

    def __post_init__(self):
        if len(self.bars) > self.max_bars:
            self.bars = self.bars[: self.max_bars]

    def is_bird_not_crashing(self, bird_x: float, bird_y: float) -> bool:
        if bird_y <= 0 or bird_y >= 1:
            return False
        for x_left, x_right, height, pos in self.bars:
            if x_left <= bird_x <= x_right:
                if pos:
                    return bird_y <= height
                else:
                    return bird_y >= 1 - height
        return True

    def build_tree(
        self,
        bird_x: float,
        bird_y: float,
        bird_vy: float,
        depth: int = 0,
        verbose: bool = False,
    ) -> np.ndarray:
        tree_depth = ceil(round((self.bars[-1][1] - bird_x) / (self.vx + 1e-5), 4))
        if verbose:
            print(
                f"\ndepth - {depth:>2}: coordinates ({bird_x:.2f}, {bird_y:.2f}), speed {bird_vy:+.2f}"
            )

        if tree_depth == 0:
            self.n_steps_computed += 1
            return np.ones(1, dtype=bool) * self.is_bird_not_crashing(bird_x, bird_y)

        if not self.is_bird_not_crashing(bird_x, bird_y):
            if verbose:
                print("Dead branch")
            self.n_steps_saved += 2**tree_depth
            return np.zeros(2**tree_depth, dtype=bool)

        # TODO: in the environment the bars move instead of the bird, this will cause an issue in the update
        outcomes = np.ones(2**tree_depth, dtype=bool)
        # the even indices correspond to standing still
        outcomes[::2] = self.build_tree(
            bird_x + self.vx,
            bird_y + bird_vy - self.gravity,
            bird_vy - self.gravity,
            depth + 1,
        )
        # the odd indices correspond to jumping
        outcomes[1::2] = self.build_tree(
            bird_x + self.vx,
            bird_y + bird_vy - self.gravity + self.force_push,
            bird_vy - self.gravity + self.force_push,
            depth + 1,
        )

        return outcomes


def update_tree(
    action: bool | int,
    observation: FlappyObs,
    tree_builder: TreeBuilder,
    outcomes: np.ndarray,
) -> np.ndarray:
    # TODO: add x, y, vy to the nodes of the tree in order to prevent re-computing the whole tree.
    # removing bars the bird successfully jumped over
    for x_left, x_right, height, pos in tree_builder.bars:
        if x_right < observation[0][0]:
            tree_builder.bars.pop(0)
            break  # you cannot jump over more than one bar at once

    if len((new_bars := observation[1][: tree_builder.max_bars])) != len(
        tree_builder.bars
    ):
        tree_builder.bars = new_bars
        return tree_builder.build_tree(*observation[0])
    else:
        return outcomes[int(action) :: 2]


def get_best_action(outcomes: np.ndarray) -> int:
    return int(outcomes[1::2].sum() > outcomes[0::2].sum())
