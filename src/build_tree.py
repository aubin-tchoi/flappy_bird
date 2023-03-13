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

    return tree_portion.sum() / outcomes.sum()


@dataclass
class TreeBuilder:
    bars: List[BarObs]
    gravity: float
    force_push: float
    vx: float
    max_bars: int = 4
    count: int = 0

    def __post_init__(self):
        if len(self.bars) > self.max_bars:
            self.bars = self.bars[: self.max_bars]

    def is_bird_not_crashing(self, bird_x: float, bird_y: float) -> bool:
        # TODO: vectorize this function
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
    ) -> np.ndarray:
        tree_depth = ceil((self.bars[-1][1] - bird_x) / (self.vx + 1e-5))

        if tree_depth == 0:
            self.count += 1
            return np.ones(1, dtype=bool) * self.is_bird_not_crashing(bird_x, bird_y)

        if not self.is_bird_not_crashing(bird_x, bird_y):
            return np.zeros(2**tree_depth, dtype=bool)

        # TODO: in the environment the bars move instead of the bird, this will cause an issue in the update
        outcomes = np.ones(2**tree_depth, dtype=bool)
        # the even indices correspond to standing still
        outcomes[::2] = self.build_tree(
            bird_x + self.vx, bird_y - bird_vy, bird_vy - self.gravity
        )
        # the odd indices correspond to jumping
        outcomes[1::2] = self.build_tree(
            bird_x + self.vx,
            bird_y - bird_vy + self.force_push,
            bird_vy - self.gravity + self.force_push,
        )

        return outcomes
