from dataclasses import dataclass
from functools import wraps
from math import ceil, log
from typing import List, cast, Callable, Any

import numpy as np
from multimethod import multimethod

from .typing import Bar, Observation
from .utils import index_to_decisions


@dataclass
class TreeBasedAgent:
    """
    Agent that takes decision based on the computation of a binary tree modelling all possible sequences of decisions.
    If max_bars is set to -1, the agent will always consider all the bars available in the observation
    """

    # for some reason my IDE only reads the docstrings if I put them below the attribute.

    # parameters of the environment
    gravity: float = 0.05
    """Gravity. A positive value is expected, it is counted negatively when updating the speed."""
    force_push: float = 0.1
    """Value that describes the speed gained vertically by the bird when being pushed."""
    vx: float = 0.05
    """Horizontal speed."""

    alpha: float = 0.0
    """Strength of the regularization towards y close to 0.5 and vy close to 0."""
    beta: float = 0.3
    """Factor between 0 and 1 that balances between the two regularizing factors (y close to 0.5 and vy close to 0)."""

    bars: List[Bar] | None = None
    """Positions of the bars to dodge, in the same format as in the observations."""
    outcomes: np.ndarray | None = None
    """Leaves of the binary tree of the predicted outcomes for each possible sequence of decision (array of bool)."""

    # TODO: maybe replace max_bars by a max distance (== max tree depth) if the RAM consumption is too high
    max_bars: int = -1
    """Maximum number of bars considered by the agent (a value of -1 means that all the bars are considered)."""

    n_steps_computed: int = 0
    """Number of leaves computed."""
    n_steps_saved: int = 0
    """Number of leaves whose computation was prevented by an internal optimization"""

    base_x: float = 0.5
    """Initial x position of the bird (it actually stays the same as only the bars move to avoid eventual overflow)."""

    def _process_bars(self, bars: List[Bar]) -> None:
        """
        Processes bars by filtering inactive bars, shifting them, sorting them, and keeping only a number of them.
        """
        if self.max_bars == -1:
            # taking all the bars, no sorting required
            self.bars = [
                cast(Bar, [bar[0] - self.vx, bar[1] - self.vx, bar[2], bar[3]])
                for bar in filter(
                    lambda bar: bar[1] >= self.base_x,
                    bars,
                )
            ]
        else:
            self.bars = sorted(
                # the bars have to be shifted to the left by one step
                map(
                    lambda bar: cast(
                        Bar, [bar[0] - self.vx, bar[1] - self.vx, bar[2], bar[3]]
                    ),
                    # filtering the bars that were not already passed
                    filter(
                        lambda bar: bar[1] >= self.base_x,
                        bars,
                    ),
                ),
                key=lambda bar: bar[0],
                # taking the first max_bars bars
            )[: self.max_bars]

    @staticmethod
    def _requires_outcomes(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for methods that require having called predict beforehand.
        """

        @wraps(func)
        def requires_outcomes_wrapper(self, *args: Any, **kwargs: Any):
            assert (
                self.outcomes is not None
            ), "Outcomes are not computed yet, please call TreeBasedAgent.predict first."
            return func(self, *args, **kwargs)

        return requires_outcomes_wrapper

    def __post_init__(self):
        """
        Processes the bars passed if any.
        """
        assert 0.0 <= self.beta <= 1.0, "beta should be between 0. and 1."
        if self.bars is not None:
            self._process_bars(self.bars)

    def _is_bird_crashing(self, bird_x: float, bird_y: float) -> bool:
        """
        Returns True if the specified bird position leads to it crashing against one of the bars.
        """
        if bird_y <= 0 or bird_y >= 1:
            return True
        for x_left, x_right, height, pos in self.bars:
            # it seems that when the environment compares bird_x to the left and right abscissas of the bar,
            # the bird first goes forward and then goes up or down.
            if x_left <= bird_x - self.vx <= x_right + self.vx:
                if pos and bird_y <= height:
                    return True
                if not pos and bird_y >= 1 - height:
                    return True
        return False

    def _build_tree(
        self,
        bird_x: float,
        bird_y: float,
        bird_vy: float,
        verbose: bool,
        depth: int = 0,
    ) -> np.ndarray:
        """
        Recursive function to build the binary tree.
        TODO: track memory consumption and execution time more closely.
        """
        # possible edge case: there is no bar.
        # temporary fix: fix a big size (can be the max distance) in order to encourage the bird to recenter itself
        tree_depth = (
            max(
                ceil(
                    round(
                        (max(x_right for _, x_right, __, ___ in self.bars) - bird_x)
                        / (self.vx + 1e-5),
                        4,
                    )
                ),
                0,
            )
            if self.bars
            else 10 - depth
        )
        if verbose:
            print(
                f"\ndepth - {depth:>2}: coordinates ({bird_x:.2f}, {bird_y:.2f}), speed {bird_vy:+.2f}"
            )

        if tree_depth == 0:
            self.n_steps_computed += 1
            return (
                np.ones(1, dtype=float)
                * (not self._is_bird_crashing(bird_x, bird_y))
                * (
                    1
                    - self.alpha
                    * np.sqrt(
                        self.beta * (bird_y - 0.5) ** 2 + (1 - self.beta) * bird_vy**2
                    )
                )
            )

        if self._is_bird_crashing(bird_x, bird_y):
            if verbose:
                print("Dead branch")
            self.n_steps_saved += 2**tree_depth
            return np.zeros(2**tree_depth, dtype=float)

        # TODO: in the environment the bars move instead of the bird, this will cause an issue in the update
        outcomes = np.ones(2**tree_depth, dtype=float)
        # the even indices correspond to standing still
        outcomes[::2] = self._build_tree(
            bird_x + self.vx,
            bird_y + bird_vy - self.gravity,
            bird_vy - self.gravity,
            verbose,
            depth + 1,
        )
        # the odd indices correspond to jumping
        outcomes[1::2] = self._build_tree(
            bird_x + self.vx,
            bird_y + bird_vy - self.gravity + self.force_push,
            bird_vy - self.gravity + self.force_push,
            verbose,
            depth + 1,
        )

        return outcomes

    @multimethod
    def predict(
        self,
        bird_x: float,
        bird_y: float,
        bird_vy: float,
        verbose: bool = False,
    ) -> None:
        """
        Builds the tree / compute the outcomes.
        """
        self.outcomes = self._build_tree(bird_x, bird_y, bird_vy, verbose)

    @multimethod
    def predict(
        self,
        observation: Observation,
        verbose: bool = False,
    ) -> None:
        """
        Builds the tree / compute the outcomes.
        """
        self._process_bars(observation[1])
        self.outcomes = self._build_tree(*observation[0], verbose)

    def update(
        self,
        action: bool | int,
        observation: Observation,
        verbose: bool = False,
    ) -> None:
        """
        Updates the outcomes (WIP).
        """
        # TODO: add x, y, vy to the nodes of the tree in order to prevent re-computing the whole tree.
        # removing bars the bird successfully jumped over
        for x_left, x_right, height, pos in self.bars:
            if x_right < observation[0][0]:
                if verbose:
                    print("Removing a bar")
                self.bars.pop(0)
                break  # you cannot jump over more than one bar at once

        if len((new_bars := observation[1][: self.max_bars])) != len(self.bars):
            if verbose:
                print("Rebuilding the tree")
            self._process_bars(new_bars)
            self.predict(*observation[0])
        else:
            self.outcomes = self.outcomes[int(action) :: 2]

    @multimethod
    @_requires_outcomes
    def act(self) -> int:
        """
        Returns the best action.
        """
        # TODO: maybe consider consuming the value of self.outcomes in this method (putting it back to None)
        return int(self.outcomes[1::2].sum() > self.outcomes[0::2].sum())

    @multimethod
    def act(self, observation: Observation, verbose: bool = False) -> int:
        """
        Returns the best action according to a given observation. Recomputes the tree entirely.
        """
        # TODO: implement a heuristic in case of equality (triggered when there is no possible trajectory)
        # one possible heuristic would consist in lowering the depth of the tree to get maybe one more point
        self.predict(observation, verbose)
        return int(self.outcomes[1::2].sum() > self.outcomes[0::2].sum())

    @_requires_outcomes
    def predict_trajectory_outcome(self, decisions: List[bool]) -> bool:
        """
        Computes the outcome (whether the bird crashes or not) of a given sequence of decisions.
        """
        assert len(decisions) == int(
            log(self.outcomes.shape[0], 2)
        ), "Invalid number of decisions"
        return bool(self.outcomes[sum(2**d for d in decisions)])

    @_requires_outcomes
    def predict_trajectory_success_rate(self, *decisions: bool | int) -> float:
        """
        Computes the proportion of successful outcomes associated with a given partial sequence of decisions.
        """
        tree_portion = self.outcomes
        for decision in decisions:
            # taking the even indices if decision == 0 and the odd indices otherwise
            tree_portion = tree_portion[int(decision) :: 2]

        return tree_portion.sum() / self.outcomes.shape[0]

    @_requires_outcomes
    def print_outcomes_stats(self) -> None:
        print(
            f"\nNumber of favorable outcomes:  {self.outcomes.sum():>5} / "
            f"{self.outcomes.shape[0]:<5} ({self.outcomes.mean() * 100:.2f}%)"
        )
        print(
            f"- Probability of winning when standing still: {self.predict_trajectory_success_rate(0) * 100:.2f}%"
        )
        print(
            f"- Probability of winning when jumping:        {self.predict_trajectory_success_rate(1) * 100:.2f}%\n"
        )

    @_requires_outcomes
    def print_successful_decisions(self, max_lines: int = 10) -> None:
        successful_trajectories = self.outcomes.nonzero()[0]
        indices = np.random.choice(
            successful_trajectories, min(max_lines, successful_trajectories.shape[0])
        )
        print("\nA few successful trajectories:\n -- ", end="")
        print(
            "\n -- ".join(
                " - ".join(
                    "JUMP" if decision else "FALL"
                    for decision in index_to_decisions(
                        index, int(np.log2(self.outcomes.shape[0]))
                    )
                )
                for index in indices
            )
        )

    @_requires_outcomes
    def print_successful_trajectories(
        self,
        bird_x: float,
        bird_y: float,
        bird_vy: float,
        max_lines: int = 3,
    ) -> None:
        """
        Utility function to print a few successful trajectory with each position taken by the bird after each decision.
        """
        successful_trajectories = self.outcomes.nonzero()[0]
        indices = np.random.choice(
            successful_trajectories, min(max_lines, successful_trajectories.shape[0])
        )
        print("\nA few successful trajectories:")
        for index in indices:
            new_bird_x, new_bird_y, new_bird_vy = bird_x, bird_y, bird_vy
            print(f" -- ({bird_x:.2f}, {bird_y:.2f}, {bird_vy:.2f})", end="")
            decisions = index_to_decisions(index, int(np.log2(self.outcomes.shape[0])))
            for decision in decisions:
                new_bird_x += self.vx
                new_bird_vy += -self.gravity + self.force_push * decision
                new_bird_y += new_bird_vy
                print(
                    f" -> {'jumps' if decision else 'falls'} to "
                    f"({new_bird_x:.2f}, {new_bird_y:.2f}, {new_bird_vy:.2f})",
                    end="",
                )
            print()
