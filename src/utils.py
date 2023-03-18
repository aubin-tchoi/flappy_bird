from typing import List

from .typing import Observation, Bar


def repr_bars(bars: List[Bar]) -> str:
    return "\n - " + "\n - ".join(
        f"({bar[0]:.2f}, {bar[1]:.2f}, {bar[2]:.2f}, {'BOTTOM' if bar[3] else 'TOP'})"
        for bar in bars
    )


def repr_obs(observation: Observation) -> str:
    """
    Utility function to represent an observation.
    """
    (bird_x, bird_y, bird_vy), bars = observation
    return f"({bird_x:.2f}, {bird_y:.2f}, {bird_vy:.2f}){repr_bars(bars)}"


def index_to_decisions(index: int, tree_depth: int) -> List[int]:
    return [index // (2**i) % (2 ** (i + 1)) // 2**i for i in range(tree_depth)]
