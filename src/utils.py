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
