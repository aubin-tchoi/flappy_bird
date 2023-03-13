from typing import Tuple

from deep_rl.environments.flappy_bird import FlappyBird


def infer_parameters(environment: FlappyBird) -> Tuple[float, float, float]:
    """
    Infers the gravity, force_push and v_x of an unknown FlappyBird environment.
    """
    while True:
        try:
            environment.reset()

            # looping until we observe a first bar (we need it to infer v_x)
            while len((obs := environment.step(0)[0])[1]) == 0:
                pass
            (_, y_0, v_y_0) = obs[0]
            (_, y_1, v_y_1), bars_1 = environment.step(0)[0]
            (_, y_2, v_y_2), bars_2 = environment.step(1)[0]

            gravity = -v_y_1 + v_y_0
            force_push = v_y_2 - v_y_1 + gravity
            # assumption: the first bar will still be in the list of the bars after 1 step
            for x, y, h, pos in bars_2:
                if h == bars_1[0][2]:
                    v_x = abs(x - bars_1[0][0])
                    break
            else:
                raise ValueError("Could not find the first bar")

            return gravity, force_push, round(v_x, 3)
        except ValueError:
            pass
