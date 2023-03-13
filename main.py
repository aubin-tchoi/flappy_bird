import numpy as np
from deep_rl.environments.flappy_bird import FlappyBird

from src import (
    infer_parameters,
    TreeBuilder,
    predict_trajectory_success_rate,
)

if __name__ == "__main__":
    params = {"gravity": 0.05, "force_push": 0.1, "vx": 0.05}
    env = FlappyBird(**params, prob_new_bar=1, max_height_bar=0.5)

    # parameters inference
    gravity, force_push, vx = infer_parameters(env)
    assert np.allclose(
        [gravity, force_push, vx],
        (
            params["gravity"],
            params["force_push"],
            params["vx"],
        ),
        rtol=3,
    ), "Parameters inference failed"

    # tree building
    env.reset()
    (bird_x, bird_y, bird_vy), bars = env.step(0)[0]
    tree_builder = TreeBuilder(bars, gravity, force_push, vx)
    tree = tree_builder.build_tree(bird_x, bird_y, bird_vy)
    print(
        f"Number of favorable outcomes: {tree.sum()} / {tree.shape[0]} ({tree.mean() * 100:.2f}%)"
    )
    print("The two probabilities below are conditional to the current state.")
    print(
        f"Probability of losing when standing still: {predict_trajectory_success_rate(tree, 0):.2f}"
    )
    print(
        f"Probability of losing when jumping: {predict_trajectory_success_rate(tree, 1):.2f}"
    )
