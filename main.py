import numpy as np
from deep_rl.environments.flappy_bird import FlappyBird
import matplotlib.pyplot as plt
import seaborn as sns

from src import (
    infer_parameters,
    TreeBuilder,
    get_best_action,
    print_outcomes_stats,
    checkpoint,
)

if __name__ == "__main__":
    params = {"gravity": 0.05, "force_push": 0.1, "vx": 0.05}
    env = FlappyBird(**params, prob_new_bar=1, max_height_bar=0.5)

    timer = checkpoint()

    # parameters inference
    gravity, force_push, vx = infer_parameters(env)
    timer("Time spent on parameters inference")
    assert np.allclose(
        [gravity, force_push, vx],
        (
            params["gravity"],
            params["force_push"],
            params["vx"],
        ),
        rtol=1e-3,
    ), "Parameters inference failed"

    # tree building
    env.reset()
    (bird_x, bird_y, bird_vy), bars = env.step(0)[0]
    tree_builder = TreeBuilder(bars, gravity, force_push, vx)
    outcomes = tree_builder.build_tree(bird_x, bird_y, bird_vy)
    print(
        f"\nNumber of leaves computed: {tree_builder.n_steps_computed}\n"
        f"Number of leave computation steps saved: {tree_builder.n_steps_saved}\n"
    )
    print_outcomes_stats(outcomes)

    # experiments
    max_steps = 1000
    n_experiments = 100

    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    for i in range(n_experiments):
        step, total_reward = 0, 0
        for __ in range(max_steps):

            action = get_best_action(outcomes)
            observation, reward, done = env.step(action)
            print(
                f"action: {action}, reward: {reward}, observation: {str(observation)}"
            )

            step += 1
            total_reward += reward

            print(f"Cumulated reward at step {step}: {total_reward:>3}.")
            if done:
                print(f"Simulation ended after {step} steps.")
                break
            tree_builder = TreeBuilder(observation[1], gravity, force_push, vx)
            outcomes = tree_builder.build_tree(*observation[0])
            print_outcomes_stats(outcomes)
        rewards[i] = total_reward
        n_steps[i] = step

    print(
        f"\n\nReward over {n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f}"
    )
    print(f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f}")

    sns.set_theme()
    plt.hist(rewards, bins=int(rewards.max()))
    plt.show()
