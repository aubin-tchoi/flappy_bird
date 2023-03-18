import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deep_rl.environments.flappy_bird import FlappyBird

from src import infer_parameters, TreeBasedAgent, checkpoint

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
    observation = env.step(0)[0]
    agent = TreeBasedAgent(gravity, force_push, vx, observation[1])
    agent.predict(*observation[0])
    print(
        f"\nNumber of leaves computed: {agent.n_steps_computed}\n"
        f"Number of leave computation steps saved: {agent.n_steps_saved}\n"
    )
    agent.print_outcomes_stats()

    # experiments
    max_steps = 1000
    n_experiments = 100

    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    for i in range(n_experiments):
        step, total_reward = 0, 0
        for __ in range(max_steps):

            action = agent.act(observation)
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
            agent.print_outcomes_stats()
        rewards[i] = total_reward
        n_steps[i] = step

    print(
        f"\n\nReward over {n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f}"
    )
    print(f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f}")

    sns.set_theme()
    plt.hist(rewards, bins=int(rewards.max()))
    plt.show()
