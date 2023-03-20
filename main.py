import argparse
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deep_rl.environments.flappy_bird import FlappyBird
from tqdm import trange

from src import infer_parameters, TreeBasedAgent, checkpoint


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Agent that plays flappy bird")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Adds verbosity on each step",
    )
    parser.add_argument(
        "--enable_inner_progress_bar",
        action="store_true",
        help="Displays progress bars within each episode",
    )
    parser.add_argument(
        "--is_final_run",
        action="store_true",
        help="Runs the experiments with a single value of (alpha, beta) instead of performing a cross-validation.",
    )
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=10,
        help="Number of experiments to average on",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of steps in an episode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Strength of the regularization towards y close to 0.5 and vy close to 0.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Factor between 0 and 1 that balances between the two regularizing factors.",
    )

    return parser.parse_args()


def launch_multiple_experiments(
    environment: FlappyBird,
    n_experiments: int,
    alpha: float = 0.0,
    beta: float = 0.3,
    max_steps: int = 1000,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> None:
    """
    Launches multiple runs for a single set of settings to obtain averaged results.
    """
    agent = TreeBasedAgent(gravity, force_push, vx, alpha=alpha, beta=beta, max_bars=-1)
    environment.reset()
    observation = environment.step(0)[0]

    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    for i in trange(n_experiments):
        step, total_reward = 0, 0
        for step in trange(1, max_steps, desc="Step", disable=disable_progress_bar):
            action = agent.act(observation)
            observation, reward, done = environment.step(action)
            if verbose:
                print(
                    f"action: {action}, reward: {reward}, observation: {str(observation)}"
                )

            total_reward += reward

            if verbose:
                print(f"Cumulated reward at step {step}: {total_reward:>3}.")
                agent.print_outcomes_stats()
            if done:
                if verbose:
                    print(
                        f"  Simulation ended after {step} steps for a total reward of {total_reward}."
                    )
                break
        rewards[i] = total_reward
        n_steps[i] = step

    print(
        f"\n\nReward over {n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f} "
        f"[{rewards.min(initial=0)}, {rewards.max(initial=0)}]"
    )
    print(
        f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f} "
        f"[{n_steps.min(initial=0)}, {n_steps.max(initial=0)}]"
    )

    sns.set_theme()
    plt.hist(rewards, bins=int(rewards.max(initial=0)))
    plt.show()


def mp_single_experiment(
    receiving_queue: Queue,
    agent: TreeBasedAgent,
    environment: FlappyBird,
    max_steps: int = 1000,
) -> None:
    """
    Launches a single experiment. Allows for multiprocessing.
    """
    step, total_reward = 0, 0
    observation = environment.reset()
    for step in range(1, max_steps + 1):
        action = agent.act(observation)
        observation, reward, done = environment.step(action)
        total_reward += reward
        if done:
            break

    receiving_queue.put(step)


def collect_data(q: Queue) -> np.ndarray:
    data = []
    while not q.empty():
        data.append(q.get())

    return np.array(data)


if __name__ == "__main__":
    args = parse_args()

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

    if args.is_final_run:
        launch_multiple_experiments(
            env,
            args.n_experiments,
            max_steps=args.max_steps,
            verbose=args.verbose,
            disable_progress_bar=args.disable_progress_bar,
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        alpha_values = [0.0, 0.5, 1.0, 2.0]
        beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]

        results = np.zeros((len(alpha_values), len(beta_values), args.n_experiments))
        for alpha_i, alpha_val in enumerate(alpha_values):
            for beta_i, beta_val in enumerate(beta_values):
                print(f"Parameters - alpha: {alpha_val}, beta: {beta_val}")
                all_processes = []
                queue = Queue()
                for _ in range(args.n_experiments):
                    process = Process(
                        target=mp_single_experiment,
                        args=(
                            queue,
                            TreeBasedAgent(
                                gravity,
                                force_push,
                                vx,
                                max_bars=-1,
                                alpha=alpha_val,
                                beta=beta_val,
                            ),
                            env,
                        ),
                    )
                    all_processes.append(process)
                    process.start()
                for process in all_processes:
                    process.join()
                results[alpha_i, beta_i, :] = collect_data(queue)
                print(
                    f"Number of steps: {results[alpha_i, beta_i, :].mean():.2f} "
                    f"+/- {1.96 * results[alpha_i, beta_i, :].std():.2f} "
                    f"[{results[alpha_i, beta_i, :].min():.2f}, {results[alpha_i, beta_i, :].max():.2f}]\n"
                )
