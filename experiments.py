from multiprocessing import Pool, Manager
from multiprocessing.managers import SyncManager
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deep_rl.environments.flappy_bird import FlappyBird
from tqdm import trange, tqdm

from src import TreeBasedAgent


def launch_multiple_experiments(
    environment: FlappyBird,
    n_experiments: int,
    alpha: float = 0.0,
    beta: float = 0.3,
    min_tree_depth: int = 20,
    heuristic: Literal["convex", "geometric", "exact"] = "convex",
    max_steps: int = 1000,
    gravity: float = 0.05,
    force_push: float = 0.1,
    vx: float = 0.05,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> None:
    """
    Launches multiple runs for a single set of settings to obtain averaged results.
    """
    agent = TreeBasedAgent(
        gravity,
        force_push,
        vx,
        alpha=alpha,
        beta=beta,
        max_bars=-1,
        heuristic=heuristic,
        min_tree_depth=min_tree_depth,
    )
    environment.reset()
    observation = environment.step(0)[0]

    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    for exp in trange(n_experiments, desc="Number of finished runs", position=1):
        step, total_reward = 0, 0
        for step in trange(
            1,
            max_steps + 1,
            desc="Steps within an episode",
            disable=disable_progress_bar,
            position=2,
            leave=False,
        ):
            action = agent.sample_action(observation)
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
        rewards[exp] = total_reward
        n_steps[exp] = step
        print(
            f"\rCurrent mean reward / number of steps: {rewards[:exp + 1].mean():.2f} / {n_steps[:exp + 1].mean():.2f}",
            end="\r",
            flush=True,
        )

    print(
        f"\n\nReward over {n_experiments} experiments: {rewards.mean():.2f} +/- {1.96 * rewards.std():.2f} "
        f"[{rewards.min(initial=max_steps)}, {rewards.max(initial=0)}]"
    )
    print(
        f"Number of steps: {n_steps.mean():.2f} +/- {1.96 * n_steps.std():.2f} "
        f"[{n_steps.min(initial=max_steps)}, {n_steps.max(initial=0)}]"
    )

    sns.set_theme()
    # showing the score function computed as a matrix
    if heuristic == "exact":
        agent.visualize_score_function()

    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.hist(
        rewards,
        bins=(int(rewards.max(initial=0)) - int(rewards.min(initial=max_steps)) // 10 or 1),
        color="cornflowerblue",
        edgecolor="cornflowerblue",
    )
    ax2.hist(
        n_steps,
        bins=(int(n_steps.max(initial=0)) - int(n_steps.min(initial=max_steps)) // 10 or 1),
        color="cornflowerblue",
        edgecolor="cornflowerblue",
    )
    ax1.set(xlabel="rewards", ylabel="number of occurrences")
    ax2.set(xlabel="n_steps", ylabel="number of occurrences")
    plt.show()


def parallel_experiment(
    environment: FlappyBird,
    agent: TreeBasedAgent,
    n_experiments: int,
    max_steps: int,
    p_id: int,
    lock_manager: SyncManager,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Launches an experiment. Allows for easy multiprocessing as it is a pure function.
    """
    rewards = np.zeros(n_experiments)
    n_steps = np.zeros(n_experiments)

    with lock_manager:
        global_progress_bar = tqdm(
            desc=f"Total over all runs - alpha: {agent.alpha}, beta: {agent.beta}",
            total=n_experiments,
            position=2 * p_id,
            leave=False,
        )
        inner_progress_bar = tqdm(
            desc=f"Progress within run - alpha: {agent.alpha}, beta: {agent.beta}",
            total=max_steps,
            position=2 * p_id + 1,
            leave=False,
        )

    for exp in range(n_experiments):
        step, total_reward = 0, 0
        observation = environment.reset()
        for step in range(1, max_steps + 1):
            action = agent.sample_action(observation)
            observation, reward, done = environment.step(action)
            total_reward += reward
            if done:
                break

            with lock_manager:
                inner_progress_bar.update()
        with lock_manager:
            inner_progress_bar.reset()
            global_progress_bar.update()

        rewards[exp] = total_reward
        n_steps[exp] = step

    with lock_manager:
        global_progress_bar.close()
        inner_progress_bar.close()

    return rewards, n_steps


def launch_cross_validation(
    alphas: List[float],
    betas: List[float],
    environment: FlappyBird,
    n_experiments: int,
    max_steps: int,
    min_tree_depth: int = 20,
    heuristic: Literal["convex", "geometric", "exact"] = "convex",
    gravity: float = 0.05,
    force_push: float = 0.1,
    vx: float = 0.05,
    n_processes: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Launches a series of experiment to cross-validate on the values of alpha and beta.
    The experiments are multiprocessed with one process for each value of (alpha, beta), which is fit for the current
    experiment that has 7 values of (alpha, beta) for 8 processes.

    Be careful with the number of processes, if you use one less process than there are combinations of (alpha, beta)
    it will take twice the time (the same applies to one less process than a divisor of the number of combinations).
    """
    lock, process_id = Manager().Lock(), 0

    all_rewards = np.zeros((len(alphas), len(betas), n_experiments))
    all_n_steps = np.zeros((len(alphas), len(betas), n_experiments))
    async_results = []

    with Pool(processes=n_processes) as pool:
        for alpha_i, alpha_val in enumerate(alphas):
            for beta_i, beta_val in enumerate(betas):
                # keeping only one computation for null alpha
                if alpha_val > 0 or beta_i == 0:
                    # one big process for each (alpha, beta) instead of n_alpha x n_beta x args.n_experiments
                    alpha_beta_result = pool.apply_async(
                        parallel_experiment,
                        args=(
                            environment,
                            TreeBasedAgent(
                                gravity,
                                force_push,
                                vx,
                                max_bars=-1,
                                alpha=alpha_val,
                                beta=beta_val,
                                heuristic=heuristic,
                                min_tree_depth=min_tree_depth,
                            ),
                            n_experiments,
                            max_steps,
                            process_id,
                            lock,
                        ),
                    )
                    async_results.append((alpha_i, beta_i, alpha_beta_result))
                    process_id += 1

        for alpha_i, beta_i, res in async_results:
            rewards, n_steps = res.get(timeout=86400)
            all_rewards[alpha_i, beta_i, :] = rewards
            all_n_steps[alpha_i, beta_i, :] = n_steps

        # copying the values for the experiments we skipped
        for alpha_i, alpha_val in enumerate(alphas):
            for beta_i, beta_val in enumerate(betas):
                if alpha_val == 0 and beta_i > 0:
                    all_rewards[alpha_i, beta_i, :] = all_rewards[alpha_i, 0, :]
                    all_n_steps[alpha_i, beta_i, :] = all_n_steps[alpha_i, 0, :]

    return all_rewards, all_n_steps


def launch_concurrent_runs(
    environment: FlappyBird,
    first_agent,
    second_agent,
    n_experiments: int,
    max_steps: int,
) -> float:
    """
    Launches a series of concurrent runs, where two agents compete against one another in order to identify the
    strengths and weaknesses of each agent through an analysis of the situations where one of the agent fails and the
    other one succeeds.
    One agent should be using the environment, and a parallel environment that has the same bars as the first one but
    whose position and velocity is controlled by the actions of the second agent. This can be done by reproducing the
    dynamic of the problem (see TreeBasedAgent._build_tree for the dynamic and TreeBasedAgent._is_bird_crashing for the
    condition of a crash).
    TODO: implement concurrent runs.
    """
    pass
