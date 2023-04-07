import numpy as np
from deep_rl.environments.flappy_bird import FlappyBird

from experiments import launch_cross_validation, launch_multiple_experiments
from src import infer_parameters, checkpoint
from utils import parse_args, display_results

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

    if not args.run_cross_val:
        launch_multiple_experiments(
            env,
            args.n_experiments,
            max_steps=args.max_steps,
            verbose=args.verbose,
            disable_progress_bar=not args.enable_inner_progress_bar,
            alpha=args.alpha,
            beta=args.beta,
            min_tree_depth=args.min_tree_depth,
            heuristic=args.heuristic,
        )
        timer("Time spent on the experiment")
    else:
        if args.heuristic == "geometric":
            alpha_values = [1.5, 1.75, 2.0]
            beta_values = [0.75, 1.25]
        elif args.heuristic == "exact":
            alpha_values = [0.2, 0.4, 0.6, 0.8, 1, 2]
            beta_values = [0.0]
        else:
            alpha_values = [0., 0.2, 0.8]
            beta_values = [0.2, 0.4, 0.9]
        rewards, n_steps = launch_cross_validation(
            alpha_values,
            beta_values,
            env,
            args.n_experiments,
            args.max_steps,
            args.min_tree_depth,
            args.heuristic,
            gravity,
            force_push,
            vx,
        )
        display_results(rewards, n_steps, alpha_values, beta_values, args.max_steps)
        timer("Time spent on parameters tuning")
