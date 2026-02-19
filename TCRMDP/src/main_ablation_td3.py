from typing import Any

from fire import Fire
from dotenv import load_dotenv

from baselines.trainer_episode import EpisodeTrainer
from baselines.algo_td3_adapter import TD3Adapter
from baselines.run_io import init_seeds


load_dotenv()


def main(
    experiment_name: str,
    env_name: str,
    nb_uncertainty_dim: int,
    seed: int,
    output_dir: str,
    start_steps: int = int(1e5),
    evaluate_interval: int = int(1e5),
    logger_interval: int = int(1e5),
    max_steps: int = int(2e6),
    device: str = "cuda:0",
    oracle_parameters_agent: bool = False,
    fix_env_params_to_midpoint: bool = True,
    track: bool = True,
    project_name: str = "ablation_td3",
    **kwargs: Any,
):
    device_obj, rand_state = init_seeds(seed, device)

    from m2td3.factory import env_factory, bound_factory

    train_env = env_factory(env_name=env_name)
    eval_env = env_factory(env_name=env_name)

    # Uncertainty set used for evaluation (and DR variants). For non-DR baselines,
    # we optionally fix env params to the midpoint of these bounds for fair ablation.
    params_bound = bound_factory(env_name=env_name, nb_dim=nb_uncertainty_dim)

    if oracle_parameters_agent:
        from m2td3.utils import ParametersObservable

        train_env = ParametersObservable(env=train_env, params_bound=params_bound)
        eval_env = ParametersObservable(env=eval_env, params_bound=params_bound)

    if fix_env_params_to_midpoint:
        mid_params = {k: (float(v[0]) + float(v[1])) / 2.0 for k, v in params_bound.items()}
        # Prefer set_params (persists across reset(options=None)); fall back to reset(options=...).
        for env in (train_env, eval_env):
            if hasattr(env, "set_params") and callable(getattr(env, "set_params")):
                env.set_params(**mid_params)
            else:
                try:
                    env.reset(options=mid_params)
                except Exception:
                    pass

    agent = TD3Adapter(env=train_env, device=device_obj, rand_state=rand_state, **kwargs)

    trainer = EpisodeTrainer(
        env=train_env,
        eval_env=eval_env,
        agent=agent,
        experiment_name=experiment_name,
        output_dir=output_dir,
        seed=seed,
        seed_each_episode=False,
        start_steps=start_steps,
        evaluate_interval=evaluate_interval,
        logger_interval=logger_interval,
        max_steps=max_steps,
        track=track,
        project_name=project_name,
    )
    trainer.train()


if __name__ == "__main__":
    Fire(main)

