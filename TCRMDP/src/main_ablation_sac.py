import numpy as np
from typing import Any

from fire import Fire
from dotenv import load_dotenv

from baselines.trainer_episode import EpisodeTrainer
from baselines.algo_sac_vanilla import VanillaSACAgent
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
    alpha_ent: float = 0.2,
    track: bool = True,
    project_name: str = "ablation_sac",
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
        for env in (train_env, eval_env):
            if hasattr(env, "set_params") and callable(getattr(env, "set_params")):
                env.set_params(**mid_params)
            else:
                try:
                    env.reset(options=mid_params)
                except Exception:
                    pass

    state_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    max_action = float(train_env.action_space.high[0])

    agent = VanillaSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device_obj,
        rand_state=rand_state,
        alpha_ent=alpha_ent,
        **kwargs,
    )

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

