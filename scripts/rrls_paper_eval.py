import argparse
import glob
import os
import sys
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import rrls  # noqa: F401  # ensure RRLS namespace is registered in Gymnasium


def _extract_step_from_policy(path: str) -> int | None:
    """Extract numeric step from a filename like .../policy-5000000.pth"""
    base = os.path.basename(path)
    # strict pattern without regex import: do simple split
    if base.startswith("policy-") and base.endswith(".pth"):
        middle = base[len("policy-") : -len(".pth")]
        if middle.isdigit():
            return int(middle)
    return None


def resolve_policy_path(policy_path_or_glob: str) -> str:
    # Expand ~ and environment variables for robustness across shells
    expanded = os.path.expandvars(os.path.expanduser(policy_path_or_glob))
    if any(ch in expanded for ch in ["*", "?", "["]):
        candidates = glob.glob(expanded)
        if not candidates:
            raise FileNotFoundError(f"No policy files match: {expanded}")
        # Prefer the largest numeric step if possible
        pairs = [(p, _extract_step_from_policy(p)) for p in candidates]
        numeric = [p for p, step in pairs if step is not None]
        if numeric:
            best = max(numeric, key=lambda p: _extract_step_from_policy(p) or -1)
            return best
        # Fallback to latest modified time
        return max(candidates, key=lambda p: os.path.getmtime(p))
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Policy file not found: {expanded}")
    return expanded


def env_id_from_name(env_name: str) -> str:
    mapping = {
        "Ant": "rrls/robust-ant-v0",
        "HalfCheetah": "rrls/robust-halfcheetah-v0",
        "Hopper": "rrls/robust-hopper-v0",
        "HumanoidStandup": "rrls/robust-humanoidstandup-v0",
        "InvertedPendulum": "rrls/robust-invertedpendulum-v0",
        # Default to Walker2d id; accept "Walker" as alias for convenience
        "Walker": "rrls/robust-walker2d-v0",
        "Walker2d": "rrls/robust-walker2d-v0",
    }
    if env_name == "Walker":
        try:
            gym.spec("rrls/robust-walker-v0")
            return "rrls/robust-walker-v0"
        except Exception:
            return mapping[env_name]
    if env_name not in mapping:
        raise ValueError(f"Unsupported env_name: {env_name}")
    return mapping[env_name]


def rrls_components(env_name: str, nb_dim: int):
    # Lazy imports to avoid importing unused modules
    from rrls.evaluate import generate_evaluation_set
    
    if env_name == "Ant":
        from rrls.envs.ant import AntParamsBound as ParamsBound, RobustAnt as ModifiedEnv
    elif env_name == "HalfCheetah":
        from rrls.envs.halfcheetah import HalfCheetahParamsBound as ParamsBound, RobustHalfCheetah as ModifiedEnv
    elif env_name == "Hopper":
        from rrls.envs.hopper import HopperParamsBound as ParamsBound, RobustHopper as ModifiedEnv
    elif env_name == "HumanoidStandup":
        from rrls.envs.humanoidstandup import HumanoidStandupParamsBound as ParamsBound, RobustHumanoidStandup as ModifiedEnv
    elif env_name == "InvertedPendulum":
        from rrls.envs.invertedpendulum import InvertedPendulumParamsBound as ParamsBound, RobustInvertedPendulum as ModifiedEnv
    elif env_name in ("Walker", "Walker2d"):
        # Prefer Walker2d; add multi-level fallbacks to handle local RRLS variants
        try:
            from rrls.envs.walker2d import Walker2dParamsBound as ParamsBound, RobustWalker2d as ModifiedEnv
        except Exception:
            try:
                from rrls.envs import Walker2dParamsBound as ParamsBound, RobustWalker2d as ModifiedEnv
            except Exception:
                from rrls.envs.walker import WalkerParamsBound as ParamsBound, RobustWalker as ModifiedEnv
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    if nb_dim == 3:
        param_bounds = ParamsBound.THREE_DIM.value
    elif nb_dim == 2:
        param_bounds = ParamsBound.TWO_DIM.value
    else:
        # NOTE: HalfCheetah の 1 次元だけ、worldfriction の上限を 4.0 に揃える
        base = dict(ParamsBound.ONE_DIM.value)
        if env_name == "HalfCheetah" and "worldfriction" in base:
            low, _ = base["worldfriction"]
            base["worldfriction"] = [low, 4.0]
        param_bounds = base

    return ModifiedEnv, param_bounds, generate_evaluation_set


def build_agent(policy_path: str, env: gym.Env, device: str, tcrmdp_src: str | None):
    if tcrmdp_src:
        sys.path.insert(0, tcrmdp_src)
    try:
        from m2td3.agent_wrapper import M2TD3AgentWrapper
    except Exception as e:
        raise ImportError(
            f"Failed to import M2TD3AgentWrapper. Provide --tcrmdp_src pointing to TCRMDP/src. Error: {e}"
        )

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = M2TD3AgentWrapper(
        policy_path=policy_path,
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_num=1,
        hidden_layer=256,
        max_action=env.action_space.high[0],
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
        policy_std=0.0,  # deterministic evaluation
    )
    return agent


def rollout_return(env: gym.Env, agent, seed: int, max_steps: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    total = 0.0
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = agent.select_action(obs, use_random=False)
        obs, r, done, truncated, _ = env.step(action)
        total += float(r)
        steps += 1
    return total


def evaluate_set(env_set: List[gym.Env], agent, seeds: List[int], max_steps: int) -> Tuple[List[float], List[float]]:
    worst_case_per_seed = []
    average_case_per_seed = []
    for seed in seeds:
        returns = []
        for env in env_set:
            # Ensure determinism per env instance
            ret = rollout_return(env, agent, seed=seed, max_steps=max_steps)
            returns.append(ret)
        worst_case_per_seed.append(float(np.min(returns)))
        average_case_per_seed.append(float(np.mean(returns)))
    return worst_case_per_seed, average_case_per_seed


def main():
    parser = argparse.ArgumentParser(description="RRLS-style paper evaluation for TCRMDP M2TD3 policies")
    parser.add_argument("--policy", required=True, help="Path or glob to policy-*.pth")
    parser.add_argument("--env", default="Ant", choices=[
        "Ant", "HalfCheetah", "Hopper", "HumanoidStandup", "InvertedPendulum", "Walker", "Walker2d"
    ])
    parser.add_argument("--nb_dim", type=int, default=2, help="Uncertainty set dimension: 1, 2, or 3")
    parser.add_argument("--nb_mesh", type=int, default=10, help="Number of partitions per dimension (paper uses 10)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to average (paper uses 10)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Steps per trajectory (paper uses 1000)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tcrmdp_src", default=None, help="Path to TCRMDP/src for importing agent wrapper")
    args = parser.parse_args()

    policy_path = resolve_policy_path(args.policy)
    # Ensure RRLS envs are registered before gym.make
    _ = rrls  # touch import to avoid linter removal
    base_env = gym.make(env_id_from_name(args.env))
    src_path = (
        os.path.expandvars(os.path.expanduser(args.tcrmdp_src)) if args.tcrmdp_src else None
    )
    agent = build_agent(policy_path, base_env, device=args.device, tcrmdp_src=src_path)

    ModifiedEnv, param_bounds, generate_evaluation_set = rrls_components(args.env, args.nb_dim)
    env_set = generate_evaluation_set(
        modified_env=ModifiedEnv,
        param_bounds=param_bounds,
        nb_mesh_dim=args.nb_mesh,
    )

    # Evaluate
    seeds = list(range(args.seeds))
    worst, avg = evaluate_set(env_set, agent, seeds=seeds, max_steps=args.max_steps)

    def summarize(xs: List[float]) -> Tuple[float, float]:
        xs_np = np.asarray(xs, dtype=float)
        return float(xs_np.mean()), float(xs_np.std(ddof=1)) if len(xs_np) > 1 else 0.0

    worst_mean, worst_std = summarize(worst)
    avg_mean, avg_std = summarize(avg)

    print("==== RRLS-style evaluation (paper-like) ====")
    print(f"Env={args.env}, nb_dim={args.nb_dim}, nb_mesh={args.nb_mesh}, seeds={args.seeds}, steps={args.max_steps}")
    print(f"Policy: {policy_path}")
    print(f"Worst-case over grid: {worst_mean:.2f} ± {worst_std:.2f}")
    print(f"Average-case over grid: {avg_mean:.2f} ± {avg_std:.2f}")


if __name__ == "__main__":
    main()


