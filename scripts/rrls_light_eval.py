import argparse
import glob
import os
import sys
from typing import List

import gymnasium as gym
import numpy as np
import torch
import rrls  # noqa: F401
import time


def expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _extract_step_from_policy(path: str) -> int | None:
    base = os.path.basename(path)
    if base.startswith("policy-") and base.endswith(".pth"):
        middle = base[len("policy-") : -len(".pth")]
        if middle.isdigit():
            return int(middle)
    return None


def resolve_policy_path(pattern: str) -> str:
    patt = expand(pattern)
    if any(ch in patt for ch in ["*", "?", "["]):
        candidates = glob.glob(patt)
        if not candidates:
            raise FileNotFoundError(f"No policy files match: {patt}")
        numeric = [p for p in candidates if _extract_step_from_policy(p) is not None]
        if numeric:
            best = max(numeric, key=lambda p: _extract_step_from_policy(p) or -1)
            return best
        return max(candidates, key=lambda p: os.path.getmtime(p))
    if not os.path.exists(patt):
        raise FileNotFoundError(f"Policy not found: {patt}")
    return patt


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
    # If upstream ever exposes "rrls/robust-walker-v0", prefer it when requested
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
        # Prefer Walker2d (as per RRLS README); add multi-level fallbacks to handle local RRLS variants
        try:
            from rrls.envs.walker2d import Walker2dParamsBound as ParamsBound, RobustWalker2d as ModifiedEnv
        except Exception:
            try:
                # Some RRLS versions export Walker2d symbols at package level
                from rrls.envs import Walker2dParamsBound as ParamsBound, RobustWalker2d as ModifiedEnv
            except Exception:
                # Legacy naming without 2d suffix
                from rrls.envs.walker import WalkerParamsBound as ParamsBound, RobustWalker as ModifiedEnv
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    if nb_dim == 3:
        param_bounds = ParamsBound.THREE_DIM.value
    elif nb_dim == 2:
        param_bounds = ParamsBound.TWO_DIM.value
    else:
        base = dict(ParamsBound.ONE_DIM.value)
        if env_name == "HalfCheetah" and "worldfriction" in base:
            low, _ = base["worldfriction"]
            base["worldfriction"] = [low, 4.0]
        param_bounds = base

    return ModifiedEnv, param_bounds, generate_evaluation_set


def build_agent(policy_path: str, env: gym.Env, device: str, tcrmdp_src: str | None):
    # NOTE:
    # 既存の `from m2td3.agent_wrapper import M2TD3AgentWrapper` は
    # パッケージ import 時に m2td3/__init__.py が実行され、
    # さらに trainer.py を import -> wandb 依存で失敗する。
    # 評価では trainer は不要なので、パッケージを経由せず
    # ファイル直指定の動的 import に切り替える。
    from importlib.util import spec_from_file_location, module_from_spec
    from types import ModuleType

    if tcrmdp_src is None:
        raise ValueError("tcrmdp_src must be provided to locate m2td3/agent_wrapper.py")
    tcrmdp_src = expand(tcrmdp_src)
    agent_wrapper_path = os.path.join(tcrmdp_src, "m2td3", "agent_wrapper.py")
    if not os.path.exists(agent_wrapper_path):
        raise FileNotFoundError(f"agent_wrapper.py not found: {agent_wrapper_path}")

    spec = spec_from_file_location("m2td3_agent_wrapper", agent_wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {agent_wrapper_path}")
    module: ModuleType = module_from_spec(spec)
    # 依存関係解決のために tcrmdp_src を sys.path に追加
    if tcrmdp_src not in sys.path:
        sys.path.insert(0, tcrmdp_src)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    M2TD3AgentWrapper = getattr(module, "M2TD3AgentWrapper")

    # Infer policy architecture (hidden layers/width) from the saved state_dict
    def _infer_m2td3_arch(path: str) -> tuple[int, int]:
        sd = torch.load(path, map_location="cpu")
        # hidden layer width = out_features of input_layer
        hidden_layer = int(sd["input_layer.weight"].shape[0]) if "input_layer.weight" in sd else 256
        # count hidden layers by scanning keys like hidden_layers.<idx>.weight
        max_idx = -1
        prefix = "hidden_layers."
        suffix = ".weight"
        for k in sd.keys():
            if k.startswith(prefix) and k.endswith(suffix):
                try:
                    idx = int(k[len(prefix):].split(".")[0])
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
        hidden_num = max_idx + 1 if max_idx >= 0 else 0
        return hidden_num, hidden_layer

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    hidden_num, hidden_layer = _infer_m2td3_arch(policy_path)
    print(f"[info] Inferred policy architecture: hidden_num={hidden_num}, hidden_layer={hidden_layer}")
    return M2TD3AgentWrapper(
        policy_path=policy_path,
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_num=hidden_num,
        hidden_layer=hidden_layer,
        max_action=env.action_space.high[0],
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
        policy_std=0.0,
    )


def rollout_return(env: gym.Env, agent, seed: int, max_steps: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    total = 0.0
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = agent.select_action(obs, use_random=False)
        obs, r, done, truncated, _ = env.step(action)
        total += float(r)
        steps += 1
    return total


def main():
    parser = argparse.ArgumentParser(description="Lightweight RRLS evaluation (~100k steps)")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--env", default="Ant")
    parser.add_argument("--nb_dim", type=int, default=3)
    parser.add_argument("--nb_mesh", type=int, default=5)  # 5^3 = 125 envs
    parser.add_argument("--seeds", type=int, default=10)   # 10 seeds
    parser.add_argument("--max_steps", type=int, default=800)  # 800 steps per traj (125*10*800 ≈ 1e6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tcrmdp_src", required=True)
    args = parser.parse_args()

    patt = resolve_policy_path(args.policy)
    print(f"[info] Using policy file: {patt}")
    print(
        f"[info] Config -> env={args.env}, nb_dim={args.nb_dim}, nb_mesh={args.nb_mesh}, seeds={args.seeds}, max_steps={args.max_steps}, device={args.device}"
    )

    _ = rrls  # ensure namespace registration
    base_env_id = env_id_from_name(args.env)
    print(f"[info] Base env id: {base_env_id}")
    base_env = gym.make(base_env_id)
    agent = build_agent(patt, base_env, args.device, args.tcrmdp_src)

    ModifiedEnv, param_bounds, generate_evaluation_set = rrls_components(args.env, args.nb_dim)
    # Print param bounds summary for transparency
    try:
        bounds_preview = {k: (float(v[0]), float(v[1])) for k, v in param_bounds.items()}
        print(f"[info] Param bounds: {bounds_preview}")
    except Exception:
        print("[warn] Failed to pretty-print param bounds")
    env_set: List[gym.Env] = generate_evaluation_set(
        modified_env=ModifiedEnv,
        param_bounds=param_bounds,
        nb_mesh_dim=args.nb_mesh,
    )

    num_envs = len(env_set)
    total_steps = (args.nb_mesh ** args.nb_dim) * args.max_steps * args.seeds
    print(f"[info] Grid envs: {num_envs}, Total steps (approx): {total_steps}")

    worst_list = []
    avg_list = []
    t0 = time.time()
    for seed in range(args.seeds):
        seed_start = time.time()
        print(f"[seed {seed}] start")
        rets = []
        # progress logging every ~10%
        log_every = max(1, num_envs // 10)
        for idx, env in enumerate(env_set):
            ret = rollout_return(env, agent, seed=seed, max_steps=args.max_steps)
            rets.append(ret)
            if (idx + 1) % log_every == 0 or (idx + 1) == num_envs:
                done_envs = idx + 1
                frac = done_envs / num_envs
                elapsed = time.time() - seed_start
                eta = (elapsed / frac) * (1 - frac) if frac > 0 else float('inf')
                print(
                    f"  [seed {seed}] {done_envs}/{num_envs} envs ({frac*100:.0f}%) | elapsed {elapsed:.1f}s | eta {eta:.1f}s"
                )
        # Identify worst-case env and try to print its params
        worst_value = float(np.min(rets))
        worst_idx = int(np.argmin(rets))
        worst_env = env_set[worst_idx]
        worst_params_str = None
        try:
            if hasattr(worst_env, "get_params") and callable(getattr(worst_env, "get_params")):
                worst_params = worst_env.get_params()
                worst_params_str = str({k: float(v) for k, v in worst_params.items()})
        except Exception:
            worst_params_str = None
        worst_list.append(worst_value)
        avg_list.append(float(np.mean(rets)))
        print(
            f"[seed {seed}] done | worst={worst_list[-1]:.2f} avg={avg_list[-1]:.2f} | seed_time={time.time()-seed_start:.1f}s"
        )
        if worst_params_str is not None:
            print(f"  [seed {seed}] worst_env_index={worst_idx} params={worst_params_str}")

    def summarize(xs: List[float]):
        arr = np.asarray(xs, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return mean, std

    worst_mean, worst_std = summarize(worst_list)
    avg_mean, avg_std = summarize(avg_list)

    print("==== Lightweight RRLS evaluation ====")
    print(f"Env={args.env}, nb_dim={args.nb_dim}, nb_mesh={args.nb_mesh}, seeds={args.seeds}, steps={args.max_steps}")
    print(f"Policy: {patt}")
    print(f"Worst-case: {worst_mean:.2f} ± {worst_std:.2f}")
    print(f"Average-case: {avg_mean:.2f} ± {avg_std:.2f}")
    print(f"[total] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()


