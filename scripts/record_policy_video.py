import argparse
import glob
import os
import sys
import numpy as np
import torch
import gymnasium as gym
import rrls  # noqa: F401  # Ensure RRLS envs are registered
from gymnasium.wrappers import RecordVideo


ENV_ID_MAP = {
    "Ant": "rrls/robust-ant-v0",
    "HalfCheetah": "rrls/robust-halfcheetah-v0",
    "Hopper": "rrls/robust-hopper-v0",
    "HumanoidStandup": "rrls/robust-humanoidstandup-v0",
    "InvertedPendulum": "rrls/robust-invertedpendulum-v0",
    "Walker": "rrls/robust-walker-v0",
    "Walker2d": "rrls/robust-walker2d-v0",
}


def _extract_step_from_policy(path: str) -> int | None:
    base = os.path.basename(path)
    if base.startswith("policy-") and base.endswith(".pth"):
        middle = base[len("policy-") : -len(".pth")]
        if middle.isdigit():
            return int(middle)
    return None


def resolve_policy_path(policy_path_or_glob: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(policy_path_or_glob))
    if any(ch in expanded for ch in ["*", "?", "["]):
        candidates = glob.glob(expanded)
        if not candidates:
            raise FileNotFoundError(f"No policy files match: {expanded}")
        pairs = [(p, _extract_step_from_policy(p)) for p in candidates]
        numeric = [p for p, step in pairs if step is not None]
        if numeric:
            best = max(numeric, key=lambda p: _extract_step_from_policy(p) or -1)
            return best
        return max(candidates, key=lambda p: os.path.getmtime(p))
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Policy file not found: {expanded}")
    return expanded


def _infer_m2td3_arch(policy_path: str) -> tuple[int, int]:
    sd = torch.load(policy_path, map_location="cpu")
    hidden_layer = int(sd["input_layer.weight"].shape[0]) if "input_layer.weight" in sd else 256
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


def build_agent(policy_path: str, env: gym.Env, device: str, tcrmdp_src: str, hidden_num: int | None, hidden_layer: int | None):
    sys.path.insert(0, os.path.expandvars(os.path.expanduser(tcrmdp_src)))
    from m2td3.agent_wrapper import M2TD3AgentWrapper  # type: ignore

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    # infer if not provided
    if hidden_num is None or hidden_layer is None:
        inf_num, inf_layer = _infer_m2td3_arch(policy_path)
        hidden_num = inf_num if hidden_num is None else hidden_num
        hidden_layer = inf_layer if hidden_layer is None else hidden_layer
    return M2TD3AgentWrapper(
        policy_path=policy_path,
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_num=hidden_num,  # may be inferred or provided via CLI
        hidden_layer=hidden_layer,  # may be inferred or provided via CLI
        max_action=env.action_space.high[0],
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
        policy_std=0.0,  # Deterministic evaluation
    )


def rollout(env: gym.Env, agent, max_steps: int, seed: int | None) -> None:
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = agent.select_action(obs, use_random=False)
        obs, _, done, truncated, _ = env.step(action)
        steps += 1


def main():
    parser = argparse.ArgumentParser(description="Record policy rollout videos (Gymnasium RecordVideo)")
    parser.add_argument("--policy", required=True, help="Path or glob to policy-*.pth")
    parser.add_argument("--env", default="Ant", choices=list(ENV_ID_MAP.keys()))
    parser.add_argument("--tcrmdp_src", required=True, help="Absolute path to TCRMDP/src")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--outdir", default="./videos")
    parser.add_argument("--hidden_num", type=int, default=None, help="Number of hidden layers in policy (override)")
    parser.add_argument("--hidden_layer", type=int, default=None, help="Hidden layer width in policy (override)")
    args = parser.parse_args()

    policy_path = resolve_policy_path(args.policy)
    env_id = ENV_ID_MAP[args.env]

    base_env = gym.make(env_id, render_mode="rgb_array")
    os.makedirs(args.outdir, exist_ok=True)
    video_dir = os.path.join(args.outdir, args.env.lower())
    env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=lambda e: True)

    agent = build_agent(policy_path, base_env, args.device, args.tcrmdp_src, args.hidden_num, args.hidden_layer)

    for ep in range(args.episodes):
        rollout(env, agent, max_steps=args.max_steps, seed=ep)

    env.close()
    print(f"Saved videos to: {video_dir}")


if __name__ == "__main__":
    main()


