from typing import Any

import numpy as np
import torch

from td3.models import Actor, QNetwork
from td3.td3 import TD3Agent


class TD3Adapter:
    """Thin adapter to expose TD3Agent with m2sac-style trainer hooks."""

    def __init__(self, env, device: torch.device, **kwargs: Any) -> None:
        # `main_ablation_td3.py` passes `rand_state` from `init_seeds()`, but
        # `td3.td3.TD3Agent` does not accept it. Consume it here to avoid
        # forwarding unexpected kwargs to TD3Agent.
        kwargs.pop("rand_state", None)
        # Avoid collisions if callers pass these explicitly.
        kwargs.pop("action_min", None)
        kwargs.pop("action_max", None)

        obs_space = env.observation_space
        action_space = env.action_space
        observation_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))

        actor = Actor(observation_dim=observation_dim, action_space=action_space).to(
            device
        )
        qf1 = QNetwork(observation_dim=observation_dim, action_dim=action_dim).to(device)
        qf2 = QNetwork(observation_dim=observation_dim, action_dim=action_dim).to(device)

        self.agent = TD3Agent(
            actor,
            qf1,
            qf2,
            action_min=float(action_space.low.min()),
            action_max=float(action_space.high.max()),
            device=device,
            **kwargs,
        )
        self.device = device
        self.action_space = action_space

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            saved_noise = self.agent.exploration_noise
            self.agent.exploration_noise = 0.0
            action = self.agent.select_action(state, use_random=False)
            self.agent.exploration_noise = saved_noise
        else:
            action = self.agent.select_action(state, use_random=False)
        return action

    def add_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.agent.store_transition(state, action, reward, next_state, done)

    def train(self, step: int) -> dict[str, Any] | None:
        if len(self.agent.replay_buffer.memory) < self.agent.batch_size:
            return None
        return self.agent.update(step)

    def save_policy(self, path: str) -> None:
        state_dict = self.agent.actor.to("cpu").state_dict()
        torch.save(state_dict, path)
        self.agent.actor.to(self.device)

    def save_critic(self, path: str) -> None:
        state_dict = {
            "qf1": self.agent.qf1.to("cpu").state_dict(),
            "qf2": self.agent.qf2.to("cpu").state_dict(),
        }
        torch.save(state_dict, path)
        self.agent.qf1.to(self.device)
        self.agent.qf2.to(self.device)

