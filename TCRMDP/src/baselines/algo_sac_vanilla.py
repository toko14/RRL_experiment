import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from m2td3.utils import GaussianPolicyNetwork, CriticNetwork, ReplayBuffer, Transition


class VanillaSACAgent:
    """Vanilla SAC without omega/hatomega tricks (ablation baseline)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        rand_state: np.random.RandomState,
        alpha_ent: float = 0.2,
        batch_size: int = 100,
        replay_size: int = int(1e6),
        policy_hidden_num: int = 2,
        policy_hidden_size: int = 256,
        critic_hidden_num: int = 2,
        critic_hidden_size: int = 256,
        p_lr: float = 3e-4,
        q_lr: float = 3e-4,
        policy_freq: int = 1,
        gamma: float = 0.99,
        polyak: float = 5e-3,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.alpha_ent = alpha_ent
        self.batch_size = batch_size
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.polyak = polyak

        self.policy_network = GaussianPolicyNetwork(
            state_dim,
            action_dim,
            policy_hidden_num,
            policy_hidden_size,
            max_action,
            device,
        ).to(self.device)
        self.policy_target = copy.deepcopy(self.policy_network)

        # omega_dim=0 to reuse critic without uncertainty inputs
        self.critic_network = CriticNetwork(
            state_dim,
            action_dim,
            0,
            critic_hidden_num,
            critic_hidden_size,
            0,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_network)

        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=p_lr)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=q_lr)

        self.replay_buffer = ReplayBuffer(rand_state, capacity=replay_size)

    def _zero_omega(self, batch: int) -> torch.Tensor:
        # Critic expects omega tensor; omega_dim=0 => empty tensor works with torch.cat
        return torch.zeros((batch, 0), device=self.device, dtype=torch.float)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(
            -1, self.state_dim
        )
        with torch.no_grad():
            if deterministic:
                action = self.policy_network(state_tensor)
            else:
                action, _ = self.policy_network.sample(state_tensor)
        return action.cpu().numpy().squeeze()

    def add_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        transition = Transition(state, action, next_state, reward, done, np.array([]))
        self.replay_buffer.append(transition)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float, device=self.device)

    def train(self, step: int) -> dict[str, Any] | None:
        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return None

        state_batch = self._to_tensor(np.stack([t.state for t in transitions]))
        action_batch = self._to_tensor(np.stack([t.action for t in transitions]))
        next_state_batch = self._to_tensor(np.stack([t.next_state for t in transitions]))
        reward_batch = self._to_tensor(np.array([t.reward for t in transitions])).view(
            -1, 1
        )
        done_batch = self._to_tensor(np.array([t.done for t in transitions])).view(-1, 1)
        omega_zero = self._zero_omega(self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.policy_target.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(
                next_state_batch, next_action, omega_zero
            )
            target_q = torch.min(target_q1, target_q2) - self.alpha_ent * next_log_prob
            target = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1, current_q2 = self.critic_network(
            state_batch, action_batch, omega_zero
        )
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        log_dict: dict[str, Any] = {"critic_loss": critic_loss.item()}

        if step % self.policy_freq == 0:
            pi_action, log_prob = self.policy_network.sample(state_batch)
            q_pi = self.critic_network.Q1(state_batch, pi_action, omega_zero)
            actor_loss = (self.alpha_ent * log_prob - q_pi).mean()

            self.optimizer_policy.zero_grad()
            actor_loss.backward()
            self.optimizer_policy.step()

            self._soft_update_targets()
            log_dict["actor_loss"] = actor_loss.item()

        return log_dict

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for param, target in zip(
                self.critic_network.parameters(), self.critic_target.parameters()
            ):
                target.data.mul_(1 - self.polyak)
                target.data.add_(self.polyak * param.data)
            for param, target in zip(
                self.policy_network.parameters(), self.policy_target.parameters()
            ):
                target.data.mul_(1 - self.polyak)
                target.data.add_(self.polyak * param.data)

    def save_policy(self, path: str) -> None:
        state_dict = self.policy_network.to("cpu").state_dict()
        torch.save(state_dict, path)
        self.policy_network.to(self.device)

    def save_critic(self, path: str) -> None:
        state_dict = self.critic_network.to("cpu").state_dict()
        torch.save(state_dict, path)
        self.critic_network.to(self.device)

