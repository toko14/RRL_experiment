from typing import Any, Optional

import numpy as np
import wandb

from baselines.run_io import setup_output_dir


class EpisodeTrainer:
    """Minimal episodic trainer aligned with m2sac-style logging/saving."""

    def __init__(
        self,
        env,
        eval_env,
        agent,
        experiment_name: str,
        output_dir: str,
        seed: int,
        seed_each_episode: bool = True,
        start_steps: int = int(1e5),
        evaluate_interval: int = int(1e5),
        logger_interval: int = int(1e5),
        max_steps: int = int(2e6),
        track: bool = True,
        project_name: str = "ablation",
    ) -> None:
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.agent = agent
        self.experiment_name = experiment_name
        self.seed = seed
        # If False, match `td3/trainer.py` behavior: pass seed only on the very first reset,
        # then call reset() without seed on subsequent episode boundaries.
        self.seed_each_episode = bool(seed_each_episode)
        self._did_first_seeded_reset = False
        self.start_steps = int(start_steps)
        self.evaluate_interval = int(evaluate_interval)
        self.logger_interval = int(logger_interval)
        self.max_steps = int(max_steps)
        self.track = track
        self.step = 0
        self.episode_len = 0

        self.output_dir, self.run_id = setup_output_dir(output_dir)
        self.policy_path_tpl = f"{self.output_dir}/policies/policy-{{step}}.pth"
        self.critic_path_tpl = (
            f"{self.output_dir}/critics/critic-{{step}}-{self.experiment_name}.pth"
        )

        if self.track:
            wandb.init(project=project_name, name=experiment_name, save_code=True)

    def _save_models(self) -> None:
        policy_path = self.policy_path_tpl.format(step=self.step)
        critic_path = self.critic_path_tpl.format(step=self.step)
        if hasattr(self.agent, "save_policy"):
            self.agent.save_policy(policy_path)
        if hasattr(self.agent, "save_critic"):
            self.agent.save_critic(critic_path)

    def _log_losses(self, losses: Optional[dict[str, Any]]) -> None:
        if self.track and losses is not None:
            wandb.log(losses, step=self.step)

    def _run_episode(self) -> float:
        if self.seed_each_episode or not self._did_first_seeded_reset:
            state, _ = self.env.reset(seed=self.seed)
            self._did_first_seeded_reset = True
        else:
            state, _ = self.env.reset()
        done, truncated = False, False
        total_reward = 0.0

        while not done and not truncated and self.step < self.max_steps:
            if self.step <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.agent.add_memory(state, action, next_state, reward, done)

            if self.step >= self.start_steps:
                losses = self.agent.train(self.step)
                if losses and self.step % self.logger_interval == 0:
                    self._log_losses(losses)

            state = next_state
            total_reward += reward
            self.step += 1
            self.episode_len += 1

            if self.step % self.evaluate_interval == 0:
                self._save_models()

        if self.track:
            wandb.log(
                {
                    "episode reward": total_reward,
                    "episode length": self.episode_len,
                },
                step=self.step,
            )

        self.episode_len = 0
        return total_reward

    def train(self) -> None:
        while self.step < self.max_steps:
            total_reward = self._run_episode()
            print(f"step={self.step} total_reward={total_reward:.2f}")
        self._save_models()

