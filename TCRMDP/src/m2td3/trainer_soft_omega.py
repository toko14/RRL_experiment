import glob
import os
import shutil
from typing import TypedDict, Unpack
import uuid
import csv

import numpy as np
import torch
import wandb

import gymnasium as gym
from m2td3.algo_soft_omega import M2TD3SoftOmega, M2TD3Config
from m2td3.factory import env_factory, bound_factory
from m2td3.utils import ParametersObservable

# from SOFT_M2TD3.soft_m2td3 import SoftM2TD3

AGENT_DICT = {
    "M2TD3": M2TD3SoftOmega,
    # "SoftM2TD3": SoftM2TD3,
}


class Trainer:
    """Initialize Trainer

    Parameters
    ----------
    config : Dict
        configs
    experiment_name : str
        experiment name

    """

    def __init__(
        self,
        experiment_name: str,
        env_name: str,
        nb_uncertainty_dim: int,
        device: str,
        seed: int,
        output_dir: str,
        start_steps: int = 1e5,
        evaluate_qvalue_interval: int = 1e4,
        logger_interval: int = 1e5,
        evaluate_interval: int = 1e6,
        max_steps: int = 2e6,
        track: bool = True,
        project_name: str = "m2td3_dev",
        experiement_name: str | None = None,
        oracle_parameters_agent: bool = False,
        uncertainty_log_interval: int = 10_000,
        omega_temperature_start: float = 1000.0,
        omega_temperature_end: float = 4.0,
        **kwargs: Unpack[M2TD3Config],
    ):
        self.uncertainty_set = bound_factory(env_name, nb_uncertainty_dim)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.rand_state = np.random.RandomState(seed)

        unique_id = str(object=uuid.uuid4())
        self.output_dir = f"{output_dir}/{unique_id}"
        self.start_steps = start_steps
        self.seed = seed
        self.experiment_name = experiment_name

        self.evaluate_qvalue_interval = evaluate_qvalue_interval
        self.logger_interval = logger_interval
        self.evaluate_interval = evaluate_interval
        self.max_steps = max_steps
        
        self.omega_temperature_start = omega_temperature_start
        self.omega_temperature_end = omega_temperature_end

        cwd_dir = output_dir

        if os.path.exists(f"{cwd_dir}/{unique_id}/policies"):
            shutil.rmtree(f"{cwd_dir}/{unique_id}/policies")
        if os.path.exists(f"{cwd_dir}/{unique_id}/critics"):
            shutil.rmtree(f"{cwd_dir}/{unique_id}/critics")
        file_list = glob.glob(f"{cwd_dir}/{unique_id}/*")
        for file_path in file_list:
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.makedirs(f"{self.output_dir}/policies", exist_ok=True)
        os.makedirs(f"{self.output_dir}/critics", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)

        # CSV file paths for uncertainty parameter logging
        self.uncertainty_log_interval = uncertainty_log_interval
        self.data_collection_csv_path = f"{self.output_dir}/metrics/uncertainty_data_collection.csv"
        self.training_update_csv_path = f"{self.output_dir}/metrics/uncertainty_training_update.csv"

        # TODO do later

        # TODO seed
        env = env_factory(env_name)
        if oracle_parameters_agent:
            env = ParametersObservable(env=env, params_bound=self.uncertainty_set)
        self.env = env

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.state_dim = len(env.observation_space.spaces)
        else:
            self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        self.change_param_min = np.array([v[0] for v in self.uncertainty_set.values()])
        self.change_param_max = np.array([v[1] for v in self.uncertainty_set.values()])
        # Hadle M2TD3 only for the moment
        # TODO: Handle SoftM2TD3
        self.agent = M2TD3SoftOmega(
            # config=config,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            omega_dim=nb_uncertainty_dim,
            max_action=self.max_action,
            rand_state=self.rand_state,
            device=self.device,
            min_omega=self.change_param_min,
            max_omega=self.change_param_max,  # TODO to confirm
            omega_temperature=self.omega_temperature_start,
            **kwargs,
        )

        self.step = 0
        self.episode_len = env.get_wrapper_attr("_max_episode_steps")

        if track:
            wandb.init(
                project=project_name,
                name=experiment_name,
                save_code=True,
            )

    def _append_csv(self, csv_path: str, header: list[str], row: list):
        """Append a row to CSV file, writing header if file doesn't exist
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        header : list[str]
            Column headers
        row : list
            Row data to append
        """
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def save_model(self):
        """Save networks"""
        torch.save(
            self.agent.policy_network.to("cpu").state_dict(),
            f"{self.output_dir}/policies/policy-{self.step}.pth",
        )
        self.agent.policy_network.to(self.device)
        torch.save(
            self.agent.critic_network.to("cpu").state_dict(),
            f"{self.output_dir}/critics/critic-{self.step}-{self.experiment_name}.pth",
        )
        self.agent.critic_network.to(self.device)

    def _guess_save_path_model(self) -> dict[str]:
        return {
            "policy": f"{self.output_dir}/policies/policy-{self.step}.pth",
            "critic": f"{self.output_dir}/critics/critic-{self.step}-{self.experiment_name}.pth",
        }

    def sample_omega(self, step):
        """Sample uncertainty parameters

        Parameters
        ----------
        step : int
            current training step

        """
        if step <= self.start_steps:
            omega = self.rand_state.uniform(
                low=self.change_param_min,
                high=self.change_param_max,
                size=len(self.change_param_min),
            )
            dis_restart_flag = "None"
            prob_restart_flag = "None"
        else:
            omega, dis_restart_flag, prob_restart_flag = self.agent.get_omega()
            if dis_restart_flag:
                dis_restart_flag = "True"
            else:
                dis_restart_flag = "False"
            if prob_restart_flag:
                prob_restart_flag = "True"
            else:
                prob_restart_flag = "False"

        assert len(omega) == len(self.change_param_min) == len(self.change_param_max)

        omega = np.clip(
            omega,
            self.change_param_min,
            self.change_param_max,
        )
        assert isinstance(omega, np.ndarray)

        return omega, dis_restart_flag, prob_restart_flag

    def _get_current_omega_temperature(self, step: int) -> float:
        """Calculate the current omega temperature based on the schedule.
        
        Schedule relative to total_training_steps (max_steps - start_steps):
        - Warmup (0-20%): Fixed at start_val
        - Decay (20-80%): Linear decay from start_val to end_val
        - Stable (80-100%): Fixed at end_val
        """
        if step <= self.start_steps:
            return self.omega_temperature_start
            
        total_training_steps = self.max_steps - self.start_steps
        progress = step - self.start_steps
        
        warmup_end = 0.2 * total_training_steps
        decay_end = 0.8 * total_training_steps
        
        if progress <= warmup_end:
            return self.omega_temperature_start
        elif progress <= decay_end:
            # Linear decay
            decay_progress = (progress - warmup_end) / (decay_end - warmup_end)
            return self.omega_temperature_start + decay_progress * (self.omega_temperature_end - self.omega_temperature_start)
        else:
            return self.omega_temperature_end

    def interact(self, env, omega, dis_restart_flag, prob_restart_flag):
        """Interaction environment with omega

        parameters
        ----------
        env : gym.Env
            gym environment
        omega : np.Array
            Uncertainty parameters defining the environment
        dis_restart_flag : str
            Distance restart flag
        prob_restart_flag : str
            Probability restart flag

        """
        episode_start_step = self.step
        omega_dict = {k: v for k, v in zip(self.uncertainty_set.keys(), omega)}
        state, _ = env.reset(seed=self.seed, options=omega_dict)
        total_reward = 0
        done, truncated = False, False
        while not done and not truncated:
            if self.step <= self.start_steps:
                action = self.rand_state.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.low.shape,
                ).astype(env.action_space.low.dtype)
            else:
                action = self.agent.get_action(state)

            next_state, reward, done, truncated, _ = env.step(action)

            self.agent.add_memory(state, action, next_state, reward, done, omega)

            if self.step >= self.start_steps:
                # Update omega temperature according to schedule
                current_temp = self._get_current_omega_temperature(self.step)
                self.agent.omega_temperature = current_temp
                
                self.agent.train(self.step)
                # Log training update statistics to CSV at specified intervals
                if (
                    self.step % self.agent.policy_freq == 0
                    and self.step % self.uncertainty_log_interval == 0
                    and self.agent._last_actor_update_stats is not None
                ):
                    stats = self.agent._last_actor_update_stats
                    header = ["step", "worst_policy_loss_value", "worst_policy_loss_index", "soft_policy_loss_value"]
                    header.extend([f"update_omega_{i}" for i in range(len(stats["update_omega"]))])
                    header.extend([f"hatomega_prob_{i}" for i in range(len(stats["hatomega_prob"]))])
                    row = [self.step, stats["worst_policy_loss_value"], stats["worst_policy_loss_index"], stats.get("soft_policy_loss_value", "")]
                    row.extend(stats["update_omega"])
                    row.extend(stats["hatomega_prob"])
                    self._append_csv(self.training_update_csv_path, header, row)

            state = next_state
            total_reward += reward
            if done or truncated:
                wandb.log(
                    {
                        "episode reward": total_reward,
                        "omega_temperature": self.agent.omega_temperature,
                    },
                    step=self.step,
                )

            if self.step % self.evaluate_qvalue_interval == 0:
                pass
            if self.step % self.logger_interval == 0:
                pass
            if self.step % self.evaluate_interval == 0:
                self.save_model()
            if self.step >= self.max_steps:
                return True, None

            self.step += 1
            self.episode_len += 1
            if done or truncated:
                # Log episode data collection statistics to CSV
                episode_end_step = self.step
                header = ["step_start", "step_end", "episode_len", "total_reward", "dis_restart_flag", "prob_restart_flag"]
                header.extend([f"omega_{i}" for i in range(len(omega))])
                row = [episode_start_step, episode_end_step, self.episode_len, total_reward, dis_restart_flag, prob_restart_flag]
                row.extend(omega.tolist())
                self._append_csv(self.data_collection_csv_path, header, row)
                return False, total_reward

        return False, total_reward

    def main(self):
        """Training"""
        while True:
            self.agent.set_current_episode_len(self.episode_len)
            self.episode_len = 0
            omega, dis_restart_flag, prob_restart_flag = self.sample_omega(self.step)

            flag, total_reward = self.interact(self.env, omega, dis_restart_flag, prob_restart_flag)
            print(f"Step: {self.step} Omega: {omega} Total reward: {total_reward}")
            if flag:
                break
        self.save_model()
        path_model = self._guess_save_path_model()
        # save file in the artifact for the experiment
        m2td3_artifact = wandb.Artifact(
            "model",
            type="model",
            metadata={"step": self.step},
            description="model",
        )
        m2td3_artifact.add_file(path_model["policy"], "policy.pth")
        m2td3_artifact.add_file(path_model["critic"], "critic.pth")
        wandb.log_artifact(m2td3_artifact)
