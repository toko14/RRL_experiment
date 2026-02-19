import numpy as np
import torch

from .algo_soft_omega import M2TD3SoftOmega, M2TD3Config


class M2TD3OnlySoftOmega(M2TD3SoftOmega):
    """M2TD3 variant: actor worst-case update, omega soft update."""

    def _update_actor(self, state_batch):
        """Update actor (worst-case) and omega (soft)."""

        worst_policy_loss_index = None
        worst_policy_loss = None
        worst_policy_loss_value = -np.inf

        for hatomega_index in range(self.hatomega_num):
            hatomega_batch = self.hatomega_list[hatomega_index](
                self.hatomega_input_batch
            )

            policy_loss = -self.critic_network.Q1(
                state_batch,
                self.policy_network(state_batch),
                hatomega_batch.detach(),
            ).mean()

            if policy_loss.item() >= worst_policy_loss_value:
                self.update_omega = list(
                    self.hatomega_list[hatomega_index](self.hatomega_input)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                worst_policy_loss = policy_loss
                worst_policy_loss_index = hatomega_index
                worst_policy_loss_value = policy_loss.item()

        # Actor update (worst-case only)
        self.optimizer_policy.zero_grad()
        worst_policy_loss.backward()
        self.optimizer_policy.step()

        # Omega update (soft over all hatomegas)
        hatomega_losses = []
        for hatomega_index in range(self.hatomega_num):
            hatomega_batch = self.hatomega_list[hatomega_index](
                self.hatomega_input_batch
            )
            q_value = self.critic_network.Q1(
                state_batch,
                self.policy_network(state_batch).detach(),
                hatomega_batch,
            ).mean()
            hatomega_losses.append(-q_value)

        hatomega_losses_tensor = torch.stack(hatomega_losses)
        soft_omega_loss = -self.omega_temperature * torch.logsumexp(
            hatomega_losses_tensor / self.omega_temperature, dim=0
        )

        for opt in self.optimizer_hatomega_list:
            opt.zero_grad()
        soft_omega_loss.backward()
        for opt in self.optimizer_hatomega_list:
            opt.step()

        self._update_hatomega_prob(worst_policy_loss_index)

        # Save statistics for CSV logging
        self._last_actor_update_stats = {
            "worst_policy_loss_value": worst_policy_loss_value,
            "worst_policy_loss_index": worst_policy_loss_index,
            "soft_policy_loss_value": worst_policy_loss_value,
            "soft_omega_loss_value": soft_omega_loss.item(),
            "update_omega": self.update_omega.copy(),
            "hatomega_prob": self.hatomega_prob.copy(),
        }


__all__ = ["M2TD3OnlySoftOmega", "M2TD3Config"]
