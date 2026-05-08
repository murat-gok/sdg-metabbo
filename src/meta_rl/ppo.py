"""
PPO Actor-Critic for ASGD-MetaBBO Meta-RL (Phase 6).

The outer controller that tunes the inner-loop parameters:
  a₁: UCB exploration coefficient c ∈ [0.1, 2.0]
  a₂: Goal-distancing step scale β ∈ [0.1, 3.0]
  a₃: Surrogate retrain interval ∈ [1, 20]

Trained with PPO (Schulman et al. 2017) over a distribution of
optimization problems (BBOB 24 × D∈{10,20} × 5 instances).

Requires PyTorch. Run training on your GPU.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from typing import Tuple, Optional, Dict, List


if HAS_TORCH:

    class ActorCritic(nn.Module):
        """
        PPO actor-critic with shared encoder backbone.

        Actor:  state → (μ, σ) for 3D continuous action
        Critic: state → scalar value estimate V(s)
        """

        def __init__(
            self,
            state_dim: int = 64,
            action_dim: int = 3,
            hidden_dim: int = 128,
        ):
            super().__init__()
            self.action_dim = action_dim

            # Shared backbone
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

            # Actor head: outputs mean of Gaussian policy
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            # Log std as learnable parameter (state-independent)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

            # Critic head
            self.critic = nn.Linear(hidden_dim, 1)

            # Action scaling: raw action ∈ (-∞, +∞) → scaled to bounds
            # a₁: c ∈ [0.1, 2.0], a₂: β ∈ [0.1, 3.0], a₃: retrain ∈ [1, 20]
            self.action_low = torch.FloatTensor([0.1, 0.1, 1.0])
            self.action_high = torch.FloatTensor([2.0, 3.0, 20.0])

        def forward(self, state: torch.Tensor):
            """
            Args:
                state: (batch, state_dim)
            Returns:
                action_mean, action_log_std, value
            """
            shared_out = self.shared(state)
            mean = self.actor_mean(shared_out)
            value = self.critic(shared_out).squeeze(-1)
            return mean, self.actor_log_std, value

        def get_action(
            self, state: torch.Tensor, deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample action from policy.

            Returns:
                action: (batch, action_dim) scaled to action bounds
                log_prob: (batch,) log probability of the action
                value: (batch,) critic value estimate
            """
            mean, log_std, value = self.forward(state)
            std = torch.exp(log_std)

            if deterministic:
                raw_action = mean
            else:
                dist = Normal(mean, std)
                raw_action = dist.rsample()

            # Log probability (sum over action dims)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            # Scale to action bounds using sigmoid
            action = torch.sigmoid(raw_action)  # [0, 1]
            low = self.action_low.to(state.device)
            high = self.action_high.to(state.device)
            action = low + action * (high - low)

            return action, log_prob, value

        def evaluate_action(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate a previously taken action (for PPO update).

            Returns:
                log_prob, value, entropy
            """
            mean, log_std, value = self.forward(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)

            # Inverse-scale action back to raw space
            low = self.action_low.to(state.device)
            high = self.action_high.to(state.device)
            action_01 = (action - low) / (high - low + 1e-8)
            action_01 = torch.clamp(action_01, 1e-6, 1 - 1e-6)
            raw_action = torch.log(action_01 / (1 - action_01))  # logit

            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            return log_prob, value, entropy


    class PPOTrainer:
        """PPO training loop for the meta-RL outer controller."""

        def __init__(
            self,
            actor_critic: ActorCritic,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_ratio: float = 0.2,
            entropy_coeff: float = 0.01,
            value_coeff: float = 0.5,
            max_grad_norm: float = 0.5,
            ppo_epochs: int = 4,
            mini_batch_size: int = 32,
        ):
            self.ac = actor_critic
            self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_ratio = clip_ratio
            self.entropy_coeff = entropy_coeff
            self.value_coeff = value_coeff
            self.max_grad_norm = max_grad_norm
            self.ppo_epochs = ppo_epochs
            self.mini_batch_size = mini_batch_size

        def compute_gae(
            self,
            rewards: List[float],
            values: List[float],
            dones: List[bool],
            next_value: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute Generalized Advantage Estimation.

            Returns:
                advantages: (T,) advantage estimates
                returns: (T,) discounted return targets for value function
            """
            T = len(rewards)
            advantages = np.zeros(T)
            gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = next_value
                else:
                    next_val = values[t + 1]

                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + np.array(values)
            return advantages, returns

        def update(self, rollout: Dict) -> Dict:
            """
            PPO update from collected rollout data.

            Args:
                rollout: dict with keys:
                    states: (T, state_dim)
                    actions: (T, action_dim)
                    log_probs: (T,)
                    rewards: (T,)
                    values: (T,)
                    dones: (T,)
                    next_value: float

            Returns:
                Training metrics dict.
            """
            # Compute GAE
            advantages, returns = self.compute_gae(
                rollout["rewards"], rollout["values"],
                rollout["dones"], rollout["next_value"]
            )

            # Convert to tensors
            states = torch.FloatTensor(np.array(rollout["states"]))
            actions = torch.FloatTensor(np.array(rollout["actions"]))
            old_log_probs = torch.FloatTensor(np.array(rollout["log_probs"]))
            advantages_t = torch.FloatTensor(advantages)
            returns_t = torch.FloatTensor(returns)

            # Normalize advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            T = len(states)
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_updates = 0

            for _ in range(self.ppo_epochs):
                # Mini-batch iteration
                indices = np.random.permutation(T)
                for start in range(0, T, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, T)
                    mb_idx = indices[start:end]

                    mb_states = states[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages_t[mb_idx]
                    mb_returns = returns_t[mb_idx]

                    # Evaluate current policy
                    new_log_probs, values, entropy = self.ac.evaluate_action(
                        mb_states, mb_actions
                    )

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, mb_returns)

                    # Total loss
                    loss = (policy_loss
                            + self.value_coeff * value_loss
                            - self.entropy_coeff * entropy.mean())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

            return {
                "policy_loss": total_policy_loss / max(n_updates, 1),
                "value_loss": total_value_loss / max(n_updates, 1),
                "entropy": total_entropy / max(n_updates, 1),
                "n_updates": n_updates,
            }


# ─────────────────────────────────────────────────
# NumPy fallback for environments without PyTorch
# ─────────────────────────────────────────────────

class FixedPolicy:
    """
    Fixed-parameter policy for use without meta-RL training.
    Returns constant actions (sensible defaults).
    """

    def __init__(self):
        # Default actions: c=0.5, β=1.0, retrain_interval=10
        self.defaults = np.array([0.5, 1.0, 10.0])

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.defaults.copy()

    def get_action_dict(self, state: np.ndarray) -> Dict:
        a = self.get_action(state)
        return {
            "exploration_coeff": float(a[0]),
            "gd_beta": float(a[1]),
            "retrain_interval": int(round(a[2])),
        }


def get_policy(state_dim: int = 64, use_torch: bool = True):
    """Factory: get best available policy."""
    if use_torch and HAS_TORCH:
        return ActorCritic(state_dim=state_dim)
    else:
        return FixedPolicy()
