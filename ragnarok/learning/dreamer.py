"""Dream training: learn policies through imagined rollouts in the world model.

Instead of learning from real environment interactions (slow, expensive),
the agent "dreams" — it imagines trajectories using the world model and
trains the policy on those imagined experiences.

This is like a human rehearsing scenarios in their head.
"""

import torch
import torch.nn as nn
from ragnarok.core.rssm import RSSM
from ragnarok.core.policy import ActorCritic
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE


def compute_lambda_returns(rewards: torch.Tensor, values: torch.Tensor,
                           continues: torch.Tensor, gamma: float = 0.99,
                           gae_lambda: float = 0.95) -> torch.Tensor:
    """Compute GAE lambda-returns for imagined trajectories.

    Args:
        rewards: (batch, horizon) predicted rewards
        values: (batch, horizon+1) predicted values
        continues: (batch, horizon) continue probabilities (sigmoid applied)
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        returns: (batch, horizon) lambda-return targets
    """
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)

    last_value = values[:, -1]
    last_return = last_value

    for t in reversed(range(horizon)):
        cont = continues[:, t]
        reward = rewards[:, t]
        value = values[:, t]
        next_value = values[:, t + 1]

        # TD target
        td_target = reward + gamma * cont * next_value
        # GAE
        td_error = td_target - value
        last_return = value + td_error + gamma * gae_lambda * cont * (last_return - next_value)
        returns[:, t] = last_return

    return returns


class DreamTrainer:
    """Trains the actor-critic through imagination in the world model.

    1. Sample initial states from the replay buffer (via world model encoding)
    2. Imagine trajectories using the world model + current policy
    3. Compute lambda-returns on imagined rewards
    4. Update actor to maximize returns, critic to predict returns
    """

    def __init__(self, rssm: RSSM, actor_critic: ActorCritic,
                 replay_buffer: ReplayBuffer,
                 imagination_horizon: int = 15,
                 imagination_batch: int = 256,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_bonus: float = 1e-3,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-4,
                 grad_clip: float = 100.0):
        self.rssm = rssm
        self.ac = actor_critic
        self.buffer = replay_buffer
        self.horizon = imagination_horizon
        self.imag_batch = imagination_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_bonus = entropy_bonus
        self.grad_clip = grad_clip

        self.actor_optimizer = torch.optim.Adam(
            self.ac.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.ac.critic.parameters(), lr=critic_lr, eps=1e-5
        )

    def _get_initial_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample initial latent states from the replay buffer.

        Encodes real observations through the world model to get
        starting states for imagination.
        """
        seq_len = min(10, self.buffer.total_steps)
        obs, actions, _, _ = self.buffer.sample_sequences(
            self.imag_batch, max(seq_len, 2)
        )
        obs_t = torch.tensor(obs, device=DEVICE)
        act_t = torch.tensor(actions, device=DEVICE)

        with torch.no_grad():
            outputs = self.rssm.observe(obs_t, act_t)
            # Take a random timestep from each sequence as starting state
            batch_size, time_steps = outputs["h"].shape[:2]
            t_idx = torch.randint(0, time_steps, (batch_size,), device=DEVICE)
            batch_idx = torch.arange(batch_size, device=DEVICE)
            h0 = outputs["h"][batch_idx, t_idx]
            z0 = outputs["z"][batch_idx, t_idx]

        return h0, z0

    def train_step(self) -> dict[str, float]:
        """Single dream training step."""
        if self.buffer.num_episodes == 0:
            return {}

        # 1. Get initial states from real experience
        h0, z0 = self._get_initial_states()

        # 2. Imagine trajectories (gradients flow through world model to actor)
        imagined = self.rssm.imagine(
            h0, z0,
            policy_fn=self.ac.policy_fn,
            horizon=self.horizon,
        )

        # 3. Compute values for all imagined states
        h_seq = imagined["h"]          # (batch, horizon+1, hidden_dim)
        z_seq = imagined["z"]          # (batch, horizon+1, stoch_dim)
        rewards = imagined["reward_pred"]    # (batch, horizon)
        continues = torch.sigmoid(imagined["continue_pred"])  # (batch, horizon)

        # Flatten for critic
        batch, time_plus_1, _ = h_seq.shape
        state_seq = torch.cat([h_seq, z_seq], dim=-1)  # (batch, horizon+1, state_dim)
        values = self.ac.critic(state_seq.reshape(-1, state_seq.shape[-1]))
        values = values.reshape(batch, time_plus_1)

        # 4. Compute lambda-returns
        returns = compute_lambda_returns(
            rewards, values.detach(), continues.detach(),
            self.gamma, self.gae_lambda
        )

        # 5. Actor loss: maximize lambda-returns + entropy
        # Compute actor's state for horizon steps (exclude last)
        actor_states = state_seq[:, :-1].reshape(-1, state_seq.shape[-1])
        actor_entropy = self.ac.actor.entropy(actor_states).mean()

        # Actor loss = negative returns (we want to maximize)
        actor_loss = -(returns.mean()) - self.entropy_bonus * actor_entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # 6. Critic loss: predict lambda-returns
        # Re-compute values (actor weights changed)
        with torch.no_grad():
            target_returns = returns.detach()

        critic_values = self.ac.critic(
            state_seq[:, :-1].detach().reshape(-1, state_seq.shape[-1])
        ).reshape(batch, self.horizon)
        critic_loss = 0.5 * (critic_values - target_returns).pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": actor_entropy.item(),
            "imagined_reward": rewards.mean().item(),
            "imagined_value": values.mean().item(),
        }

    def train(self, steps: int) -> dict[str, float]:
        """Train for multiple dream steps, return average metrics."""
        if self.buffer.num_episodes == 0:
            return {}

        totals: dict[str, float] = {}
        count = 0

        for _ in range(steps):
            metrics = self.train_step()
            if metrics:
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                count += 1

        if count == 0:
            return {}
        return {k: v / count for k, v in totals.items()}
