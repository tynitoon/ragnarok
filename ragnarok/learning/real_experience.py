"""Policy gradient training from real environment episodes.

Supports both discrete (Categorical A2C) and continuous (Gaussian A2C)
action spaces. The direct policy operates on raw observations.
Also provides PixelPolicyNet for pixel-based observations (CNN encoder
directly in the policy, DQN-style — no RSSM reconstruction bottleneck).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.cnn import CNNEncoder


class DirectPolicyNet(nn.Module):
    """Simple actor-critic for discrete actions on raw observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """Returns (action_logits, value)."""
        features = self.shared(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """Select action from observation."""
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1).item()
        return torch.distributions.Categorical(logits=logits).sample().item()


class ContinuousPolicyNet(nn.Module):
    """Actor-critic for continuous actions with tanh-squashed Gaussian.

    Outputs mean and log_std for each action dimension. Actions are
    squashed through tanh and rescaled to [action_low, action_high].
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        super().__init__()
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

        # Action rescaling: tanh outputs [-1, 1], rescale to [low, high]
        if action_low is not None and action_high is not None:
            self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        else:
            self.register_buffer("action_low", -torch.ones(action_dim))
            self.register_buffer("action_high", torch.ones(action_dim))

        # Initialize log_std to a reasonable value
        nn.init.constant_(self.logstd_head.bias, -0.5)

    def forward(self, obs: torch.Tensor):
        """Returns (mean, log_std, value)."""
        features = self.shared(obs)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features).clamp(-5.0, 2.0)
        value = self.critic_head(features).squeeze(-1)
        return mean, logstd, value

    def get_dist(self, obs: torch.Tensor):
        """Get action distribution and value."""
        mean, logstd, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, logstd.exp())
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select action, returns numpy array rescaled to action bounds."""
        mean, logstd, _ = self.forward(obs)
        if deterministic:
            raw = mean
        else:
            dist = torch.distributions.Normal(mean, logstd.exp())
            raw = dist.sample()
        # Squash through tanh and rescale
        squashed = torch.tanh(raw)
        action = self._rescale(squashed)
        return action.squeeze(0).detach().cpu().numpy()

    def _rescale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        """Rescale tanh output [-1, 1] to [action_low, action_high]."""
        return self.action_low + (tanh_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def log_prob(self, obs: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of pre-tanh action."""
        mean, logstd, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, logstd.exp())
        lp = dist.log_prob(raw_action).sum(dim=-1)
        # Tanh correction: log|det(d tanh/d raw)|
        lp -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)
        return lp

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Approximate entropy (Gaussian entropy, ignoring tanh)."""
        _, logstd, _ = self.forward(obs)
        return (0.5 + 0.5 * np.log(2 * np.pi) + logstd).sum(dim=-1)


class PixelDQN(nn.Module):
    """Dueling DQN with Nature-style CNN and auxiliary state prediction.

    Architecture: Nature CNN (Mnih 2015) + dueling (Wang 2016) + state aux head.
    The auxiliary state-prediction head forces the CNN to learn useful features
    even when the Q-signal is too noisy (bootstraps representation learning).
    """

    def __init__(self, action_dim: int, channels: int = 1,
                 feature_dim: int = 512, state_dim: int = 0):
        super().__init__()
        self.channels = channels
        self.action_dim = action_dim
        self.state_dim = state_dim
        # Nature DQN conv layers for 84x84 input
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),   # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),          # 20 -> 9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),          # 9 -> 7
            nn.ReLU(),
        )
        conv_out = 64 * 7 * 7  # 3136
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
        )
        # Auxiliary: predict underlying state vector from CNN features
        if state_dim > 0:
            self.state_head = nn.Sequential(
                nn.Linear(conv_out, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim),
            )

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract CNN features (shared backbone)."""
        if obs.dim() == 2:
            size = int((obs.shape[1] / self.channels) ** 0.5)
            obs = obs.view(-1, self.channels, size, size)
        x = self.conv(obs)
        return x.view(x.size(0), -1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))."""
        x = self.features(obs)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def predict_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict underlying state vector from pixels (auxiliary task)."""
        x = self.features(obs)
        return self.state_head(x)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        q_values = self.forward(obs)
        return q_values.argmax(dim=-1).item()


class PixelDQNTrainer:
    """DQN trainer for pixel observations.

    Handles replay buffer, target network updates, and epsilon-greedy.
    """

    def __init__(self, action_dim: int, channels: int = 3,
                 capacity: int = 50000, batch_size: int = 32,
                 gamma: float = 0.99, lr: float = 3e-4,
                 target_update: int = 500, tau: float = 0.005,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.02, epsilon_decay: int = 5000,
                 grad_clip: float = 10.0, train_every: int = 1,
                 min_buffer: int = 2000, state_dim: int = 0,
                 aux_weight: float = 1.0):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.grad_clip = grad_clip
        self.train_every = train_every
        self.min_buffer = min_buffer
        self.state_dim = state_dim
        self.aux_weight = aux_weight

        self.q_net = PixelDQN(action_dim, channels, state_dim=state_dim).to(DEVICE)
        self.target_net = PixelDQN(action_dim, channels, state_dim=state_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Simple replay buffer
        self.capacity = capacity
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.next_obs_buf = []
        self.done_buf = []
        self.state_buf = []  # Vector state for auxiliary loss
        self.pos = 0
        self.size = 0
        self.total_steps = 0

    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.total_steps / self.epsilon_decay)

    def add(self, obs, action, reward, next_obs, done, state=None):
        if self.size < self.capacity:
            self.obs_buf.append(obs)
            self.act_buf.append(action)
            self.rew_buf.append(reward)
            self.next_obs_buf.append(next_obs)
            self.done_buf.append(done)
            self.state_buf.append(state)
            self.size += 1
        else:
            self.obs_buf[self.pos] = obs
            self.act_buf[self.pos] = action
            self.rew_buf[self.pos] = reward
            self.next_obs_buf[self.pos] = next_obs
            self.done_buf[self.pos] = done
            self.state_buf[self.pos] = state
        self.pos = (self.pos + 1) % self.capacity

    @staticmethod
    def _random_shift(obs: torch.Tensor, channels: int, pad: int = 4) -> torch.Tensor:
        """DrQ-style random shift augmentation for pixel observations.

        Pad image by `pad` pixels, then random crop back to original size.
        Forces CNN to learn translation-invariant features.
        """
        size = int((obs.shape[1] / channels) ** 0.5)
        imgs = obs.view(-1, channels, size, size)
        padded = F.pad(imgs, [pad] * 4, mode='replicate')
        b, c, h, w = padded.shape
        crop_h = torch.randint(0, 2 * pad + 1, (b,))
        crop_w = torch.randint(0, 2 * pad + 1, (b,))
        cropped = torch.stack([
            padded[i, :, crop_h[i]:crop_h[i]+size, crop_w[i]:crop_w[i]+size]
            for i in range(b)
        ])
        return cropped.view(b, -1)

    def train_step(self) -> dict[str, float]:
        if self.size < self.batch_size:
            return {"q_loss": 0.0}

        idx = np.random.randint(0, self.size, self.batch_size)
        obs = torch.tensor(np.array([self.obs_buf[i] for i in idx]),
                           dtype=torch.float32, device=DEVICE)
        acts = torch.tensor([self.act_buf[i] for i in idx],
                            dtype=torch.long, device=DEVICE)
        rews = torch.tensor([self.rew_buf[i] for i in idx],
                            dtype=torch.float32, device=DEVICE)
        next_obs = torch.tensor(np.array([self.next_obs_buf[i] for i in idx]),
                                dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([self.done_buf[i] for i in idx],
                             dtype=torch.float32, device=DEVICE)

        # DrQ augmentation: random shift on both obs and next_obs
        obs_aug = self._random_shift(obs, self.q_net.channels)
        next_obs_aug = self._random_shift(next_obs, self.q_net.channels)

        # Current Q (on augmented obs)
        q_vals = self.q_net(obs_aug).gather(1, acts.unsqueeze(1)).squeeze(1)

        # Target Q (Double DQN on augmented next_obs)
        with torch.no_grad():
            next_actions = self.q_net(next_obs_aug).argmax(dim=1)
            next_q = self.target_net(next_obs_aug).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rews + self.gamma * next_q * (1 - dones)

        q_loss = F.smooth_l1_loss(q_vals, target)
        total_loss = q_loss

        # Auxiliary state-prediction loss (on un-augmented obs for accuracy)
        aux_loss_val = 0.0
        if self.state_dim > 0 and self.state_buf[0] is not None:
            states = torch.tensor(np.array([self.state_buf[i] for i in idx]),
                                  dtype=torch.float32, device=DEVICE)
            pred_states = self.q_net.predict_state(obs)  # Un-augmented
            aux_loss = F.mse_loss(pred_states, states)
            total_loss = q_loss + self.aux_weight * aux_loss
            aux_loss_val = aux_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft target network update (Polyak averaging)
        for p_target, p_online in zip(self.target_net.parameters(),
                                       self.q_net.parameters()):
            p_target.data.mul_(1 - self.tau).add_(p_online.data * self.tau)

        return {"q_loss": q_loss.item(), "aux_loss": aux_loss_val}

    def evaluate(self, env, episodes: int = 5) -> float:
        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            total = 0.0
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action_idx = self.q_net.act(obs_t, deterministic=True)
                action_np = env.action_to_onehot(action_idx)
                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        return float(np.mean(rewards))


class PixelPPONet(nn.Module):
    """CNN actor-critic for pixel observations with auxiliary state prediction.

    Nature CNN backbone shared between actor and critic.
    Auxiliary state prediction head bootstraps feature learning.
    """

    def __init__(self, action_dim: int, channels: int = 2,
                 state_dim: int = 0):
        super().__init__()
        self.channels = channels
        self.action_dim = action_dim
        # Nature CNN backbone (shared)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),   # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),          # 20 -> 9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),          # 9 -> 7
            nn.ReLU(),
        )
        conv_out = 64 * 7 * 7  # 3136
        self.actor = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        if state_dim > 0:
            self.state_head = nn.Sequential(
                nn.Linear(conv_out, 128), nn.ReLU(),
                nn.Linear(128, state_dim),
            )
        self.state_dim = state_dim
        # Orthogonal initialization (important for PPO)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for policy head (more uniform initial distribution)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            size = int((obs.shape[1] / self.channels) ** 0.5)
            obs = obs.view(-1, self.channels, size, size)
        return self.conv(obs).view(obs.size(0), -1)

    def forward(self, obs: torch.Tensor):
        """Returns (action_logits, value)."""
        x = self.features(obs)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1).item()
        return torch.distributions.Categorical(logits=logits).sample().item()


class PixelPPOTrainer:
    """PPO trainer for pixel observations.

    On-policy: collect rollout, compute GAE, run PPO epochs.
    No replay buffer or target network — avoids DQN's Q-divergence issues.
    """

    def __init__(self, action_dim: int, channels: int = 2,
                 state_dim: int = 0, aux_weight: float = 2.0,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, lr: float = 2.5e-4,
                 grad_clip: float = 0.5, ppo_epochs: int = 4,
                 minibatch_size: int = 64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.grad_clip = grad_clip
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.aux_weight = aux_weight

        self.net = PixelPPONet(action_dim, channels, state_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.total_steps = 0

    def collect_rollout(self, env, n_steps: int = 512):
        """Collect n_steps of experience for PPO training.

        Returns dict of tensors ready for training.
        """
        obs_list, act_list, rew_list, done_list = [], [], [], []
        logp_list, val_list, state_list = [], [], []

        obs = env.reset()
        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.net(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            action_idx = action.item()
            action_np = env.action_to_onehot(action_idx)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            obs_list.append(obs.copy())
            act_list.append(action_idx)
            rew_list.append(reward)
            done_list.append(float(done))
            logp_list.append(logp.item())
            val_list.append(value.item())
            state_list.append(env.last_raw_obs.copy())

            self.total_steps += 1

            if done:
                obs = env.reset()
            else:
                obs = next_obs

        # Bootstrap value for last state
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_val = self.net(obs_t)
            last_val = last_val.item()

        return {
            "obs": np.array(obs_list),
            "actions": np.array(act_list),
            "rewards": np.array(rew_list),
            "dones": np.array(done_list),
            "logps": np.array(logp_list),
            "values": np.array(val_list),
            "states": np.array(state_list),
            "last_value": last_val,
        }

    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def train_on_rollout(self, rollout: dict) -> dict[str, float]:
        """Run PPO epochs on collected rollout."""
        advantages, returns = self._compute_gae(
            rollout["rewards"], rollout["values"],
            rollout["dones"], rollout["last_value"]
        )

        obs_t = torch.tensor(rollout["obs"], dtype=torch.float32, device=DEVICE)
        acts_t = torch.tensor(rollout["actions"], dtype=torch.long, device=DEVICE)
        old_logps_t = torch.tensor(rollout["logps"], dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        states_t = torch.tensor(rollout["states"], dtype=torch.float32, device=DEVICE)

        n = len(obs_t)
        total_pg, total_vf, total_ent, total_aux = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                logits, values = self.net(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts_t[mb])
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(logp - old_logps_t[mb])
                adv_mb = adv_t[mb]
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1 - self.clip_ratio,
                                                  1 + self.clip_ratio)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                vf_loss = F.mse_loss(values, returns_t[mb])

                # Auxiliary state prediction
                aux_loss = torch.tensor(0.0, device=DEVICE)
                if self.net.state_dim > 0:
                    pred_state = self.net.state_head(self.net.features(obs_t[mb]))
                    aux_loss = F.mse_loss(pred_state, states_t[mb])

                loss = (pg_loss + self.value_coeff * vf_loss
                        - self.entropy_coeff * entropy
                        + self.aux_weight * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += entropy.item()
                total_aux += aux_loss.item()
                n_updates += 1

        # Compute episode rewards from rollout
        ep_rewards = []
        ep_reward = 0.0
        for r, d in zip(rollout["rewards"], rollout["dones"]):
            ep_reward += r
            if d:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "aux_loss": total_aux / max(n_updates, 1),
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "n_episodes": len(ep_rewards),
        }

    def evaluate(self, env, episodes: int = 5) -> float:
        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            total = 0.0
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action_idx = self.net.act(obs_t, deterministic=True)
                action_np = env.action_to_onehot(action_idx)
                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        return float(np.mean(rewards))


class RealExperienceTrainer:
    """A2C trainer from real environment episodes.

    Automatically handles discrete or continuous action spaces.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 discrete: bool = True,
                 gamma: float = 0.99, entropy_coeff: float = 0.01,
                 lr: float = 3e-4, grad_clip: float = 0.5,
                 reward_shaper=None,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.action_dim = action_dim
        self.discrete = discrete
        self.reward_shaper = reward_shaper

        if discrete:
            self.policy = DirectPolicyNet(obs_dim, action_dim).to(DEVICE)
        else:
            self.policy = ContinuousPolicyNet(
                obs_dim, action_dim, action_low=action_low, action_high=action_high
            ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_and_train(self, env, deterministic: bool = False):
        """Collect one episode and train on it.

        Returns (total_reward, metrics, episode_data).
        """
        if self.discrete:
            return self._collect_discrete(env, deterministic)
        else:
            return self._collect_continuous(env, deterministic)

    def _collect_discrete(self, env, deterministic: bool = False):
        """Discrete action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            action_idx = action.item()
            action_onehot = env.action_to_onehot(action_idx)

            observations.append(obs.copy())
            actions.append(action_onehot)

            next_obs, reward, terminated, truncated, _ = env.step(action_onehot)

            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        return total_reward, self._train_a2c(log_probs, values, entropies, rewards), episode_data

    def _collect_continuous(self, env, deterministic: bool = False):
        """Continuous action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        raw_actions = []  # Pre-tanh actions for log_prob computation
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, logstd, value = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, logstd.exp())

            if deterministic:
                raw_action = mean
            else:
                raw_action = dist.rsample()  # Reparameterized

            lp = dist.log_prob(raw_action).sum(dim=-1)
            # Tanh correction
            squashed = torch.tanh(raw_action)
            lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

            log_probs.append(lp)
            values.append(value)
            entropies.append(self.policy.entropy(obs_t))
            raw_actions.append(raw_action)

            # Rescale to environment bounds and step
            env_action = self.policy._rescale(squashed).squeeze(0).detach().cpu().numpy()

            observations.append(obs.copy())
            actions.append(env_action.copy())

            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        return total_reward, self._train_a2c(log_probs, values, entropies, rewards), episode_data

    def _build_episode_data(self, observations, actions, rewards):
        """Build episode data tuple for replay buffer."""
        dones = [0.0] * len(rewards)
        dones[-1] = 1.0
        return (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def _train_a2c(self, log_probs, values, entropies, rewards) -> dict:
        """Shared A2C training on collected episode data."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)
        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "real/actor_loss": actor_loss.item(),
            "real/critic_loss": critic_loss.item(),
            "real/entropy": entropies.mean().item(),
            "real/mean_return": returns.mean().item(),
        }

    def collect_batch_and_train(self, env, batch_episodes: int = 4,
                                ppo_epochs: int = 4, clip_eps: float = 0.2):
        """Collect episodes, then train with PPO clipping for stability.

        Multi-epoch PPO is much more sample-efficient and stable for
        continuous control than single-pass A2C.
        Returns (mean_reward, metrics, last_episode_data).
        """
        all_obs, all_raw_actions, all_rewards = [], [], []
        total_rewards = []
        last_episode_data = None

        for _ in range(batch_episodes):
            if self.discrete:
                ep_data = self._collect_with_storage_discrete(env)
            else:
                ep_data = self._collect_with_storage_continuous(env)

            ep_reward, obs_list, raw_act_list, rewards, episode_data = ep_data
            total_rewards.append(ep_reward)
            all_obs.extend(obs_list)
            all_raw_actions.extend(raw_act_list)
            all_rewards.extend(rewards)
            last_episode_data = episode_data

        if len(all_rewards) < 2:
            return float(np.mean(total_rewards)), {}, last_episode_data

        metrics = self._train_ppo(
            all_obs, all_raw_actions, all_rewards,
            epochs=ppo_epochs, clip_eps=clip_eps,
        )
        return float(np.mean(total_rewards)), metrics, last_episode_data

    def _collect_with_storage_continuous(self, env):
        """Collect one continuous episode, storing obs + raw actions for PPO."""
        obs = env.reset()
        obs_list, raw_act_list, rewards = [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mean, logstd, _ = self.policy(obs_t)
                dist = torch.distributions.Normal(mean, logstd.exp())
                raw_action = dist.sample()

            squashed = torch.tanh(raw_action)
            env_action = self.policy._rescale(squashed).squeeze(0).cpu().numpy()

            obs_list.append(obs.copy())
            raw_act_list.append(raw_action.squeeze(0).cpu().numpy())
            observations.append(obs.copy())
            actions.append(env_action.copy())

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, obs_list, raw_act_list, rewards, episode_data

    def _collect_with_storage_discrete(self, env):
        """Collect one discrete episode, storing obs + action indices for PPO."""
        obs = env.reset()
        obs_list, act_idx_list, rewards = [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            action_idx = action.item()
            action_onehot = env.action_to_onehot(action_idx)

            obs_list.append(obs.copy())
            act_idx_list.append(np.array([action_idx], dtype=np.float32))
            observations.append(obs.copy())
            actions.append(action_onehot)

            next_obs, reward, terminated, truncated, _ = env.step(action_onehot)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, obs_list, act_idx_list, rewards, episode_data

    def _train_ppo(self, obs_list, action_list, rewards,
                   epochs: int = 4, clip_eps: float = 0.2) -> dict:
        """PPO training with clipped surrogate loss.

        Re-evaluates the policy multiple times on stored data for
        more stable and sample-efficient continuous control learning.
        """
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE)
        act_t = torch.tensor(np.array(action_list), dtype=torch.float32, device=DEVICE)

        # Compute returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Compute old log_probs (frozen)
        with torch.no_grad():
            if self.discrete:
                logits, old_values = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                old_log_probs = dist.log_prob(act_t.squeeze(-1).long())
            else:
                mean, logstd, old_values = self.policy(obs_t)
                dist = torch.distributions.Normal(mean, logstd.exp())
                old_log_probs = dist.log_prob(act_t).sum(dim=-1)
                squashed = torch.tanh(act_t)
                old_log_probs -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

        total_metrics = {}
        for epoch in range(epochs):
            if self.discrete:
                logits, values = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(act_t.squeeze(-1).long())
                entropy = dist.entropy()
            else:
                mean, logstd, values = self.policy(obs_t)
                dist = torch.distributions.Normal(mean, logstd.exp())
                new_log_probs = dist.log_prob(act_t).sum(dim=-1)
                squashed = torch.tanh(act_t)
                new_log_probs -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)
                entropy = self.policy.entropy(obs_t)

            # PPO clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages = returns_t - values.detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, returns_t)
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()

        return {
            "real/actor_loss": actor_loss.item(),
            "real/critic_loss": critic_loss.item(),
            "real/entropy": entropy.mean().item(),
            "real/mean_return": returns_t.mean().item(),
        }

    def _collect_continuous_data(self, env):
        """Collect one continuous episode without training. Returns components."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, logstd, value = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, logstd.exp())
            raw_action = dist.rsample()

            lp = dist.log_prob(raw_action).sum(dim=-1)
            squashed = torch.tanh(raw_action)
            lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

            log_probs.append(lp)
            values.append(value)
            entropies.append(self.policy.entropy(obs_t))

            env_action = self.policy._rescale(squashed).squeeze(0).detach().cpu().numpy()
            observations.append(obs.copy())
            actions.append(env_action.copy())

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, log_probs, values, entropies, rewards, episode_data

    def _collect_discrete_data(self, env):
        """Collect one discrete episode without training. Returns components."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            action_idx = action.item()
            action_onehot = env.action_to_onehot(action_idx)
            observations.append(obs.copy())
            actions.append(action_onehot)

            next_obs, reward, terminated, truncated, _ = env.step(action_onehot)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, log_probs, values, entropies, rewards, episode_data

    def evaluate(self, env, episodes: int = 5) -> float:
        """Evaluate policy without training."""
        rewards = []
        for _ in range(episodes):
            r, _, _ = self.collect_and_train(env, deterministic=True)
            rewards.append(r)
        return float(np.mean(rewards))

    def get_action_onehot(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action as one-hot from raw observation (discrete only)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_idx = self.policy.act(obs_t, deterministic)
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action_idx] = 1.0
        return onehot
