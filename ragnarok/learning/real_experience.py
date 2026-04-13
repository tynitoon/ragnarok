"""Policy training from real environment episodes.

Supports:
- Discrete (Categorical A2C) and continuous (Gaussian A2C) for vector obs
- PPO with Nature CNN for pixel observations (84x84 grayscale + frame diff)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ragnarok.infrastructure.device import DEVICE


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
                 curiosity=None,
                 latent_curiosity=None,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.action_dim = action_dim
        self.discrete = discrete
        self.reward_shaper = reward_shaper
        self.curiosity = curiosity  # ForwardPredictor (fallback)
        self.latent_curiosity = latent_curiosity  # LatentCuriosityModule or None
        # Trust region: set externally by agent after transfer
        self.trust_region_ref = None  # Frozen reference policy
        self.trust_region_alpha = 0.0  # Current KL penalty weight

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

    def collect_and_train_vec(self, vec_env, num_episodes: int = None):
        """Collect episodes from vectorized env with batched GPU inference.

        Runs all envs simultaneously. When an env finishes, its episode
        is stored and the env resets. After collection, each episode is
        re-forward-passed through the current policy for A2C training
        (avoids stale computation graph issues from batch collection).

        Returns list of (total_reward, metrics, episode_data).
        """
        if num_episodes is None:
            num_episodes = vec_env.num_envs

        N = vec_env.num_envs
        obs_batch = vec_env.reset()  # (N, obs_dim)

        # Per-env accumulators (numpy only — no graph)
        env_obs = [[] for _ in range(N)]
        env_acts = [[] for _ in range(N)]
        env_act_indices = [[] for _ in range(N)]  # For discrete: action indices
        env_next_obs = [[] for _ in range(N)]
        env_rewards = [[] for _ in range(N)]
        env_total_reward = [0.0] * N

        # Completed episodes: (reward, obs, acts, act_indices, next_obs, rewards)
        completed = []

        while len(completed) < num_episodes:
            # Batched policy inference (no_grad — we recompute during training)
            with torch.no_grad():
                obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE)

                if self.discrete:
                    logits, _ = self.policy(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_indices = dist.sample().cpu().numpy()  # (N,)
                    actions_np = np.zeros((N, self.action_dim), dtype=np.float32)
                    for i, idx in enumerate(action_indices):
                        actions_np[i, idx] = 1.0
                else:
                    mean, logstd, _ = self.policy(obs_t)
                    dist = torch.distributions.Normal(mean, logstd.exp())
                    raw_actions = dist.rsample()
                    squashed = torch.tanh(raw_actions)
                    actions_np = self.policy._rescale(squashed).cpu().numpy()
                    action_indices = None

            # Store per-env data
            for i in range(N):
                env_obs[i].append(obs_batch[i].copy())
                env_acts[i].append(actions_np[i].copy())
                if action_indices is not None:
                    env_act_indices[i].append(int(action_indices[i]))

            # Step all envs
            next_obs_batch, rewards_batch, terminated, truncated, _ = vec_env.step(actions_np)
            dones = terminated | truncated

            for i in range(N):
                train_reward = rewards_batch[i]
                if self.reward_shaper is not None:
                    raw_obs = getattr(vec_env.envs[i], 'last_raw_obs', next_obs_batch[i])
                    train_reward = self.reward_shaper(obs_batch[i], rewards_batch[i], raw_obs)

                env_rewards[i].append(train_reward)
                env_next_obs[i].append(next_obs_batch[i].copy())
                env_total_reward[i] += rewards_batch[i]

                if dones[i]:
                    completed.append((
                        env_total_reward[i],
                        env_obs[i], env_acts[i], env_act_indices[i],
                        env_next_obs[i], env_rewards[i],
                    ))
                    env_obs[i] = []
                    env_acts[i] = []
                    env_act_indices[i] = []
                    env_next_obs[i] = []
                    env_rewards[i] = []
                    env_total_reward[i] = 0.0
                    next_obs_batch[i] = vec_env.reset_single(i)

            obs_batch = next_obs_batch

        # Train on completed episodes with fresh forward passes
        results = []
        for (total_reward, obs_list, act_list, act_idx_list,
             next_list, rew_list) in completed[:num_episodes]:

            # Curiosity
            curiosity_loss = None
            if len(obs_list) > 1:
                obs_arr = np.array(obs_list)
                act_arr = np.array(act_list)
                next_arr = np.array(next_list)
                intrinsic = self._compute_curiosity(obs_arr, act_arr, next_arr)
                if intrinsic is not None:
                    for j in range(len(rew_list)):
                        rew_list[j] += intrinsic[j]
                if self.curiosity is not None:
                    curiosity_loss = self.curiosity.train_on_transitions(
                        obs_arr, act_arr, next_arr)

            episode_data = self._build_episode_data(obs_list, act_list, rew_list)

            if len(rew_list) < 2:
                results.append((total_reward, {}, episode_data))
                continue

            # Recompute log_probs/values/entropies with current policy
            obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE)

            if self.discrete:
                logits, values = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                act_t = torch.tensor(act_idx_list, dtype=torch.long, device=DEVICE)
                log_probs = [dist.log_prob(act_t)[i:i+1] for i in range(len(act_idx_list))]
                vals = [values[i:i+1] for i in range(len(obs_list))]
                ents = [dist.entropy()[i:i+1] for i in range(len(obs_list))]
            else:
                mean, logstd, values = self.policy(obs_t)
                act_t = torch.tensor(np.array(act_list), dtype=torch.float32, device=DEVICE)
                dist_new = torch.distributions.Normal(mean, logstd.exp())
                # Inverse rescale to get raw action for log_prob
                tanh_act = 2.0 * (act_t - self.policy.action_low) / (
                    self.policy.action_high - self.policy.action_low) - 1.0
                raw_act = torch.atanh(tanh_act.clamp(-0.999, 0.999))
                lp = dist_new.log_prob(raw_act).sum(dim=-1)
                lp -= torch.log(1 - tanh_act.pow(2) + 1e-6).sum(dim=-1)
                log_probs = [lp[i:i+1] for i in range(len(obs_list))]
                vals = [values[i:i+1] for i in range(len(obs_list))]
                ents_t = self.policy.entropy(obs_t)
                ents = [ents_t[i:i+1] for i in range(len(obs_list))]

            metrics = self._train_a2c(log_probs, vals, ents, rew_list,
                                      obs_list=obs_list)
            if curiosity_loss is not None:
                metrics["curiosity_loss"] = curiosity_loss
            results.append((total_reward, metrics, episode_data))

        return results

    def _collect_discrete(self, env, deterministic: bool = False):
        """Discrete action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions, next_observations = [], [], []
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
            next_observations.append(next_obs.copy())
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        # Add intrinsic curiosity rewards (batch computation)
        curiosity_loss = None
        if len(observations) > 1:
            obs_arr = np.array(observations)
            act_arr = np.array(actions)
            next_arr = np.array(next_observations)
            intrinsic = self._compute_curiosity(obs_arr, act_arr, next_arr)
            if intrinsic is not None:
                for i in range(len(rewards)):
                    rewards[i] += intrinsic[i]
            # Train forward predictor regardless (it's the fallback)
            if self.curiosity is not None:
                curiosity_loss = self.curiosity.train_on_transitions(obs_arr, act_arr, next_arr)

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        metrics = self._train_a2c(log_probs, values, entropies, rewards,
                                  obs_list=observations)
        if curiosity_loss is not None:
            metrics["curiosity_loss"] = curiosity_loss
        return total_reward, metrics, episode_data

    def _collect_continuous(self, env, deterministic: bool = False):
        """Continuous action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        raw_actions = []  # Pre-tanh actions for log_prob computation
        observations, actions, next_observations = [], [], []
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
            next_observations.append(next_obs.copy())
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        # Add intrinsic curiosity rewards
        curiosity_loss = None
        if len(observations) > 1:
            obs_arr = np.array(observations)
            act_arr = np.array(actions)
            next_arr = np.array(next_observations)
            intrinsic = self._compute_curiosity(obs_arr, act_arr, next_arr)
            if intrinsic is not None:
                for i in range(len(rewards)):
                    rewards[i] += intrinsic[i]
            if self.curiosity is not None:
                curiosity_loss = self.curiosity.train_on_transitions(obs_arr, act_arr, next_arr)

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        metrics = self._train_a2c(log_probs, values, entropies, rewards,
                                  obs_list=observations)
        if curiosity_loss is not None:
            metrics["curiosity_loss"] = curiosity_loss
        return total_reward, metrics, episode_data

    def _compute_curiosity(self, obs_arr: np.ndarray, act_arr: np.ndarray,
                           next_arr: np.ndarray) -> np.ndarray | None:
        """Compute intrinsic rewards using latent KL or forward prediction fallback.

        Forward prediction is the primary signal. When the RSSM has trained
        enough (min_rssm_episodes), latent KL blends in gradually over 50
        episodes, capping at 50% weight. This avoids diluting forward curiosity
        with noisy KL from an undertrained world model.
        """
        # Track episodes for latent curiosity readiness
        if self.latent_curiosity is not None:
            self.latent_curiosity._episodes_seen += 1

        forward_rewards = None
        if self.curiosity is not None:
            forward_rewards = self.curiosity.compute_intrinsic_rewards(
                obs_arr, act_arr, next_arr)

        latent_rewards = None
        if self.latent_curiosity is not None and self.latent_curiosity.rssm_ready:
            latent_rewards = self.latent_curiosity.compute_batch_kl(obs_arr, act_arr)

        # Ramp up latent weight gradually (0 -> 0.5 over 50 episodes after ready)
        if forward_rewards is not None and latent_rewards is not None:
            ramp_episodes = 50
            eps_since_ready = (self.latent_curiosity._episodes_seen
                               - self.latent_curiosity.min_rssm_episodes)
            latent_weight = min(0.5, 0.5 * eps_since_ready / ramp_episodes)
            return (1 - latent_weight) * forward_rewards + latent_weight * latent_rewards
        if latent_rewards is not None:
            return latent_rewards
        return forward_rewards

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

    def _train_a2c(self, log_probs, values, entropies, rewards,
                   obs_list=None) -> dict:
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

        # Trust region: KL penalty toward transferred policy
        kl_penalty = 0.0
        if self.trust_region_ref is not None and self.trust_region_alpha > 0 and obs_list:
            kl_penalty = self._compute_trust_kl(obs_list)
            loss = loss + self.trust_region_alpha * kl_penalty

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        metrics = {
            "real/actor_loss": actor_loss.item(),
            "real/critic_loss": critic_loss.item(),
            "real/entropy": entropies.mean().item(),
            "real/mean_return": returns.mean().item(),
        }
        if kl_penalty:
            metrics["real/trust_kl"] = kl_penalty.item() if torch.is_tensor(kl_penalty) else kl_penalty
        return metrics

    def _compute_trust_kl(self, obs_list) -> torch.Tensor:
        """Compute KL(current_policy || reference_policy) for trust region."""
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE)
        if self.discrete:
            logits_cur, _ = self.policy(obs_t)
            with torch.no_grad():
                logits_ref, _ = self.trust_region_ref(obs_t)
            p_cur = torch.softmax(logits_cur, dim=-1)
            p_ref = torch.softmax(logits_ref, dim=-1)
            kl = (p_cur * (p_cur.log() - p_ref.log())).sum(dim=-1).mean()
        else:
            mean_cur, logstd_cur, _ = self.policy(obs_t)
            with torch.no_grad():
                mean_ref, logstd_ref, _ = self.trust_region_ref(obs_t)
            cur = torch.distributions.Normal(mean_cur, logstd_cur.exp())
            ref = torch.distributions.Normal(mean_ref, logstd_ref.exp())
            kl = torch.distributions.kl_divergence(cur, ref).sum(dim=-1).mean()
        return kl

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
