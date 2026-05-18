"""Latent-space policy head operating on RSSM state cat(h, z).

Unlike obs-space policies (DirectPolicyNet, ContinuousPolicyNet), this
policy takes the RSSM hidden state as input. Since (h_dim, z_dim) are
constant across all environments, this enables:
  1. Cross-environment transfer (CartPole -> Acrobot despite different obs/act dims)
  2. Efficient dream training (no lossy decode -> re-encode round-trip)

Architecture:
  cat(h, z) -> shared MLP -> actor head (env-specific) + critic head

The actor head is swapped per-environment (different action dims), but the
shared trunk weights transfer directly across tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE, mark_step


class LatentPolicyHead(nn.Module):
    """Policy operating on RSSM latent state cat(h, z).

    The shared trunk (latent_dim -> hidden -> hidden) is environment-agnostic
    and transfers across tasks. Only the actor head is env-specific.
    """

    def __init__(self, latent_dim: int, action_dim: int,
                 hidden: int = 128, discrete: bool = True,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(hidden, 1)

        if discrete:
            self.actor_head = nn.Linear(hidden, action_dim)
        else:
            self.mean_head = nn.Linear(hidden, action_dim)
            self.logstd_head = nn.Linear(hidden, action_dim)
            # Match SACPolicy (-1.0) so latent/obs continuous trainers start
            # from the same exploration scale and H1 comparisons aren't
            # confounded by an inherited -0.5 divergence.
            nn.init.constant_(self.logstd_head.bias, -1.0)
            # Action bounds for tanh-squash + rescale (mirrors SACPolicy).
            # Without this, `act()` emits raw Gaussian samples that violate
            # env.action_space on every continuous target — silent failure.
            if action_low is not None and action_high is not None:
                self.register_buffer(
                    "action_low",
                    torch.as_tensor(action_low, dtype=torch.float32))
                self.register_buffer(
                    "action_high",
                    torch.as_tensor(action_high, dtype=torch.float32))
            else:
                self.register_buffer("action_low", -torch.ones(action_dim))
                self.register_buffer("action_high", torch.ones(action_dim))

    def _rescale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (tanh_action + 1.0) * 0.5 * (
            self.action_high - self.action_low)

    def _inverse_rescale(self, env_action: torch.Tensor) -> torch.Tensor:
        """Inverse of `_rescale` — env-space action back to tanh range (-1, 1)."""
        return 2.0 * (env_action - self.action_low) / (
            self.action_high - self.action_low) - 1.0

    def forward(self, latent: torch.Tensor):
        """Forward pass on cat(h, z).

        Returns:
            Discrete: (logits, value)
            Continuous: (mean, logstd, value)
        """
        features = self.shared(latent)
        value = self.critic_head(features).squeeze(-1)

        if self.discrete:
            logits = self.actor_head(features)
            return logits, value
        else:
            mean = self.mean_head(features)
            logstd = self.logstd_head(features).clamp(-5.0, 2.0)
            return mean, logstd, value

    @torch.no_grad()
    def act(self, latent: torch.Tensor, deterministic: bool = True):
        """Select an action from cat(h, z) suitable for env.step().

        Discrete: returns int action index.
        Continuous: returns numpy array of shape (action_dim,).

        Mirrors the interface of `real_trainer.policy.act` (discrete: int)
        and `sac_trainer.policy.act` (continuous: ndarray) so the caller in
        `collect_episode` can swap policies without further branching.
        """
        if self.discrete:
            logits, _ = self.forward(latent)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.distributions.Categorical(probs).sample().item())

        mean, logstd, _ = self.forward(latent)
        if deterministic:
            action = self._rescale(torch.tanh(mean))
        else:
            std = logstd.exp()
            raw = torch.distributions.Normal(mean, std).sample()
            action = self._rescale(torch.tanh(raw))
        return action.squeeze(0).cpu().numpy()

    def evaluate_action(self, latent: torch.Tensor, action: torch.Tensor):
        """Compute (log_prob, entropy, value) for an already-taken action.

        For continuous, the stored `action` is env-space (post-tanh+rescale
        from `act()`). Naively evaluating log_prob of the raw Gaussian on
        that action would be a train/inference distribution mismatch — the
        policy would train on the wrong density. This method inverts the
        squash and adds the tanh log-det correction (cf. SACPolicy.sample
        L82-85), so the log-prob reflects the *actual* distribution the
        policy sampled from.

        The affine `_rescale` Jacobian is a constant w.r.t. policy params
        and cancels in the gradient, so we drop it — mirrors SAC.

        Returns:
            (log_prob[B], entropy[B], value[B])

        Entropy is returned per-step (not reduced) so callers training on
        padded sequences can mask the padding out of the entropy bonus.
        """
        if self.discrete:
            logits, value = self.forward(latent)
            dist = torch.distributions.Categorical(logits=logits)
            idx = action.argmax(dim=-1) if action.dim() == 2 else action.long()
            return dist.log_prob(idx), dist.entropy(), value

        mean, logstd, value = self.forward(latent)
        std = logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        # Invert rescale+tanh to recover the pre-squash sample point.
        tanh_a = self._inverse_rescale(action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(tanh_a)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        # Tanh log-det correction: log|d/dx tanh(x)| = log(1 - tanh(x)^2)
        log_prob = log_prob - torch.log(1.0 - tanh_a.pow(2) + 1e-6).sum(dim=-1)
        # Gaussian entropy as exploration bonus proxy (standard in squashed-
        # Gaussian actors — the exact squashed entropy has no closed form).
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def device_sample(self, latent: torch.Tensor):
        """Batched on-device sampling for collection — (action, logp, value).

        Discrete: action is (N,) int indices. Continuous: action is
        (N, action_dim) env-space (tanh-squashed then rescaled), with the
        squashed-Gaussian log-prob correction so logp matches the density
        actually sampled (cf. evaluate_action). All device tensors — no host
        sync, unlike act() which returns host ints / numpy.
        """
        if self.discrete:
            logits, value = self.forward(latent)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value
        mean, logstd, value = self.forward(latent)
        dist = torch.distributions.Normal(mean, logstd.exp())
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        logp = (dist.log_prob(raw).sum(dim=-1)
                - torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1))
        return self._rescale(squashed), logp, value

    @torch.no_grad()
    def device_act(self, latent: torch.Tensor) -> torch.Tensor:
        """Batched on-device greedy action for evaluation — action only.

        Discrete: (N,) int indices. Continuous: (N, action_dim) env-space.
        The device-tensor counterpart of act() (which returns host ints /
        numpy and so cannot run inside a device collection/eval loop).
        """
        if self.discrete:
            logits, _ = self.forward(latent)
            return logits.argmax(dim=-1)
        mean, _, _ = self.forward(latent)
        return self._rescale(torch.tanh(mean))

    def get_trunk_state_dict(self) -> dict:
        """Get only the shared trunk + critic weights (transferable)."""
        trunk_keys = set()
        for name, _ in self.shared.named_parameters():
            trunk_keys.add(f"shared.{name}")
        for name, _ in self.critic_head.named_parameters():
            trunk_keys.add(f"critic_head.{name}")

        return {k: v for k, v in self.state_dict().items()
                if k in trunk_keys}

    def load_trunk_state_dict(self, state_dict: dict):
        """Load only the shared trunk + critic weights (from different env)."""
        current = self.state_dict()
        for k, v in state_dict.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
        self.load_state_dict(current)


class LatentPolicyTrainer:
    """Trains a LatentPolicyHead from device rollouts (Phase 2 Stage 5.2).

    The device-path counterpart of RagnarokAgent._train_latent_policy: A2C on
    RSSM latent states cat(h, z). The RSSM runs no-grad — the world-model
    trainer owns the RSSM; the latent policy trains only its own weights on
    the detached latents — then A2C runs over env-flattened minibatches. No
    padding mask: device rollouts have no padding, every step is real.

    Returns are per-env-row, done-reset, G=0-seeded discounted reward sums —
    the same estimator as the gym _train_latent_policy, which also trains on
    fixed windows with no tail bootstrap. observe() resets the GRU at each
    episode seam (a rollout row spans several auto-reset episodes).

    The shared trunk + critic of the LatentPolicyHead are the transferable
    subset (get_trunk_state_dict) — this trainer is what gives cross-task
    transfer something trained to carry.
    """

    def __init__(self, latent_dim: int, action_dim: int, discrete: bool = True,
                 hidden: int = 128, action_low=None, action_high=None,
                 gamma: float = 0.99, value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01, grad_clip: float = 0.5,
                 lr: float = 3e-4):
        self.policy = LatentPolicyHead(
            latent_dim, action_dim, hidden, discrete,
            action_low=action_low, action_high=action_high).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip

    def train_on_rollout(self, batch, rssm, epochs: int = 1,
                         n_minibatches: int = 1) -> dict:
        """A2C update of the latent policy from a RolloutBatch.

        rssm.observe runs no-grad — the latent policy trains on detached
        cat(h, z); the RSSM is trained separately by WorldModelTrainer.
        """
        # 1. RSSM latents for the whole rollout (no-grad, done-reset seams).
        with torch.no_grad():
            actions = batch.actions
            if actions.dim() == 2:                       # discrete index form
                actions = F.one_hot(actions.long(),
                                    rssm.action_dim).float()
            out = rssm.observe(batch.obs, actions, done_seq=batch.dones)
            latent = torch.cat([out["h"], out["z"]], dim=-1)   # (N, T, L)

        # 2. Per-row done-reset discounted returns (G=0 seed — as the gym
        #    _train_latent_policy, which also trains windowed, no bootstrap).
        T = batch.horizon
        R = torch.zeros(batch.num_envs, device=DEVICE)
        returns_rev = []
        for t in reversed(range(T)):
            R = batch.rewards[:, t] + self.gamma * (1.0 - batch.dones[:, t]) * R
            returns_rev.append(R)
        returns = torch.stack(returns_rev[::-1], dim=1)        # (N, T)

        # 3. Flatten; A2C over env-flattened minibatches (evaluate_action is
        #    per-transition — the recurrence already ran in observe).
        latent_f = latent.reshape(-1, latent.shape[-1])
        ret_f = returns.reshape(-1)
        act_f = (batch.actions.reshape(-1) if batch.actions.dim() == 2
                 else batch.actions.reshape(-1, batch.actions.shape[-1]))
        M = latent_f.shape[0]
        assert M % n_minibatches == 0, (
            f"rollout size {M} must be divisible by n_minibatches "
            f"{n_minibatches}")
        mb = M // n_minibatches

        actor_t = value_t = ent_t = torch.zeros((), device=DEVICE)
        # One fully on-policy A2C update per rollout (epochs=1,
        # n_minibatches=1 by default). A2C is on-policy and UNCLIPPED, so
        # BOTH re-using the rollout across epochs AND splitting it into
        # minibatches (the policy shifts between minibatch steps, so later
        # minibatches train on off-policy advantages) over-update and
        # collapse a continuous latent policy to a saturated action —
        # observed on MountainCar latent-acting (full throttle, return
        # -99.9). The gym _train_latent_policy is one update per episode.
        for _ in range(epochs):
            # rand().argsort() not randperm — randperm's int64 RNG emits an
            # s64 HLO the TPU's X64 pass cannot lower.
            perm = torch.rand(M, device=DEVICE).argsort()
            for k in range(n_minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                log_prob, entropy, value = self.policy.evaluate_action(
                    latent_f[idx], act_f[idx])
                ret = ret_f[idx]
                adv = ret - value.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                actor_loss = -(adv * log_prob).mean()
                value_loss = F.mse_loss(value, ret)
                entropy_mean = entropy.mean()
                loss = (actor_loss + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy_mean)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.grad_clip)
                self.optimizer.step()
                mark_step()  # XLA: one graph per minibatch update
                actor_t = actor_loss.detach()
                value_t = value_loss.detach()
                ent_t = entropy_mean.detach()

        return {
            "latent/actor_loss": actor_t.item(),
            "latent/value_loss": value_t.item(),
            "latent/entropy": ent_t.item(),
        }
