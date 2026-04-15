"""Recurrent State-Space Model (RSSM) - The World Model / "Intuition Engine".

The RSSM learns a compressed internal representation of the environment.
It encodes observations into latent vectors (not human language) and
predicts how the world evolves in response to actions.

State = (h_t, z_t) where:
    h_t: deterministic recurrent state (GRU hidden, 128-dim)
    z_t: stochastic latent state (sampled, 32-dim)

The agent can "imagine" future trajectories without interacting with
the real environment — this is dream training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class RSSMEncoder(nn.Module):
    """Encodes raw observations into feature vectors."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RSSMCore(nn.Module):
    """Recurrent core: GRU + prior/posterior stochastic state.

    At each timestep:
        1. h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))
        2. Prior:     p(z_t | h_t)
        3. Posterior:  q(z_t | h_t, obs_features_t)
    """

    def __init__(self, stoch_dim: int = 32, hidden_dim: int = 128,
                 action_dim: int = 4, encoder_dim: int = 128):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

        # Pre-GRU projection: concat(z, action) -> hidden_dim
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ELU(),
        )

        # GRU cell
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prior: h_t -> (mean, logstd) of z_t
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, stoch_dim * 2),
        )

        # Posterior: concat(h_t, obs_features) -> (mean, logstd) of z_t
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + encoder_dim, 64),
            nn.ELU(),
            nn.Linear(64, stoch_dim * 2),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h_0, z_0)."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def forward_prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution parameters from deterministic state."""
        params = self.prior(h)
        mean, logstd = params.chunk(2, dim=-1)
        logstd = logstd.clamp(-5.0, 2.0)
        return mean, logstd

    def forward_posterior(self, h: torch.Tensor, obs_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution parameters from h_t and observation features."""
        x = torch.cat([h, obs_features], dim=-1)
        params = self.posterior(x)
        mean, logstd = params.chunk(2, dim=-1)
        logstd = logstd.clamp(-5.0, 2.0)
        return mean, logstd

    def step(self, prev_h: torch.Tensor, prev_z: torch.Tensor,
             prev_action: torch.Tensor) -> torch.Tensor:
        """Single GRU step: (h_{t-1}, z_{t-1}, a_{t-1}) -> h_t."""
        x = torch.cat([prev_z, prev_action], dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev_h)
        return h

    @staticmethod
    def sample(mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """Sample z from Normal(mean, exp(logstd)) with reparameterization."""
        std = logstd.exp()
        dist = Normal(mean, std)
        return dist.rsample()


class EnsembleRSSMCore(nn.Module):
    """Ensemble of N GRU cores sharing the same encoder.

    Each core maintains its own GRU + prior network, producing independent
    prior distributions. Disagreement between cores indicates model uncertainty.
    Posterior network is shared (we always have ground truth observations).
    """

    def __init__(self, n_cores: int = 2, stoch_dim: int = 32,
                 hidden_dim: int = 128, action_dim: int = 4,
                 encoder_dim: int = 128):
        super().__init__()
        self.n_cores = n_cores
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

        # Each core has its own pre_gru, gru, and prior
        self.pre_grus = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.priors = nn.ModuleList()

        for _ in range(n_cores):
            self.pre_grus.append(nn.Sequential(
                nn.Linear(stoch_dim + action_dim, hidden_dim),
                nn.ELU(),
            ))
            self.grus.append(nn.GRUCell(hidden_dim, hidden_dim))
            self.priors.append(nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ELU(),
                nn.Linear(64, stoch_dim * 2),
            ))

        # Shared posterior (same observation -> same posterior)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + encoder_dim, 64),
            nn.ELU(),
            nn.Linear(64, stoch_dim * 2),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        """Return zero-initialized states for all cores: list of (h, z)."""
        states = []
        for _ in range(self.n_cores):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            z = torch.zeros(batch_size, self.stoch_dim, device=device)
            states.append((h, z))
        return states

    def step_all(self, states: list, prev_action: torch.Tensor):
        """Step all cores forward. Returns list of (h_new,) per core."""
        new_hs = []
        for i, (h, z) in enumerate(states):
            x = torch.cat([z, prev_action], dim=-1)
            x = self.pre_grus[i](x)
            h_new = self.grus[i](x, h)
            new_hs.append(h_new)
        return new_hs

    def prior_all(self, hs: list) -> list:
        """Compute prior (mean, logstd) for each core."""
        results = []
        for i, h in enumerate(hs):
            params = self.priors[i](h)
            mean, logstd = params.chunk(2, dim=-1)
            logstd = logstd.clamp(-5.0, 2.0)
            results.append((mean, logstd))
        return results

    def disagreement(self, hs: list) -> torch.Tensor:
        """Compute disagreement (variance of prior means across cores).

        Returns scalar variance per batch element: (batch,)
        """
        priors = self.prior_all(hs)
        means = torch.stack([m for m, _ in priors], dim=0)  # (n_cores, batch, stoch_dim)
        # Variance across cores, mean across stoch_dim
        return means.var(dim=0).mean(dim=-1)  # (batch,)


class RewardPredictor(nn.Module):
    """Predicts scalar reward from (h_t, z_t)."""

    def __init__(self, hidden_dim: int = 128, stoch_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predicts probability that the episode continues (not done)."""

    def __init__(self, hidden_dim: int = 128, stoch_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class RSSM(nn.Module):
    """Complete Recurrent State-Space Model.

    Combines encoder, recurrent core, decoder, reward predictor,
    and continue predictor into a single world model.

    Supports pluggable encoder/decoder for different observation types
    (vector observations use MLP, pixel observations use CNN).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 128, stoch_dim: int = 32,
                 encoder_hidden: int = 128,
                 encoder: nn.Module | None = None,
                 decoder: nn.Module | None = None,
                 ensemble_cores: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim

        # Pluggable encoder/decoder (default: MLP for vector observations)
        self.encoder = encoder or RSSMEncoder(obs_dim, encoder_hidden)
        self.core = RSSMCore(stoch_dim, hidden_dim, action_dim, encoder_hidden)

        # Optional ensemble for uncertainty estimation
        self.ensemble: EnsembleRSSMCore | None = None
        if ensemble_cores > 1:
            self.ensemble = EnsembleRSSMCore(
                n_cores=ensemble_cores, stoch_dim=stoch_dim,
                hidden_dim=hidden_dim, action_dim=action_dim,
                encoder_dim=encoder_hidden,
            )

        self.decoder = decoder or nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, 128),
            nn.ELU(),
            nn.Linear(128, obs_dim),
        )
        self.reward_predictor = RewardPredictor(hidden_dim, stoch_dim)
        self.continue_predictor = ContinuePredictor(hidden_dim, stoch_dim)

    @property
    def state_dim(self) -> int:
        """Total state dimension (h + z), used as input to policy."""
        return self.hidden_dim + self.stoch_dim

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized state."""
        return self.core.initial_state(batch_size, device)

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                ) -> dict[str, torch.Tensor]:
        """Process a sequence of real observations.

        Args:
            obs_seq: (batch, time, obs_dim) - real observations
            action_seq: (batch, time, action_dim) - actions taken (one-hot for discrete)

        Returns:
            Dict with keys: h, z, prior_mean, prior_logstd,
                            post_mean, post_logstd, recon_obs, reward_pred, continue_pred
        """
        batch_size, seq_len, _ = obs_seq.shape
        device = obs_seq.device

        h, z = self.initial_state(batch_size, device)

        # Pre-encode all observations
        obs_flat = obs_seq.reshape(batch_size * seq_len, -1)
        features_flat = self.encoder(obs_flat)
        features = features_flat.reshape(batch_size, seq_len, -1)

        # Collect outputs
        hs, zs = [], []
        prior_means, prior_logstds = [], []
        post_means, post_logstds = [], []

        for t in range(seq_len):
            # Action at t-1 (zero for first step)
            if t == 0:
                prev_action = torch.zeros(batch_size, self.action_dim, device=device)
            else:
                prev_action = action_seq[:, t - 1]

            # GRU step
            h = self.core.step(h, z, prev_action)

            # Prior and posterior
            prior_mean, prior_logstd = self.core.forward_prior(h)
            post_mean, post_logstd = self.core.forward_posterior(h, features[:, t])

            # Sample from posterior (training uses posterior)
            z = self.core.sample(post_mean, post_logstd)

            hs.append(h)
            zs.append(z)
            prior_means.append(prior_mean)
            prior_logstds.append(prior_logstd)
            post_means.append(post_mean)
            post_logstds.append(post_logstd)

        # Stack along time dimension
        hs = torch.stack(hs, dim=1)          # (batch, time, hidden_dim)
        zs = torch.stack(zs, dim=1)          # (batch, time, stoch_dim)
        prior_means = torch.stack(prior_means, dim=1)
        prior_logstds = torch.stack(prior_logstds, dim=1)
        post_means = torch.stack(post_means, dim=1)
        post_logstds = torch.stack(post_logstds, dim=1)

        # Decode observations, predict rewards and continues
        hz = torch.cat([hs, zs], dim=-1)
        hz_flat = hz.reshape(batch_size * seq_len, -1)

        recon_obs = self.decoder(hz_flat).reshape(batch_size, seq_len, -1)
        reward_pred = self.reward_predictor(
            hs.reshape(-1, self.hidden_dim),
            zs.reshape(-1, self.stoch_dim)
        ).reshape(batch_size, seq_len)
        continue_pred = self.continue_predictor(
            hs.reshape(-1, self.hidden_dim),
            zs.reshape(-1, self.stoch_dim)
        ).reshape(batch_size, seq_len)

        return {
            "h": hs, "z": zs,
            "prior_mean": prior_means, "prior_logstd": prior_logstds,
            "post_mean": post_means, "post_logstd": post_logstds,
            "recon_obs": recon_obs,
            "reward_pred": reward_pred,
            "continue_pred": continue_pred,
        }

    def imagine(self, initial_h: torch.Tensor, initial_z: torch.Tensor,
                policy_fn, horizon: int) -> dict[str, torch.Tensor]:
        """Imagine a trajectory using the learned world model.

        Args:
            initial_h: (batch, hidden_dim) - starting deterministic state
            initial_z: (batch, stoch_dim) - starting stochastic state
            policy_fn: callable(h, z) -> action tensor
            horizon: number of imagination steps

        Returns:
            Dict with: h, z, action, reward_pred, continue_pred
        """
        h, z = initial_h, initial_z
        hs, zs, actions = [h], [z], []
        reward_preds, continue_preds = [], []

        for _ in range(horizon):
            action = policy_fn(h, z)
            h = self.core.step(h, z, action)

            prior_mean, prior_logstd = self.core.forward_prior(h)
            z = self.core.sample(prior_mean, prior_logstd)

            reward_pred = self.reward_predictor(h, z)
            continue_pred = self.continue_predictor(h, z)

            hs.append(h)
            zs.append(z)
            actions.append(action)
            reward_preds.append(reward_pred)
            continue_preds.append(continue_pred)

        return {
            "h": torch.stack(hs, dim=1),           # (batch, horizon+1, hidden_dim)
            "z": torch.stack(zs, dim=1),            # (batch, horizon+1, stoch_dim)
            "action": torch.stack(actions, dim=1),   # (batch, horizon, action_dim)
            "reward_pred": torch.stack(reward_preds, dim=1),  # (batch, horizon)
            "continue_pred": torch.stack(continue_preds, dim=1),  # (batch, horizon)
        }

    def loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
             reward_seq: torch.Tensor, done_seq: torch.Tensor,
             kl_weight: float = 0.1, free_nats: float = 1.0) -> dict[str, torch.Tensor]:
        """Compute total RSSM loss.

        Returns dict with: total_loss, recon_loss, reward_loss, continue_loss, kl_loss
        """
        outputs = self.observe(obs_seq, action_seq)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs["recon_obs"], obs_seq)

        # Reward prediction loss
        reward_loss = F.mse_loss(outputs["reward_pred"], reward_seq)

        # Continue prediction loss (class-weighted binary cross-entropy)
        # Done examples are rare (~5%), so we upweight them heavily
        continue_target = 1.0 - done_seq.float()
        # Weight: done steps (target=0) get 10x weight
        pos_weight = torch.ones_like(continue_target)
        done_mask = done_seq.float() > 0.5
        pos_weight[done_mask] = 0.1  # Lower pos_weight for done=True (target=0)
        # Use per-element weighting
        continue_loss_raw = F.binary_cross_entropy_with_logits(
            outputs["continue_pred"], continue_target, reduction="none"
        )
        # Weight the done examples 10x more
        weights = torch.where(done_mask, 10.0, 1.0)
        continue_loss = (continue_loss_raw * weights).mean()

        # KL divergence between posterior and prior
        posterior = Normal(outputs["post_mean"], outputs["post_logstd"].exp())
        prior = Normal(outputs["prior_mean"], outputs["prior_logstd"].exp())
        kl = kl_divergence(posterior, prior)
        # Free nats: ignore KL below threshold (per dimension)
        kl = torch.clamp(kl, min=free_nats / self.stoch_dim).sum(dim=-1).mean()

        total_loss = recon_loss + reward_loss + continue_loss + kl_weight * kl

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
            "kl_loss": kl,
        }

    def encode_observation(self, obs: torch.Tensor, h: torch.Tensor,
                           z: torch.Tensor, action: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single observation into (h_t, z_t) given previous state.

        Used during real-environment interaction (not training).
        """
        h = self.core.step(h, z, action)
        features = self.encoder(obs)
        post_mean, post_logstd = self.core.forward_posterior(h, features)
        z = self.core.sample(post_mean, post_logstd)
        return h, z

    # ── Cross-dim transfer: split the RSSM into env-agnostic vs per-env ──
    #
    # Phase 3 pre-launch, Bug E: the original skill serialization stored
    # only the `LatentPolicyHead.shared` MLP + `critic_head` weights. But
    # that MLP consumes `cat(h, z)` — features produced by the RSSM. If
    # the target env's RSSM is fresh-random at transfer time, those
    # features are noise, and the transferred policy trunk cannot
    # exploit any source-task structure. Result: null transfer.
    #
    # The correct fix is to co-transfer the env-agnostic subset of the
    # RSSM: the GRU core, the prior MLP, and the posterior MLP — all of
    # which operate purely on fixed-dim (hidden_dim, stoch_dim,
    # encoder_hidden) tensors. The per-env components (encoder,
    # core.pre_gru, decoder, reward_predictor, continue_predictor) stay
    # fresh-random on the target, which is correct — they have to learn
    # the new obs/action dims and the new reward/termination structure.
    #
    # Rationale per layer:
    #   - encoder:            obs_dim → hidden  — depends on obs_dim
    #   - core.pre_gru:       (stoch_dim + action_dim) → hidden_dim — depends on action_dim
    #   - core.gru:           hidden_dim → hidden_dim — SHAREABLE
    #   - core.prior:         hidden_dim → stoch_dim*2 — SHAREABLE
    #   - core.posterior:     (hidden_dim + encoder_hidden) → stoch_dim*2 — SHAREABLE
    #   - decoder:            (hidden+stoch) → obs_dim — depends on obs_dim
    #   - reward_predictor:   env-specific reward scale/structure
    #   - continue_predictor: env-specific episode termination structure

    _TRANSFERABLE_PREFIXES = (
        "core.gru.",
        "core.prior.",
        "core.posterior.",
    )

    def transferable_state_dict(self) -> dict:
        """Return only the env-agnostic subset of the RSSM state_dict.

        Transferable keys are prefixed with one of ``core.gru.``,
        ``core.prior.``, or ``core.posterior.``. Everything else is
        excluded because it either depends on obs_dim / action_dim
        (encoder, pre_gru, decoder) or on env-specific semantics
        (reward, continue predictors).

        This is the slice that ``save_skill`` serializes and that
        ``try_transfer`` loads when the target env has different obs or
        action dimensions from the source.
        """
        sd = self.state_dict()
        return {k: v for k, v in sd.items()
                if k.startswith(self._TRANSFERABLE_PREFIXES)}

    def load_transferable_state_dict(self, sd: dict, strict: bool = True):
        """Load only the env-agnostic subset of the state_dict.

        Preserves the per-env components (encoder, decoder, pre_gru,
        reward/continue predictors) untouched — they retain whatever
        random initialization the target-env RSSM was built with and
        will be trained up from there.

        Args:
            sd: state dict produced by ``transferable_state_dict()``.
            strict: if True, raise ValueError on non-transferable keys,
                missing target keys, or shape mismatches. Non-strict
                mode silently skips incompatible entries — used only
                by old checkpoint migration.
        """
        current = self.state_dict()
        unknown_keys = [k for k in sd
                        if not k.startswith(self._TRANSFERABLE_PREFIXES)]
        if unknown_keys and strict:
            raise ValueError(
                f"load_transferable_state_dict received non-transferable "
                f"keys: {unknown_keys}. Only {self._TRANSFERABLE_PREFIXES} "
                f"are allowed."
            )
        for k, v in sd.items():
            if not k.startswith(self._TRANSFERABLE_PREFIXES):
                continue
            if k not in current:
                if strict:
                    raise ValueError(
                        f"Key {k!r} in transferable state_dict does not "
                        f"exist on target RSSM.")
                continue
            if current[k].shape != v.shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch on {k!r}: target expects "
                        f"{tuple(current[k].shape)}, got {tuple(v.shape)}. "
                        f"Transferable subset should be env-agnostic — "
                        f"if shapes differ, the RSSM hidden_dim/stoch_dim "
                        f"was changed between source and target.")
                continue
            current[k] = v
        self.load_state_dict(current)

    def transferable_params(self):
        """Iterate parameters in the env-agnostic transferable subset.

        Used by WorldModelTrainer to put these params in a separate
        optimizer group so the learning rate can be scaled independently
        during post-transfer warmup (Bug E Phase 5 fix).
        """
        yield from self.core.gru.parameters()
        yield from self.core.prior.parameters()
        yield from self.core.posterior.parameters()

    def non_transferable_params(self):
        """Iterate parameters NOT in the transferable subset.

        Complement of ``transferable_params()``. Partition is exact:
        every ``self.parameters()`` lands in one or the other with no
        overlap (see test_rssm_transfer.py).
        """
        transferable_ids = {id(p) for p in self.transferable_params()}
        for p in self.parameters():
            if id(p) not in transferable_ids:
                yield p
