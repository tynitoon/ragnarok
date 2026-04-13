"""Centralized configuration with dataclass validation."""

from dataclasses import dataclass, field


@dataclass
class WorldModelConfig:
    obs_dim: int = 8
    action_dim: int = 4
    hidden_dim: int = 128
    stoch_dim: int = 32
    encoder_hidden: int = 128
    lr: float = 3e-4
    kl_weight: float = 0.1
    free_nats: float = 1.0
    grad_clip: float = 100.0
    train_steps: int = 100
    train_every: int = 1000
    sequence_length: int = 50
    batch_size: int = 50


@dataclass
class PolicyConfig:
    hidden_dim: int = 128
    mid_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4
    imagination_horizon: int = 15
    imagination_batch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_bonus: float = 3e-3
    grad_clip: float = 100.0
    train_steps: int = 100


@dataclass
class MemoryConfig:
    replay_capacity: int = 1_000_000
    episodic_capacity: int = 50_000
    episodic_k: int = 5


@dataclass
class SkillConfig:
    crystallization_window: int = 100
    min_episodes: int = 200
    skills_dir: str = "skills_data"
    thresholds: dict = field(default_factory=lambda: {
        "CartPole-v1": 450.0,
        "MountainCar-v0": -120.0,
        "Acrobot-v1": -100.0,
        "Pendulum-v1": -200.0,
        "MountainCarContinuous-v0": 90.0,
    })


@dataclass
class CuriosityConfig:
    enabled: bool = True
    beta: float = 0.1         # Weight of intrinsic vs extrinsic reward
    lr: float = 1e-3          # Predictor learning rate
    hidden_dim: int = 64      # Predictor network hidden size
    grad_clip: float = 1.0


@dataclass
class NormalizerConfig:
    clip: float = 5.0
    warmup_steps: int = 1000
    decay: float = 0.99


@dataclass
class RagnarokConfig:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        assert self.world_model.hidden_dim > 0, "hidden_dim must be positive"
        assert self.world_model.stoch_dim > 0, "stoch_dim must be positive"
        assert self.policy.gamma > 0 and self.policy.gamma <= 1, "gamma must be in (0, 1]"
        assert self.policy.gae_lambda >= 0 and self.policy.gae_lambda <= 1, "lambda must be in [0, 1]"
