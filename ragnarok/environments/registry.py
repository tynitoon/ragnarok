"""Environment registry: maps names to configurations."""

from dataclasses import dataclass


@dataclass
class EnvSpec:
    gym_name: str
    obs_dim: int
    action_dim: int
    is_discrete: bool
    reward_threshold: float


REGISTRY: dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        gym_name="CartPole-v1",
        obs_dim=4,
        action_dim=2,
        is_discrete=True,
        reward_threshold=450.0,
    ),
    "mountaincar": EnvSpec(
        gym_name="MountainCar-v0",
        obs_dim=2,
        action_dim=3,
        is_discrete=True,
        reward_threshold=-120.0,
    ),
    "acrobot": EnvSpec(
        gym_name="Acrobot-v1",
        obs_dim=6,
        action_dim=3,
        is_discrete=True,
        reward_threshold=-100.0,
    ),
    # === Continuous control ===
    "pendulum": EnvSpec(
        gym_name="Pendulum-v1",
        obs_dim=3,
        action_dim=1,
        is_discrete=False,
        reward_threshold=-200.0,
    ),
    "mountaincar-continuous": EnvSpec(
        gym_name="MountainCarContinuous-v0",
        obs_dim=2,
        action_dim=1,
        is_discrete=False,
        reward_threshold=90.0,
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    """Get environment specification by name."""
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]
