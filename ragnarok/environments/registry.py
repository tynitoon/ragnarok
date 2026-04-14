"""Environment registry: maps names to configurations."""

from dataclasses import dataclass
from ragnarok.environments.dmcontrol import DMC_AVAILABLE, DMC_TASKS, get_dmc_obs_dim, get_dmc_action_dim


@dataclass
class EnvSpec:
    gym_name: str
    obs_dim: int
    action_dim: int
    is_discrete: bool
    reward_threshold: float
    pixel_obs: bool = False


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
    # === Pixel observation variants ===
    "cartpole-pixels": EnvSpec(
        gym_name="CartPole-v1",
        obs_dim=3 * 64 * 64,  # CHW flattened
        action_dim=2,
        is_discrete=True,
        reward_threshold=400.0,
        pixel_obs=True,
    ),
    "pendulum-pixels": EnvSpec(
        gym_name="Pendulum-v1",
        obs_dim=3 * 64 * 64,
        action_dim=1,
        is_discrete=False,
        reward_threshold=-300.0,
        pixel_obs=True,
    ),
}


# === DMControl Suite (requires dm_control) ===
for _dmc_name, (_domain, _task) in DMC_TASKS.items():
    REGISTRY[_dmc_name] = EnvSpec(
        gym_name=f"dmc:{_domain}-{_task}",
        obs_dim=get_dmc_obs_dim(_domain, _task),
        action_dim=get_dmc_action_dim(_domain, _task),
        is_discrete=False,
        reward_threshold=800.0,  # DMC rewards are 0-1 per step, 1000 steps
    )


def get_env_spec(name: str) -> EnvSpec:
    """Get environment specification by name."""
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


def make_env(name: str, seed: int | None = None, normalize: bool = True):
    """Factory: create the right env wrapper for a registry name.

    Returns a RagnarokEnv for Gymnasium envs, DMControlEnv for DMC envs.
    Both share the same interface (reset, step, close, etc.).
    """
    spec = get_env_spec(name)

    if spec.gym_name.startswith("dmc:"):
        from ragnarok.environments.dmcontrol import DMControlEnv, DMC_TASKS
        if not DMC_AVAILABLE:
            raise ImportError(
                f"dm_control is required for {name}. "
                "Install with: pip install dm_control"
            )
        domain, task = DMC_TASKS[name]
        return DMControlEnv(domain, task, seed=seed, normalize=normalize)

    from ragnarok.environments.wrapper import RagnarokEnv
    return RagnarokEnv(
        spec.gym_name, seed=seed,
        pixel_obs=spec.pixel_obs, normalize=normalize,
    )
