"""Accelerator-resident batched environments.

The environment physics runs ON the XLA/CUDA device as batched tensor
ops: stepping N environments is one batched computation, not a Python
loop of N gym calls. This removes the serial env-loop that caps
vectorized collection (see scripts/bench_vec_collection.py — at N=256,
~96% of collection time was the serial loop).

Phase 1 of the TPU re-architecture. CartPole + MountainCarContinuous —
the primary cartpole_mcc pair. Physics matches gymnasium exactly
(classic_control.cartpole / continuous_mountain_car, euler integrator),
so the task — and therefore the calibration — is unchanged.

Design for XLA: N is fixed, every op is batched, branching is torch.where
(no data-dependent indexing), auto-reset is a masked blend, and step()
touches no .cpu()/.item() — the whole collection loop stays on-device.
"""

import math
import torch

from ragnarok.infrastructure.device import DEVICE


class DeviceVecCartPole:
    """N CartPole-v1 environments, device-resident and batched.

    State (N, 4) = [x, x_dot, theta, theta_dot] on DEVICE. step() is
    batched tensor ops with auto-reset on termination/truncation.
    obs == state (CartPole's observation is its raw state).
    """

    obs_dim = 4
    action_dim = 2
    is_discrete = True

    _GRAVITY = 9.8
    _MASSPOLE = 0.1
    _TOTAL_MASS = 1.1
    _LENGTH = 0.5
    _POLEMASS_LENGTH = 0.05
    _FORCE_MAG = 10.0
    _TAU = 0.02
    _THETA_THRESH = 12 * 2 * math.pi / 360
    _X_THRESH = 2.4
    _MAX_STEPS = 500

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def reset(self) -> torch.Tensor:
        # gym: uniform(-0.05, 0.05) on all 4 state dims.
        self.state = (torch.rand(self.num_envs, 4, device=DEVICE) - 0.5) * 0.1
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        return self.state

    def step(self, action: torch.Tensor):
        """action: (N,) int {0,1} or (N, 2) one-hot. All-batched, on-device."""
        if action.dim() == 2:
            action = action.argmax(dim=-1)
        # force = +10 for action 1, -10 for action 0 (no data-dependent branch)
        force = (action == 1).float() * (2.0 * self._FORCE_MAG) - self._FORCE_MAG

        x, x_dot, theta, theta_dot = self.state.unbind(dim=-1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        temp = (force + self._POLEMASS_LENGTH * theta_dot ** 2 * sin_t) / self._TOTAL_MASS
        thetaacc = (self._GRAVITY * sin_t - cos_t * temp) / (
            self._LENGTH * (4.0 / 3.0 - self._MASSPOLE * cos_t ** 2 / self._TOTAL_MASS))
        xacc = temp - self._POLEMASS_LENGTH * thetaacc * cos_t / self._TOTAL_MASS

        # euler integrator (matches gym's default kinematics_integrator)
        x = x + self._TAU * x_dot
        x_dot = x_dot + self._TAU * xacc
        theta = theta + self._TAU * theta_dot
        theta_dot = theta_dot + self._TAU * thetaacc
        new_state = torch.stack([x, x_dot, theta, theta_dot], dim=-1)
        self.steps = self.steps + 1.0

        terminated = (x.abs() > self._X_THRESH) | (theta.abs() > self._THETA_THRESH)
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated
        reward = torch.ones(self.num_envs, device=DEVICE)  # +1 every step

        # Auto-reset: masked blend — done envs get a fresh uniform state.
        fresh = (torch.rand(self.num_envs, 4, device=DEVICE) - 0.5) * 0.1
        self.state = torch.where(done.unsqueeze(-1), fresh, new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceVecMountainCarContinuous:
    """N MountainCarContinuous-v0 environments, device-resident and batched.

    State (N, 2) = [position, velocity]. Physics matches gymnasium's
    continuous_mountain_car. obs == state.
    """

    obs_dim = 2
    action_dim = 1
    is_discrete = False

    _MIN_POS = -1.2
    _MAX_POS = 0.6
    _MAX_SPEED = 0.07
    _GOAL_POS = 0.45
    _GOAL_VEL = 0.0
    _POWER = 0.0015
    _MAX_STEPS = 999

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def reset(self) -> torch.Tensor:
        # gym: position uniform(-0.6, -0.4), velocity 0.
        pos = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        vel = torch.zeros(self.num_envs, device=DEVICE)
        self.state = torch.stack([pos, vel], dim=-1)
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        return self.state

    def step(self, action: torch.Tensor):
        """action: (N,) or (N, 1) float, clamped to [-1, 1]. All on-device."""
        force = action.reshape(self.num_envs).clamp(-1.0, 1.0)
        pos, vel = self.state.unbind(dim=-1)

        vel = vel + force * self._POWER - 0.0025 * torch.cos(3.0 * pos)
        vel = vel.clamp(-self._MAX_SPEED, self._MAX_SPEED)
        pos = (pos + vel).clamp(self._MIN_POS, self._MAX_POS)
        # at the left wall with negative velocity, velocity is zeroed
        vel = torch.where((pos <= self._MIN_POS) & (vel < 0),
                          torch.zeros_like(vel), vel)
        new_state = torch.stack([pos, vel], dim=-1)
        self.steps = self.steps + 1.0

        terminated = (pos >= self._GOAL_POS) & (vel >= self._GOAL_VEL)
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated
        reward = terminated.float() * 100.0 - 0.1 * force ** 2

        pos0 = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        fresh = torch.stack([pos0, torch.zeros_like(pos0)], dim=-1)
        self.state = torch.where(done.unsqueeze(-1), fresh, new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceVecMountainCarContinuousHard(DeviceVecMountainCarContinuous):
    """MountainCarContinuous with a weaker engine — a graded-difficulty
    variant for the v3.10 corrected-transfer experiment.

    Same dynamics family as DeviceVecMountainCarContinuous: identical
    obs/action dims, hill term, gravity, velocity clamp and Euler
    integration — only the engine ``_POWER`` is reduced, so the car needs
    more energy-pumping skill to reach the goal. Because the physics is
    otherwise identical, an RSSM world-model core trained on the standard
    variant carries genuinely transferable dynamics structure — this is
    the "shared-dynamics, graded-difficulty" pair v3.10 calls for.
    """

    _POWER = 0.0011   # vs 0.0015 standard — a ~27%-weaker engine


class DeviceVecPendulum:
    """N Pendulum-v1 environments, device-resident and batched.

    Physical state (theta, theta_dot); observation is (cos theta,
    sin theta, theta_dot) — obs_dim 3, while the physics has only 2
    DOF (the cos/sin pair encodes the angle without a discontinuity at
    +/- pi). Physics matches gymnasium's Pendulum-v1 exactly: a single
    nonlinear DOF with gravity, torque actuation and a velocity clamp.
    Reward is a per-step cost ``-(theta_n^2 + 0.1 thetadot^2 + 0.001 u^2)``
    where ``theta_n`` is theta wrapped to [-pi, pi].

    Why this env exists here: it shares dynamics structure (energy
    pumping in a 1-DOF nonlinear gravitational potential, continuous
    torque control) with MountainCarContinuous despite having a
    different obs_dim — the v3.16 heterogeneous-dim transfer test
    needs a source whose dynamics are close enough to MCC's that the
    env-agnostic RSSM core can plausibly carry useful structure
    (CartPole's pole-balancing dynamics were too far, per v3.15).
    """

    obs_dim = 3
    action_dim = 1
    is_discrete = False

    _G = 10.0
    _M = 1.0
    _L = 1.0
    _MAX_SPEED = 8.0
    _MAX_TORQUE = 2.0
    _DT = 0.05
    _MAX_STEPS = 200

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def _obs(self) -> torch.Tensor:
        return torch.stack(
            [torch.cos(self._th), torch.sin(self._th), self._thdot], dim=-1)

    def reset(self) -> torch.Tensor:
        # gym: theta uniform(-pi, pi), theta_dot uniform(-1, 1).
        self._th = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 2.0 * math.pi
        self._thdot = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 2.0
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        self.state = self._obs()
        return self.state

    def step(self, action: torch.Tensor):
        """action: (N,) or (N, 1) float, clamped to [-2, 2] (torque)."""
        u = action.reshape(self.num_envs).clamp(-self._MAX_TORQUE,
                                                self._MAX_TORQUE)
        # angle_normalize: ((th + pi) mod 2pi) - pi
        th_n = ((self._th + math.pi) % (2.0 * math.pi)) - math.pi
        cost = th_n ** 2 + 0.1 * self._thdot ** 2 + 0.001 * u ** 2
        reward = -cost                                  # gym returns -cost

        # Euler integrator (matches gym's Pendulum-v1 dynamics).
        thdot_new = self._thdot + (
            3.0 * self._G / (2.0 * self._L) * torch.sin(self._th)
            + 3.0 / (self._M * self._L ** 2) * u) * self._DT
        thdot_new = thdot_new.clamp(-self._MAX_SPEED, self._MAX_SPEED)
        th_new = self._th + thdot_new * self._DT
        self.steps = self.steps + 1.0

        # Pendulum-v1 has no termination — only truncation at MAX_STEPS.
        terminated = torch.zeros(self.num_envs, device=DEVICE, dtype=torch.bool)
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated

        fresh_th = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 2.0 * math.pi
        fresh_thdot = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 2.0
        self._th = torch.where(done, fresh_th, th_new)
        self._thdot = torch.where(done, fresh_thdot, thdot_new)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        self.state = self._obs()
        return self.state, reward, terminated, truncated, done


class DeviceVecCartPoleOnHill:
    """N CartPole-on-Hill composite environments — a car on the MCC hill
    with a pole balanced on top. The agent applies a continuous engine
    force; it must climb the hill (energy pumping, MCC skill) WHILE
    keeping the pole upright (balance, CartPole skill). Neither sub-skill
    alone suffices — aggressive driving swings the pole; gentle driving
    cannot climb. This is the v3.19 composite task for the multi-skill
    transfer experiment.

    obs (4) = [cart_pos, cart_vel, pole_angle, pole_angvel] (CART on
    the hill, POLE on the cart). action (1) = continuous engine force.

    Physics: the cart follows MCC dynamics on its x-axis (hill gravity
    via cos(3 pos), velocity clamp, position clamped); the pole follows
    CartPole dynamics under an effective horizontal force scaled from
    the same action (action_scaled = action * pole_force_scale). The
    coupling — the pole reacts to the agent's force, not directly to
    the hill — is a simplification (a fully coupled cart-pole-on-curve
    is far more complex) but preserves the composite-skill nature.

    Reward is SPARSE: +100 at goal reach (cart_pos >= goal_pos AND pole
    still upright), 0 otherwise; minus a small force^2 control cost.
    Episode terminates on goal-reach OR pole fall (theta > pi/4),
    truncates at 999 steps.
    """

    obs_dim = 4
    action_dim = 1
    is_discrete = False

    # Cart-on-hill (MCC) parameters
    _MIN_POS = -1.2
    _MAX_POS = 0.6
    _MAX_SPEED = 0.07
    _GOAL_POS = 0.45
    _GOAL_VEL = 0.0
    _CART_POWER = 0.0015         # same as standard MCC

    # Pole-on-cart (CartPole) parameters
    _G = 9.8
    _MASSPOLE = 0.1
    _TOTAL_MASS = 1.1
    _LENGTH = 0.5
    _POLEMASS_LENGTH = 0.05
    _POLE_FORCE_SCALE = 10.0     # action in [-1, 1] -> pole feels +/-10 N
    _TAU = 0.02
    _THETA_THRESH = math.pi / 4   # 45 deg — generous; the agent can lean
    _MAX_STEPS = 999

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def reset(self) -> torch.Tensor:
        pos = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        vel = torch.zeros(self.num_envs, device=DEVICE)
        theta = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        thdot = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        self.state = torch.stack([pos, vel, theta, thdot], dim=-1)
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        return self.state

    def step(self, action: torch.Tensor):
        """action: (N,) or (N, 1) float, clamped to [-1, 1]."""
        u = action.reshape(self.num_envs).clamp(-1.0, 1.0)
        pos, vel, theta, thdot = self.state.unbind(dim=-1)

        # --- Cart on hill (MCC dynamics) ---
        vel_new = vel + u * self._CART_POWER - 0.0025 * torch.cos(3.0 * pos)
        vel_new = vel_new.clamp(-self._MAX_SPEED, self._MAX_SPEED)
        pos_new = (pos + vel_new).clamp(self._MIN_POS, self._MAX_POS)
        # At the left wall with negative velocity, velocity is zeroed.
        vel_new = torch.where((pos_new <= self._MIN_POS) & (vel_new < 0),
                              torch.zeros_like(vel_new), vel_new)

        # --- Pole on cart (CartPole dynamics) ---
        force = u * self._POLE_FORCE_SCALE
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        temp = (force + self._POLEMASS_LENGTH * thdot ** 2 * sin_t) / self._TOTAL_MASS
        thetaacc = (self._G * sin_t - cos_t * temp) / (
            self._LENGTH * (4.0 / 3.0 - self._MASSPOLE * cos_t ** 2 / self._TOTAL_MASS))
        theta_new = theta + self._TAU * thdot
        thdot_new = thdot + self._TAU * thetaacc

        new_state = torch.stack([pos_new, vel_new, theta_new, thdot_new], dim=-1)
        self.steps = self.steps + 1.0

        # --- Termination + reward ---
        pole_fell = theta_new.abs() > self._THETA_THRESH
        reached_goal = (pos_new >= self._GOAL_POS) & (vel_new >= self._GOAL_VEL) & ~pole_fell
        terminated = pole_fell | reached_goal
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated

        # Sparse: +100 only if the cart reaches the goal with pole upright.
        reward = 100.0 * reached_goal.float() - 0.1 * u ** 2

        # Auto-reset.
        pos0 = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        vel0 = torch.zeros(self.num_envs, device=DEVICE)
        th0 = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        thdot0 = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        fresh = torch.stack([pos0, vel0, th0, thdot0], dim=-1)
        self.state = torch.where(done.unsqueeze(-1), fresh, new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceRunningNormalizer:
    """Device-resident running observation normalizer (batched Welford).

    The device-env counterpart of ragnarok.core.normalizer.RunningNormalizer:
    tracks running mean/variance, normalizes observations to ~unit scale
    and clips. All state lives in device tensors and update() folds in a
    whole batch — no host sync, fixed shapes, XLA-clean.

    Collection-loop contract: the stats are read-only WHILE a rollout is
    collected (so every step of one rollout sees a single consistent
    scaling, and the obs the policy acts on == the obs it trains on);
    update() is called once BETWEEN rollouts with that rollout's raw obs.

    Unlike RunningNormalizer there is no 1000-step raw warmup: the device
    path collects N*T >> 1000 obs in the first rollout, and at init
    (mean=0, var=1) normalize() is already ~identity, so warmup is moot.
    """

    def __init__(self, obs_dim: int, clip: float = 5.0):
        self.clip = clip
        self.mean = torch.zeros(obs_dim, device=DEVICE)
        self.var = torch.ones(obs_dim, device=DEVICE)
        self._m2 = torch.zeros(obs_dim, device=DEVICE)
        self.count = torch.zeros((), device=DEVICE)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold a batch of raw observations (M, obs_dim) into the stats.

        Chan's parallel variance — equal to RunningNormalizer's sequential
        Welford up to floating-point summation order.
        """
        m = batch.shape[0]
        b_mean = batch.mean(dim=0)
        b_m2 = ((batch - b_mean) ** 2).sum(dim=0)
        delta = b_mean - self.mean
        tot = self.count + m
        self.mean = self.mean + delta * (m / tot)
        self._m2 = self._m2 + b_m2 + delta ** 2 * (self.count * m / tot)
        self.count = tot
        self.var = (self._m2 / torch.clamp(tot - 1.0, min=1.0)).clamp(min=1e-6)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """(obs - mean) / std, clipped to +-clip. ~identity at init."""
        return ((obs - self.mean) / torch.sqrt(self.var)).clamp(-self.clip, self.clip)
