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


class DeviceVecCartPoleOnHillClimbOnly(DeviceVecCartPoleOnHill):
    """Cart-pole-on-hill with a RIGID pole — the pole never rotates and
    never falls. Isolates the CLIMBING sub-skill of the composite: the
    agent only has to drive the cart up the hill. Same 4-d obs / 1-d
    action / goal / reward shape as the full composite, so a policy
    trained here is dimensionally drop-in for the composite. The pole
    obs dims (theta, theta_dot) are present but held at 0.

    v3.24 source task A (the "climb" skill).
    """

    def step(self, action: torch.Tensor):
        u = action.reshape(self.num_envs).clamp(-1.0, 1.0)
        pos, vel, _theta, _thdot = self.state.unbind(dim=-1)

        vel_new = vel + u * self._CART_POWER - 0.0025 * torch.cos(3.0 * pos)
        vel_new = vel_new.clamp(-self._MAX_SPEED, self._MAX_SPEED)
        pos_new = (pos + vel_new).clamp(self._MIN_POS, self._MAX_POS)
        vel_new = torch.where((pos_new <= self._MIN_POS) & (vel_new < 0),
                              torch.zeros_like(vel_new), vel_new)
        # Pole is rigid — theta, theta_dot stay 0.
        zero = torch.zeros_like(pos_new)
        new_state = torch.stack([pos_new, vel_new, zero, zero], dim=-1)
        self.steps = self.steps + 1.0

        reached_goal = (pos_new >= self._GOAL_POS) & (vel_new >= self._GOAL_VEL)
        terminated = reached_goal
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated
        reward = 100.0 * reached_goal.float() - 0.1 * u ** 2

        pos0 = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        z = torch.zeros_like(pos0)
        fresh = torch.stack([pos0, z, z, z], dim=-1)
        self.state = torch.where(done.unsqueeze(-1), fresh, new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceVecCartPoleOnHillBalanceOnly(DeviceVecCartPoleOnHill):
    """Cart-pole-on-hill on FLAT ground — the hill-gravity term is
    removed, so climbing is trivial; the only challenge is keeping the
    pole upright while driving to the goal. Isolates the BALANCING
    sub-skill of the composite. Same 4-d obs / 1-d action / goal /
    reward shape as the full composite.

    v3.24 source task B (the "balance" skill).
    """

    def step(self, action: torch.Tensor):
        u = action.reshape(self.num_envs).clamp(-1.0, 1.0)
        pos, vel, theta, thdot = self.state.unbind(dim=-1)

        # Flat ground — NO hill-gravity ``- 0.0025*cos(3*pos)`` term.
        vel_new = (vel + u * self._CART_POWER).clamp(-self._MAX_SPEED,
                                                     self._MAX_SPEED)
        pos_new = (pos + vel_new).clamp(self._MIN_POS, self._MAX_POS)
        vel_new = torch.where((pos_new <= self._MIN_POS) & (vel_new < 0),
                              torch.zeros_like(vel_new), vel_new)

        # Pole dynamics — identical to the full composite.
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

        pole_fell = theta_new.abs() > self._THETA_THRESH
        reached_goal = (pos_new >= self._GOAL_POS) & (vel_new >= self._GOAL_VEL) & ~pole_fell
        terminated = pole_fell | reached_goal
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated
        reward = 100.0 * reached_goal.float() - 0.1 * u ** 2

        pos0 = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        vel0 = torch.zeros_like(pos0)
        th0 = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        thdot0 = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        fresh = torch.stack([pos0, vel0, th0, thdot0], dim=-1)
        self.state = torch.where(done.unsqueeze(-1), fresh, new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceVecNavigateThenBalance:
    """Sequential two-phase composite for the v3.25 hierarchical
    composition test. obs (5) = [cart_pos, cart_vel, pole_angle,
    pole_angvel, phase].

    mode='composite': PHASE 0 — drive the cart up the MCC hill to the
      goal (the pole is rigid, irrelevant). On reaching the goal the
      cart freezes and PHASE 1 begins — the pole activates and must be
      kept upright; +1 reward per phase-1 step with the pole up, the
      episode ends on pole-fall. The two phases need two DIFFERENT
      skills used at DIFFERENT TIMES — no shared-actuator conflict
      (the v3.24 design flaw, where climb and balance fought over one
      actuator simultaneously).
    mode='nav': the phase-0 task in isolation — reach the goal for
      +100, episode ends. Trains the NAVIGATE skill.
    mode='balance': starts already in phase 1 — cart frozen at the
      goal, pole active from step 0, +1/step pole-up. Trains the
      BALANCE skill.

    All three modes share the 5-d obs / 1-d action, so a policy trained
    on a sub-skill is dimensionally drop-in for the composite.
    """

    obs_dim = 5
    action_dim = 1
    is_discrete = False

    _MIN_POS = -1.2
    _MAX_POS = 0.6
    _MAX_SPEED = 0.07
    _GOAL_POS = 0.45
    _CART_POWER = 0.0015

    _G = 9.8
    _MASSPOLE = 0.1
    _TOTAL_MASS = 1.1
    _LENGTH = 0.5
    _POLEMASS_LENGTH = 0.05
    _POLE_FORCE_SCALE = 10.0
    _TAU = 0.02
    _THETA_THRESH = math.pi / 4
    _MAX_STEPS = 999

    def __init__(self, num_envs: int, mode: str = "composite"):
        assert mode in ("composite", "nav", "balance")
        self.num_envs = num_envs
        self.mode = mode
        self.reset()

    def _fresh(self) -> torch.Tensor:
        pos = torch.rand(self.num_envs, device=DEVICE) * 0.2 - 0.6
        vel = torch.zeros(self.num_envs, device=DEVICE)
        theta = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        thdot = (torch.rand(self.num_envs, device=DEVICE) - 0.5) * 0.1
        if self.mode == "balance":
            # start AT the goal, already in phase 1
            pos = torch.full_like(pos, self._GOAL_POS)
            phase = torch.ones(self.num_envs, device=DEVICE)
        else:
            phase = torch.zeros(self.num_envs, device=DEVICE)
        return torch.stack([pos, vel, theta, thdot, phase], dim=-1)

    def reset(self) -> torch.Tensor:
        self.state = self._fresh()
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        return self.state

    def step(self, action: torch.Tensor):
        u = action.reshape(self.num_envs).clamp(-1.0, 1.0)
        pos, vel, theta, thdot, phase = self.state.unbind(dim=-1)
        in_p0 = phase < 0.5

        # Phase-0 dynamics: MCC cart, rigid pole.
        vel_p0 = (vel + u * self._CART_POWER
                  - 0.0025 * torch.cos(3.0 * pos)).clamp(-self._MAX_SPEED,
                                                         self._MAX_SPEED)
        pos_p0 = (pos + vel_p0).clamp(self._MIN_POS, self._MAX_POS)
        vel_p0 = torch.where((pos_p0 <= self._MIN_POS) & (vel_p0 < 0),
                             torch.zeros_like(vel_p0), vel_p0)

        # Phase-1 dynamics: cart frozen, pole active (CartPole physics).
        force = u * self._POLE_FORCE_SCALE
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        temp = (force + self._POLEMASS_LENGTH * thdot ** 2 * sin_t) / self._TOTAL_MASS
        thetaacc = (self._G * sin_t - cos_t * temp) / (
            self._LENGTH * (4.0 / 3.0 - self._MASSPOLE * cos_t ** 2 / self._TOTAL_MASS))
        theta_p1 = theta + self._TAU * thdot
        thdot_p1 = thdot + self._TAU * thetaacc

        # Per-env phase blend: phase 0 -> cart moves / pole rigid;
        # phase 1 -> cart frozen / pole active.
        pos_new = torch.where(in_p0, pos_p0, pos)
        vel_new = torch.where(in_p0, vel_p0, torch.zeros_like(vel))
        theta_new = torch.where(in_p0, theta, theta_p1)
        thdot_new = torch.where(in_p0, thdot, thdot_p1)

        self.steps = self.steps + 1.0
        reached_goal = in_p0 & (pos_new >= self._GOAL_POS)

        if self.mode == "nav":
            phase_new = phase                                 # never advances
            reward = 100.0 * reached_goal.float() - 0.1 * u ** 2
            terminated = reached_goal
        else:
            phase_new = torch.where(reached_goal, torch.ones_like(phase), phase)
            in_p1 = phase_new > 0.5
            pole_up = theta_new.abs() <= self._THETA_THRESH
            reward = (in_p1 & pole_up).float() - 0.01 * u ** 2
            terminated = in_p1 & (theta_new.abs() > self._THETA_THRESH)

        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated

        new_state = torch.stack(
            [pos_new, vel_new, theta_new, thdot_new, phase_new], dim=-1)
        self.state = torch.where(done.unsqueeze(-1), self._fresh(), new_state)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, terminated, truncated, done


class DeviceVecPointMass2D:
    """N 2-D point-mass navigation environments, device-resident, batched.

    obs (4) = [x, y, vx, vy]. action (2) = [fx, fy] continuous force.
    A point mass with drag in a bounded square arena; reach a GOAL
    (gx, gy) within a radius. ONE dynamics, infinitely many goals — the
    v4.0 substrate for "reuse the dynamics model across tasks": a single
    world model trained on the (goal-agnostic) dynamics can solve any
    goal by planning, and goals compose naturally for Phase-2 multi-skill
    navigation.

    The observation is GOAL-AGNOSTIC (the goal is NOT in obs) so the
    learned world model is pure dynamics, reusable across all goals; the
    goal enters only through the reward (for SAC) and the planner's
    scoring (for MPC). With ``goal=None`` the goal is resampled each
    episode (for goal-agnostic dynamics-model data); pass a fixed
    ``goal`` (2-vector) to train/evaluate a single specific task.
    """

    obs_dim = 4
    action_dim = 2
    is_discrete = False

    _BOUND = 1.0
    _DRAG = 0.10
    _POWER = 0.05
    _MAX_SPEED = 0.30
    _GOAL_RADIUS = 0.10
    _MAX_STEPS = 100

    # v4.0 Phase 3: distinct DYNAMICS regimes = distinct "notions". obs/
    # action dims and the reach-a-goal reward are identical across regimes;
    # only the transition changes, so a skill learned in one regime is
    # expected NOT to transfer to another (verified by a probe matrix).
    #   free    : the Phase-1/2 dynamics.
    #   drift   : a constant wind added to velocity each step (aim upwind).
    #   ice     : near-frictionless (low drag) -> overshoot, must brake.
    #   reverse : action-to-force map inverted (push the opposite way).
    # Rotation regimes (rot = degrees the action vector is rotated before it
    # becomes force): a clean group of mutually NON-transferring sensorimotor
    # "notions" — a skill tuned for one rotation moves at the wrong angle under
    # another (the inverted-goggles analogy). reverse == rot180.
    #   rot90 / rot270 : action rotated +90 / +270 deg.
    _REGIMES = {
        "free":    dict(drag=0.10, power=0.05, wind=(0.0, 0.0),  rot=0.0),
        "drift":   dict(drag=0.10, power=0.05, wind=(0.04, 0.0), rot=0.0),
        "ice":     dict(drag=0.02, power=0.05, wind=(0.0, 0.0),  rot=0.0),
        "reverse": dict(drag=0.10, power=0.05, wind=(0.0, 0.0),  rot=180.0),
        "rot90":   dict(drag=0.10, power=0.05, wind=(0.0, 0.0),  rot=90.0),
        "rot270":  dict(drag=0.10, power=0.05, wind=(0.0, 0.0),  rot=270.0),
    }

    def __init__(self, num_envs: int, goal=None, regime: str = "free"):
        self.num_envs = num_envs
        if regime not in self._REGIMES:
            raise ValueError(f"unknown regime {regime!r}; "
                             f"choose from {list(self._REGIMES)}")
        self.regime = regime
        p = self._REGIMES[regime]
        self._drag = p["drag"]
        self._power = p["power"]
        self._wind = torch.tensor(p["wind"], dtype=torch.float32, device=DEVICE)
        th = float(p["rot"]) * 3.141592653589793 / 180.0
        # Right-multiply row-vector actions by this matrix == rotating each
        # action vector by `rot` degrees. Identity when rot == 0.
        c, s = math.cos(th), math.sin(th)
        self._rot = torch.tensor([[c, s], [-s, c]], dtype=torch.float32,
                                 device=DEVICE)
        self._has_rot = abs(p["rot"]) > 1e-9
        self._fixed_goal = (None if goal is None
                            else torch.as_tensor(goal, dtype=torch.float32,
                                                 device=DEVICE).reshape(2))
        self.reset()

    def _sample_goals(self, n: int) -> torch.Tensor:
        return (torch.rand(n, 2, device=DEVICE) - 0.5) * 1.6   # [-0.8, 0.8]^2

    def _sample_start(self, n: int) -> torch.Tensor:
        return (torch.rand(n, 2, device=DEVICE) - 0.5) * 1.6

    def _goals(self, n: int) -> torch.Tensor:
        if self._fixed_goal is not None:
            return self._fixed_goal.unsqueeze(0).expand(n, 2).clone()
        return self._sample_goals(n)

    def reset(self) -> torch.Tensor:
        self.pos = self._sample_start(self.num_envs)
        self.vel = torch.zeros(self.num_envs, 2, device=DEVICE)
        self.goal = self._goals(self.num_envs)
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        self.state = torch.cat([self.pos, self.vel], dim=-1)
        return self.state

    def step(self, action: torch.Tensor):
        """action: (N, 2) or (N*2,) float, clamped to [-1, 1]."""
        f = action.reshape(self.num_envs, 2).clamp(-1.0, 1.0)
        if self._has_rot:
            f = f @ self._rot
        self.vel = self.vel * (1.0 - self._drag) + f * self._power + self._wind
        self.vel = self.vel.clamp(-self._MAX_SPEED, self._MAX_SPEED)
        self.pos = (self.pos + self.vel).clamp(-self._BOUND, self._BOUND)
        self.steps = self.steps + 1.0

        dist = torch.norm(self.pos - self.goal, dim=-1)
        reached = dist < self._GOAL_RADIUS
        # Dense shaping (-dist) + control cost + reach bonus. MPC scores
        # by the same -dist; SAC optimizes the same reward.
        reward = -dist - 0.01 * (f ** 2).sum(dim=-1) + 10.0 * reached.float()

        terminated = reached
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated

        fresh_pos = self._sample_start(self.num_envs)
        fresh_vel = torch.zeros(self.num_envs, 2, device=DEVICE)
        fresh_goal = self._goals(self.num_envs)
        m = done.unsqueeze(-1)
        self.pos = torch.where(m, fresh_pos, self.pos)
        self.vel = torch.where(m, fresh_vel, self.vel)
        self.goal = torch.where(m, fresh_goal, self.goal)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        self.state = torch.cat([self.pos, self.vel], dim=-1)
        return self.state, reward, terminated, truncated, done


class DeviceVecOrderedVisit:
    """N point-mass environments where the task is to VISIT 3 zones in a
    fixed order — the v4.0 Phase-2 compositional task.

    obs (5) = [x, y, vx, vy, progress] where progress in {0,1,2,3}/3 is
    how many zones have been visited in order so far. action (2) = force.
    Same point-mass DYNAMICS as DeviceVecPointMass2D — so a world model
    trained on the free-space dynamics (Phase 1) is reusable UNCHANGED
    for the first 4 obs dims; "progress" is task bookkeeping the
    high-level tracks, not part of the dynamics model.

    Reward is SPARSE: +1 when the next-required zone is reached (advancing
    progress), +10 on completing all three, small control cost. Reaching
    a wrong (not-next) zone does nothing. Sparse + long-horizon, so flat
    primitive RL faces a hard exploration problem; a hierarchical agent
    that REUSES a "reach a point" skill (Phase-1 world-model planning)
    only has to learn the ORDER — the composition — and should learn it
    far faster (or solve a task flat RL cannot).

    The "basic notion" (reach a point) is reused; the "complex notion"
    (the ordered visit) is the new composition.
    """

    obs_dim = 5
    action_dim = 2
    is_discrete = False

    _BOUND = 1.0
    _DRAG = 0.10
    _POWER = 0.05
    _MAX_SPEED = 0.30
    _ZONE_RADIUS = 0.15
    _MAX_STEPS = 150
    # Three fixed zones; required order is index 0 -> 1 -> 2.
    _ZONES = ((-0.6, 0.6), (0.6, 0.6), (0.0, -0.6))
    _N_ZONES = 3

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        zr = torch.tensor(self._ZONES, dtype=torch.float32, device=DEVICE)
        self.zones = zr                                    # (3, 2)
        self.reset()

    def _obs(self) -> torch.Tensor:
        prog = (self.progress.float() / self._N_ZONES).unsqueeze(-1)
        return torch.cat([self.pos, self.vel, prog], dim=-1)

    def reset(self) -> torch.Tensor:
        self.pos = (torch.rand(self.num_envs, 2, device=DEVICE) - 0.5) * 1.6
        self.vel = torch.zeros(self.num_envs, 2, device=DEVICE)
        self.progress = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        self.state = self._obs()
        return self.state

    def next_zone(self) -> torch.Tensor:
        """(N, 2) the position of each env's next-required zone (clamped at
        the last zone once complete) — used by a high-level / oracle."""
        idx = self.progress.clamp(max=self._N_ZONES - 1)
        return self.zones[idx]

    def step(self, action: torch.Tensor):
        f = action.reshape(self.num_envs, 2).clamp(-1.0, 1.0)
        self.vel = (self.vel * (1.0 - self._DRAG) + f * self._POWER).clamp(
            -self._MAX_SPEED, self._MAX_SPEED)
        self.pos = (self.pos + self.vel).clamp(-self._BOUND, self._BOUND)
        self.steps = self.steps + 1.0

        # Did we reach the next-required zone? (only the next one counts)
        nz = self.next_zone()
        at_next = (torch.norm(self.pos - nz, dim=-1) < self._ZONE_RADIUS)
        incomplete = self.progress < self._N_ZONES
        advanced = at_next & incomplete
        self.progress = self.progress + advanced.long()

        completed = self.progress >= self._N_ZONES
        reward = (advanced.float()                       # +1 per correct zone
                  + 10.0 * (completed & advanced).float()  # +10 finishing bonus
                  - 0.01 * (f ** 2).sum(dim=-1))

        terminated = completed
        truncated = self.steps >= self._MAX_STEPS
        done = terminated | truncated

        fresh_pos = (torch.rand(self.num_envs, 2, device=DEVICE) - 0.5) * 1.6
        m = done.unsqueeze(-1)
        self.pos = torch.where(m, fresh_pos, self.pos)
        self.vel = torch.where(m, torch.zeros_like(self.vel), self.vel)
        self.progress = torch.where(done, torch.zeros_like(self.progress),
                                    self.progress)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        self.state = self._obs()
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
