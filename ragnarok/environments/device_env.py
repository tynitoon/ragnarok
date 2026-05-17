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
