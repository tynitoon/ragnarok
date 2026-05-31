"""DeviceVecProjectileCatch — a minimal task that USES the concept of gravity.

A projectile launches from the left and arcs to the right under gravity (with
top/bottom wall bounces). A catcher on the right plane must be at the ball's y
when it arrives. The optimal policy is NOT to track the ball's current y (it arcs)
but to be at its future LANDING y — which requires the concept of gravity.

This env is the substrate for the concept-transfer demo (v36): an agent that has
already learned gravity (a landing predictor) can be handed the landing as a
feature and solves this in very few trials; an agent from scratch must re-infer
gravity from the raw state and is much slower.

`concept` (optional callable: ball-state (N,4) -> predicted landing (N,)) selects
the observation: with a concept, obs = [cy, landing_pred] (dim 2); without, obs =
[cy, bx, by, bvx, bvy] (dim 5). State-based (no pixels) to isolate the concept.
"""

import torch

from ragnarok.infrastructure.device import DEVICE


class DeviceVecProjectileCatch:
    A_STAY, A_UP, A_DOWN = 0, 1, 2

    def __init__(self, num_envs, concept=None, gravity=0.004, catcher_speed=0.035,
                 tol=0.08, x_plane=0.97, max_steps=70, img=0, seed=0):
        self.num_envs = num_envs
        self.concept = concept
        self.img = img
        self.g, self.cs, self.tol = gravity, catcher_speed, tol
        self.x_plane, self.max_steps = x_plane, max_steps
        self.action_dim = 3
        if img > 0:
            self.img_hw = img
            self.obs_dim = 2 if concept is not None else 3 * img * img
        else:
            self.obs_dim = 2 if concept is not None else 5
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        self._launch(torch.ones(num_envs, dtype=torch.bool, device=DEVICE))
        self.cy = torch.full((num_envs,), 0.5, device=DEVICE)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        self.cum_catch = torch.zeros(num_envs, device=DEVICE)
        self.cum_ep = torch.zeros(num_envs, device=DEVICE)

    def _u(self, lo, hi):
        return torch.rand(self.num_envs, generator=self._gen, device=DEVICE) * (hi - lo) + lo

    def _launch(self, mask):
        if not hasattr(self, "bx"):
            z = torch.zeros(self.num_envs, device=DEVICE)
            self.bx, self.by, self.bvx, self.bvy = z.clone(), z.clone(), z.clone(), z.clone()
        self.bx = torch.where(mask, torch.zeros_like(self.bx), self.bx)
        self.by = torch.where(mask, self._u(0.1, 0.6), self.by)
        self.bvx = torch.where(mask, self._u(0.018, 0.030), self.bvx)
        self.bvy = torch.where(mask, self._u(-0.02, 0.06), self.bvy)

    @torch.no_grad()
    def _landing(self):
        """Analytic landing y at x_plane incl. wall bounces, from the CURRENT state."""
        bx, by, bvx, bvy = self.bx, self.by, self.bvx, self.bvy
        t = ((self.x_plane - bx) / bvx.clamp(min=1e-5)).clamp(min=0)
        # y under gravity: y + vy*t - 0.5*g*t^2, then fold into [0,1] (reflect)
        yraw = by + bvy * t - 0.5 * self.g * t * t
        m = torch.remainder(yraw, 2.0)
        return torch.where(m <= 1.0, m, 2.0 - m)

    @torch.no_grad()
    def _pixels(self):
        N, H = self.num_envs, self.img
        im = torch.zeros(N, 3, H, H, device=DEVICE)
        ri = torch.arange(H, device=DEVICE).view(1, H, 1)
        ci = torch.arange(H, device=DEVICE).view(1, 1, H)

        def dot(px, py, chan, val, rad=1):
            r = (py.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
            c = (px.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
            m = ((ri - r).abs() <= rad) & ((ci - c).abs() <= rad)
            im[:, chan] = torch.maximum(im[:, chan], m.float() * val)
        dot(self.bx - self.bvx, self.by - self.bvy, 2, 0.5)   # velocity cue (prev pos, dim blue)
        dot(self.bx, self.by, 0, 1.0)                          # ball (red)
        cr = (self.cy.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
        cc = int(self.x_plane * (H - 1))
        bar = ((ri - cr).abs() <= 2) & ((ci - cc).abs() <= 0)
        im[:, 1] = torch.maximum(im[:, 1], bar.float())        # catcher (green bar)
        return im.reshape(N, -1)

    @property
    def state(self):
        if self.img > 0:
            pix = self._pixels()
            if self.concept is not None:
                return torch.stack([self.cy, self.concept(pix).reshape(-1)], -1)
            return pix
        if self.concept is not None:
            ball = torch.stack([self.bx, self.by, self.bvx, self.bvy], -1)
            return torch.stack([self.cy, self.concept(ball).reshape(-1)], -1)
        return torch.stack([self.cy, self.bx, self.by, self.bvx, self.bvy], -1)

    def reset(self):
        return self.state

    def step(self, action):
        move = (action == self.A_UP).float() - (action == self.A_DOWN).float()
        self.cy = (self.cy + move * self.cs).clamp(0.0, 1.0)
        # advance ball physics
        self.bvy = self.bvy - self.g
        self.bx = self.bx + self.bvx
        self.by = self.by + self.bvy
        lo = self.by < 0; hi = self.by > 1
        self.by = torch.where(lo, -self.by, torch.where(hi, 2.0 - self.by, self.by))
        self.bvy = torch.where(lo | hi, -self.bvy, self.bvy)

        self.steps = self.steps + 1
        arrived = self.bx >= self.x_plane
        timeout = self.steps >= self.max_steps
        resolve = arrived | timeout
        err = (self.cy - self.by).abs()
        caught = resolve & arrived & (err <= self.tol)
        # graded terminal reward: catch bonus minus distance-to-ball at the plane
        reward = torch.where(resolve, caught.float() * 1.0 - err.clamp(max=1.0) * 0.5,
                             torch.zeros_like(self.cy))
        self.cum_catch = self.cum_catch + caught.float()
        self.cum_ep = self.cum_ep + resolve.float()

        terminated = resolve
        truncated = torch.zeros_like(resolve)
        done = resolve
        if bool(done.any()):
            self._reset_done(done)
        return self.state, reward, terminated, truncated, done

    def _reset_done(self, done):
        self._launch(done)
        self.cy = torch.where(done, torch.full_like(self.cy, 0.5), self.cy)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)

    def catch_rate(self):
        return float(self.cum_catch.sum() / self.cum_ep.sum().clamp(min=1))
