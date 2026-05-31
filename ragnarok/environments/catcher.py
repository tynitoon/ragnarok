"""DeviceVecCatcher — a falling-object catching game (dense reward, reliably
learnable), structurally different from the bounce games (Pong/Breakout).

A paddle at the bottom moves left/right to catch fruit that falls from the top
(constant fall, no bounce). Catch = +1, miss = -1; a new fruit respawns. Endless
score-maximisation (like Snake), dense reward (a fruit resolves every fall) so
there is NO hard-exploration trap. Adds a distinct dynamic — intercept a falling
object — to the library.

Device-resident batched env: `.state` (N, 3*H*W) flattened RGB, `.step(action)`
discrete (0 stay, 1 left, 2 right).
"""

import torch

from ragnarok.infrastructure.device import DEVICE


class DeviceVecCatcher:
    A_STAY, A_LEFT, A_RIGHT = 0, 1, 2

    def __init__(self, num_envs, img=48, max_steps=400, paddle_half=0.12,
                 paddle_speed=0.040, fall_speed=0.030, catch_pad=0.02,
                 shaping=0.05, seed=0):
        self.num_envs = num_envs
        self.H = self.W = img
        self.img_hw = img
        self.obs_dim = 3 * img * img
        self.action_dim = 3
        self.max_steps = max_steps
        self.PH, self.PS, self.FS = paddle_half, paddle_speed, fall_speed
        self.catch_pad, self.shaping = catch_pad, shaping
        self.y_pad = 0.88
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        z = torch.zeros(num_envs, device=DEVICE)
        self.px = z + 0.5
        self.fx = self._rand_x()
        self.fy = z.clone()
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        self.cum_catch = z.clone()
        self._img = torch.zeros(num_envs, 3, img, img, device=DEVICE)
        self._render()

    def _rand_x(self):
        return torch.rand(self.num_envs, generator=self._gen, device=DEVICE) * 0.9 + 0.05

    @property
    def state(self):
        return self._img.reshape(self.num_envs, -1)

    def _render(self):
        self._img.zero_()
        N, H, W = self.num_envs, self.H, self.W
        ri = torch.arange(H, device=DEVICE).view(1, H, 1)
        ci = torch.arange(W, device=DEVICE).view(1, 1, W).float()
        # paddle: white horizontal bar at y_pad spanning [px-PH, px+PH]
        prow = int(self.y_pad * (H - 1))
        pcx = (self.px * (W - 1)).view(N, 1, 1)
        pw = self.PH * (W - 1)
        bar = ((ri - prow).abs() <= 1) & ((ci - pcx).abs() <= pw)          # (N,H,W)
        bf = bar.float()
        for c in range(3):
            self._img[:, c] = torch.maximum(self._img[:, c], bf)
        # fruit: red dot at (fx, fy)
        fr = (self.fy.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
        fc = (self.fx * (W - 1)).view(N, 1, 1)
        rad = max(1, int(0.03 * H))
        fruit = ((ri - fr).abs() <= rad) & ((ci - fc).abs() <= rad)
        self._img[:, 0] = torch.maximum(self._img[:, 0], fruit.float())    # red

    def reset(self):
        return self.state

    def step(self, action):
        N = self.num_envs
        move = (action == self.A_RIGHT).float() - (action == self.A_LEFT).float()
        self.px = (self.px + move * self.PS).clamp(self.PH, 1 - self.PH)
        self.fy = self.fy + self.FS
        landed = self.fy >= self.y_pad
        caught = landed & ((self.fx - self.px).abs() <= (self.PH + self.catch_pad))
        missed = landed & ~caught
        # dense shaping toward the fruit x (keeps it reliably learnable)
        shaped = self.shaping * (1.0 - (self.px - self.fx).abs() * 2.0).clamp(min=0.0)
        reward = caught.float() - missed.float() + shaped
        self.cum_catch = self.cum_catch + caught.float()
        # respawn fruit after it lands
        new_x = self._rand_x()
        self.fx = torch.where(landed, new_x, self.fx)
        self.fy = torch.where(landed, torch.zeros_like(self.fy), self.fy)

        self.steps = self.steps + 1
        truncated = self.steps >= self.max_steps
        terminated = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        done = truncated
        if bool(done.any()):
            self._reset_done(done)
        self._render()
        return self.state, reward, terminated, truncated, done

    def _reset_done(self, done):
        z = torch.zeros(self.num_envs, device=DEVICE)
        self.px = torch.where(done, z + 0.5, self.px)
        self.fx = torch.where(done, self._rand_x(), self.fx)
        self.fy = torch.where(done, z, self.fy)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)

    def metrics(self):
        return dict(mean_cum_catch=float(self.cum_catch.mean()))
