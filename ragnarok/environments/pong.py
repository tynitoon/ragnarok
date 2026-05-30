"""DeviceVecPong — a GPU-batched, device-resident Pong game with PIXEL obs.

Same device-resident paradigm as DeviceVecCraftWorld: N games stepped in
parallel on the accelerator, `.state` is a flattened RGB image, `.step(action)`
takes a discrete action per env. The agent (left paddle, GREEN) plays a
SCRIPTED tracking opponent (right paddle, RED); the ball is WHITE. Reward is
+1 when the agent scores, -1 when it concedes, with a small bonus for
returning the ball. "Winning" = positive score margin over an episode.

This is the first step of the general-game-mastery north star: drop the same
agent on a game, from pixels, and it learns to WIN.

Coordinates are continuous in [0,1]^2; scoring is decided AT the paddle plane
(return if aligned, concede if not) — no tunnelling. Actions: 0 stay, 1 up,
2 down.
"""

import torch

from ragnarok.infrastructure.device import DEVICE

A_STAY, A_UP, A_DOWN = 0, 1, 2


class DeviceVecPong:
    def __init__(self, num_envs, img=48, max_steps=800, paddle_half=0.12,
                 paddle_speed=0.040, ball_speed=0.030, opp_speed=0.020,
                 spin=0.5, contact_bonus=0.1, seed=0):
        self.num_envs = num_envs
        self.H = self.W = img
        self.img_hw = img
        self.obs_dim = 3 * img * img
        self.action_dim = 3
        self.max_steps = max_steps
        self.PH = paddle_half
        self.PS = paddle_speed
        self.BS = ball_speed
        self.OS = opp_speed
        self.SPIN = spin
        self.CONTACT = contact_bonus
        self.x_a, self.x_o = 0.06, 0.94          # paddle planes (left, right)
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        # precomputed pixel coordinate grids for rendering
        self._ys = torch.arange(self.H, device=DEVICE).view(1, self.H, 1).float()
        self._xs = torch.arange(self.W, device=DEVICE).view(1, 1, self.W).float()
        self.reset()

    # ---- serving / reset ------------------------------------------------
    def _serve(self, mask):
        """(Re)serve the ball at centre for envs in mask, random direction."""
        n = self.num_envs
        ang = (torch.rand(n, generator=self._gen, device=DEVICE) - 0.5) * 1.4  # vy/vx tilt
        sgn = torch.where(torch.rand(n, generator=self._gen, device=DEVICE) < 0.5,
                          -1.0, 1.0)
        vx = sgn * self.BS / torch.sqrt(1 + ang ** 2)
        vy = ang * vx.abs()
        self.bx = torch.where(mask, torch.full_like(self.bx, 0.5), self.bx)
        self.by = torch.where(mask, torch.full_like(self.by, 0.5), self.by)
        self.vx = torch.where(mask, vx, self.vx)
        self.vy = torch.where(mask, vy, self.vy)

    def reset(self):
        n = self.num_envs
        z = torch.zeros(n, device=DEVICE)
        self.bx, self.by = z.clone() + 0.5, z.clone() + 0.5
        self.vx, self.vy = z.clone(), z.clone()
        self.pad_a, self.pad_o = z.clone() + 0.5, z.clone() + 0.5
        self.score_a, self.score_o = z.clone(), z.clone()
        self.steps = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self._serve(torch.ones(n, dtype=torch.bool, device=DEVICE))
        self._render()
        return self.state

    # ---- rendering ------------------------------------------------------
    def _render(self):
        N, H, W = self.num_envs, self.H, self.W
        pw = 1.5                                   # paddle half-width (px)
        php = self.PH * H                          # paddle half-height (px)
        br = max(1.0, 0.02 * H)                    # ball half-size (px)
        xa, xo = self.x_a * W, self.x_o * W
        col = (self._xs - xa).abs() <= pw          # (1,1,W)
        row_a = (self._ys - (self.pad_a.view(N, 1, 1) * H)).abs() <= php
        mask_a = col & row_a                       # (N,H,W)
        col_o = (self._xs - xo).abs() <= pw
        row_o = (self._ys - (self.pad_o.view(N, 1, 1) * H)).abs() <= php
        mask_o = col_o & row_o
        mask_b = ((self._xs - (self.bx.view(N, 1, 1) * W)).abs() <= br) & \
                 ((self._ys - (self.by.view(N, 1, 1) * H)).abs() <= br)
        img = torch.zeros(N, 3, H, W, device=DEVICE)
        img[:, 0] = torch.maximum(mask_o.float(), mask_b.float())   # R: opp + ball
        img[:, 1] = torch.maximum(mask_a.float(), mask_b.float())   # G: agent + ball
        img[:, 2] = mask_b.float()                                  # B: ball -> white
        self._img = img.reshape(N, -1)

    @property
    def state(self):
        return self._img

    # ---- dynamics -------------------------------------------------------
    def step(self, action):
        a = action.reshape(self.num_envs).long()
        N = self.num_envs
        reward = torch.zeros(N, device=DEVICE)

        # agent paddle (action) + opponent paddle (scripted tracking)
        self.pad_a = (self.pad_a + (a == A_UP).float() * self.PS
                      - (a == A_DOWN).float() * self.PS).clamp(self.PH, 1 - self.PH)
        # scripted opponent: track the ball only while it approaches (vx>0),
        # else drift back to centre — a classic beatable Pong AI (well-angled
        # returns beat it because it starts from centre).
        target_o = torch.where(self.vx > 0, self.by, torch.full_like(self.by, 0.5))
        self.pad_o = (self.pad_o + (target_o - self.pad_o).clamp(-self.OS, self.OS)
                      ).clamp(self.PH, 1 - self.PH)

        # move ball
        self.bx = self.bx + self.vx
        self.by = self.by + self.vy
        # top/bottom bounce (reflect)
        lo = self.by < 0; hi = self.by > 1
        self.by = torch.where(lo, -self.by, self.by)
        self.by = torch.where(hi, 2 - self.by, self.by)
        self.vy = torch.where(lo | hi, -self.vy, self.vy)

        # agent plane (left): return if aligned, else concede
        cross_a = (self.bx <= self.x_a) & (self.vx < 0)
        hit_a = cross_a & ((self.by - self.pad_a).abs() <= self.PH)
        miss_a = cross_a & ~hit_a
        self.vx = torch.where(hit_a, self.vx.abs(), self.vx)             # bounce right
        self.vy = torch.where(hit_a, self.vy + (self.by - self.pad_a) * self.SPIN, self.vy)
        self.bx = torch.where(hit_a, 2 * self.x_a - self.bx, self.bx)
        reward += hit_a.float() * self.CONTACT
        reward -= miss_a.float()
        self.score_o += miss_a.float()

        # opponent plane (right): return if aligned, else AGENT scores
        cross_o = (self.bx >= self.x_o) & (self.vx > 0)
        hit_o = cross_o & ((self.by - self.pad_o).abs() <= self.PH)
        miss_o = cross_o & ~hit_o
        self.vx = torch.where(hit_o, -self.vx.abs(), self.vx)            # bounce left
        self.vy = torch.where(hit_o, self.vy + (self.by - self.pad_o) * self.SPIN, self.vy)
        self.bx = torch.where(hit_o, 2 * self.x_o - self.bx, self.bx)
        reward += miss_o.float()
        self.score_a += miss_o.float()

        self._serve(miss_a | miss_o)               # re-serve after a point
        # keep ball speed bounded (spin can otherwise grow vy unboundedly)
        sp = torch.sqrt(self.vx ** 2 + self.vy ** 2).clamp(min=1e-6)
        cap = sp > self.BS * 1.6
        self.vx = torch.where(cap, self.vx * (self.BS * 1.6 / sp), self.vx)
        self.vy = torch.where(cap, self.vy * (self.BS * 1.6 / sp), self.vy)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        done = truncated
        if bool(done.any()):
            self._reset_done(done)
        self._render()
        return self.state, reward, terminated, truncated, done

    def _reset_done(self, done):
        z = torch.zeros(self.num_envs, device=DEVICE)
        self.pad_a = torch.where(done, z + 0.5, self.pad_a)
        self.pad_o = torch.where(done, z + 0.5, self.pad_o)
        self.score_a = torch.where(done, z, self.score_a)
        self.score_o = torch.where(done, z, self.score_o)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        self._serve(done)
