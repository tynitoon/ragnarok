"""DeviceVecBreakout — a GPU-batched, device-resident Breakout with PIXEL obs.

Same paradigm as DeviceVecPong/CraftWorld: N games on the accelerator, `.state`
is a flattened RGB image, `.step(action)` a discrete action per env. The agent
moves a paddle (GREEN) at the bottom; a WHITE ball bounces and breaks a wall of
coloured BRICKS at the top (+1 each). Missing the ball costs a life (-1); lose
all lives -> game over (LOSE); clear the wall -> WIN (+bonus). Games auto-reset.

This is rung P1 of the game curriculum: a clear win (cleared wall) and lose
(out of lives) outcome, learned from pixels by the SAME agent as Pong.
Actions: 0 stay, 1 left, 2 right.
"""

import torch

from ragnarok.infrastructure.device import DEVICE

A_STAY, A_LEFT, A_RIGHT = 0, 1, 2


class DeviceVecBreakout:
    def __init__(self, num_envs, img=48, max_steps=1000, rows=4, cols=8,
                 paddle_half=0.11, paddle_speed=0.045, ball_speed=0.028,
                 spin=0.35, lives=3, clear_bonus=5.0, seed=0):
        self.num_envs = num_envs
        self.H = self.W = img
        self.img_hw = img
        self.obs_dim = 3 * img * img
        self.action_dim = 3
        self.max_steps = max_steps
        self.BR, self.BC = rows, cols
        self.PW = paddle_half
        self.PS = paddle_speed
        self.BS = ball_speed
        self.SPIN = spin
        self.LIVES0 = lives
        self.CLEAR = clear_bonus
        self.y_p = 0.90                       # paddle plane
        self.by0, self.rh = 0.10, 0.06        # brick band top + row height
        self.cw = 1.0 / cols                  # brick col width
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        self._ys = torch.arange(self.H, device=DEVICE).view(1, self.H, 1).float()
        self._xs = torch.arange(self.W, device=DEVICE).view(1, 1, self.W).float()
        # brick colours (BR,BC,3) for rendering
        hue = torch.linspace(0.2, 0.9, rows, device=DEVICE).view(rows, 1, 1)
        self._brick_rgb = torch.cat([hue.expand(rows, cols, 1),
                                     (1 - hue).expand(rows, cols, 1),
                                     torch.full((rows, cols, 1), 0.6, device=DEVICE)], -1)
        # cumulative eval counters (never auto-reset)
        self.cum_bricks = torch.zeros(num_envs, device=DEVICE)
        self.cum_wins = torch.zeros(num_envs, device=DEVICE)
        self.cum_losses = torch.zeros(num_envs, device=DEVICE)
        self.reset()

    def _serve(self, mask):
        n = self.num_envs
        vx = (torch.rand(n, generator=self._gen, device=DEVICE) - 0.5) * 2 * self.BS * 0.7
        self.bx = torch.where(mask, torch.full_like(self.bx, 0.5), self.bx)
        self.by = torch.where(mask, torch.full_like(self.by, 0.6), self.by)
        self.vx = torch.where(mask, vx, self.vx)
        self.vy = torch.where(mask, torch.full_like(self.vy, -self.BS), self.vy)  # upward

    def _new_game(self, mask):
        """Refill bricks, reset lives + paddle + serve, for envs in mask."""
        n = self.num_envs
        full = torch.ones(n, self.BR, self.BC, dtype=torch.bool, device=DEVICE)
        self.bricks = torch.where(mask.view(n, 1, 1), full, self.bricks)
        self.lives = torch.where(mask, torch.full_like(self.lives, self.LIVES0), self.lives)
        self.pad_x = torch.where(mask, torch.full_like(self.pad_x, 0.5), self.pad_x)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self._serve(mask)

    def reset(self):
        n = self.num_envs
        z = torch.zeros(n, device=DEVICE)
        self.bx, self.by = z + 0.5, z + 0.6
        self.vx, self.vy = z.clone(), z.clone()
        self.pad_x = z + 0.5
        self.bricks = torch.zeros(n, self.BR, self.BC, dtype=torch.bool, device=DEVICE)
        self.lives = torch.full((n,), self.LIVES0, dtype=torch.long, device=DEVICE)
        self.steps = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self._new_game(torch.ones(n, dtype=torch.bool, device=DEVICE))
        self._render()
        return self.state

    # ---- rendering ------------------------------------------------------
    def _render(self):
        N, H, W = self.num_envs, self.H, self.W
        img = torch.zeros(N, 3, H, W, device=DEVICE)
        # bricks: paint each present brick cell's pixel block
        # pixel -> brick row/col
        by_idx = ((self._ys / H - self.by0) / self.rh).floor().long()   # (1,H,1)
        bx_idx = (self._xs / W / self.cw).floor().long()                # (1,1,W)
        in_band = (by_idx >= 0) & (by_idx < self.BR) & (bx_idx >= 0) & (bx_idx < self.BC)
        r = by_idx.clamp(0, self.BR - 1).expand(N, H, W)
        c = bx_idx.clamp(0, self.BC - 1).expand(N, H, W)
        n_idx = torch.arange(N, device=DEVICE).view(N, 1, 1).expand(N, H, W)
        present = self.bricks[n_idx, r, c] & in_band                    # (N,H,W)
        brgb = self._brick_rgb[r, c]                                    # (N,H,W,3)
        for ch in range(3):
            img[:, ch] = torch.where(present, brgb[..., ch], img[:, ch])
        # paddle (green) at bottom
        php = 0.02 * H
        col_p = (self._ys - self.y_p * H).abs() <= php
        row_p = (self._xs - self.pad_x.view(N, 1, 1) * W).abs() <= self.PW * W
        mask_p = col_p & row_p
        img[:, 1] = torch.maximum(img[:, 1], mask_p.float())
        # ball (white)
        br = max(1.0, 0.02 * H)
        mask_b = ((self._xs - self.bx.view(N, 1, 1) * W).abs() <= br) & \
                 ((self._ys - self.by.view(N, 1, 1) * H).abs() <= br)
        for ch in range(3):
            img[:, ch] = torch.maximum(img[:, ch], mask_b.float())
        self._img = img.reshape(N, -1)

    @property
    def state(self):
        return self._img

    # ---- dynamics -------------------------------------------------------
    def step(self, action):
        a = action.reshape(self.num_envs).long()
        N = self.num_envs
        reward = torch.zeros(N, device=DEVICE)

        self.pad_x = (self.pad_x + (a == A_RIGHT).float() * self.PS
                      - (a == A_LEFT).float() * self.PS).clamp(self.PW, 1 - self.PW)
        self.bx = self.bx + self.vx
        self.by = self.by + self.vy
        # walls: left/right reflect vx, top reflect vy
        lo = self.bx < 0; hi = self.bx > 1
        self.bx = torch.where(lo, -self.bx, torch.where(hi, 2 - self.bx, self.bx))
        self.vx = torch.where(lo | hi, -self.vx, self.vx)
        top = self.by < 0
        self.by = torch.where(top, -self.by, self.by)
        self.vy = torch.where(top, -self.vy, self.vy)

        # brick collision (ball in band & brick present)
        r = ((self.by - self.by0) / self.rh).floor().long()
        c = (self.bx / self.cw).floor().long()
        in_band = (r >= 0) & (r < self.BR) & (c >= 0) & (c < self.BC)
        rcl, ccl = r.clamp(0, self.BR - 1), c.clamp(0, self.BC - 1)
        idx = torch.arange(N, device=DEVICE)
        hit_brick = in_band & self.bricks[idx, rcl, ccl]
        if bool(hit_brick.any()):
            self.bricks[idx[hit_brick], rcl[hit_brick], ccl[hit_brick]] = False
        reward += hit_brick.float()
        self.cum_bricks += hit_brick.float()
        self.vy = torch.where(hit_brick, -self.vy, self.vy)

        # paddle plane: catch if aligned, else lose a life
        cross = (self.by >= self.y_p) & (self.vy > 0)
        hit_p = cross & ((self.bx - self.pad_x).abs() <= self.PW)
        miss = cross & ~hit_p
        self.vy = torch.where(hit_p, -self.vy.abs(), self.vy)
        self.vx = torch.where(hit_p, self.vx + (self.bx - self.pad_x) * self.SPIN, self.vx)
        self.by = torch.where(hit_p, 2 * self.y_p - self.by, self.by)
        self.lives = self.lives - miss.long()
        reward -= miss.float()
        # speed cap (spin growth)
        sp = torch.sqrt(self.vx ** 2 + self.vy ** 2).clamp(min=1e-6)
        cap = sp > self.BS * 1.7
        self.vx = torch.where(cap, self.vx * (self.BS * 1.7 / sp), self.vx)
        self.vy = torch.where(cap, self.vy * (self.BS * 1.7 / sp), self.vy)

        cleared = self.bricks.flatten(1).sum(-1) == 0          # WIN
        dead = self.lives <= 0                                  # LOSE
        reward += cleared.float() * self.CLEAR
        reward -= dead.float() * self.CLEAR
        self.cum_wins += cleared.float()
        self.cum_losses += dead.float()

        # re-serve after a (non-terminal) miss; new game after win/lose/timeout
        reserve = miss & ~dead
        self._serve(reserve)
        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = cleared | dead
        done = terminated | truncated
        if bool(done.any()):
            self._new_game(done)
        self._render()
        return self.state, reward, terminated, truncated, done

    def stats(self):
        return dict(cum_bricks=float(self.cum_bricks.mean()),
                    wins=float(self.cum_wins.sum()), losses=float(self.cum_losses.sum()))
