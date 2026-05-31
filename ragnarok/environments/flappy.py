"""DeviceVecFlappy — a gravity+timing game, structurally DIFFERENT from the
paddle-ball (Pong/Breakout) and grid (Snake) games.

The bird falls under constant gravity; a single discrete action (FLAP) sets an
upward impulse. A pipe with a gap scrolls in from the right; the bird must be
inside the gap when the pipe reaches it, else it dies (also dies on ceiling/
ground). Reward: +1 per pipe passed, small survival bonus, -1 on death. This adds
a genuinely new skill to the library — anticipatory timing under gravity — and a
new visual structure (vertical pipe + gap), giving the cross-game tests a target
that is NOT visually similar to any existing game.

Device-resident batched env, mirroring the rest of the codebase: `.state` is a
flattened RGB image (N, 3*H*W), `.step(action:(N,))` a discrete action per env.
"""

import torch

from ragnarok.infrastructure.device import DEVICE


class DeviceVecFlappy:
    A_NOOP, A_FLAP = 0, 1

    def __init__(self, num_envs, img=48, max_steps=600, gravity=0.0016,
                 flap=0.026, scroll=0.016, gap_half=0.17, bird_x=0.30,
                 bird_r=0.045, survive_bonus=0.01, shaping=0.0, seed=0):
        self.num_envs = num_envs
        self.H = self.W = img
        self.img_hw = img
        self.obs_dim = 3 * img * img
        self.action_dim = 2
        self.max_steps = max_steps
        self.G, self.FLAP, self.SCROLL = gravity, flap, scroll
        self.GH, self.bird_x, self.bird_r = gap_half, bird_x, bird_r
        self.survive = survive_bonus
        self.shaping = shaping
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        self._ys = (torch.arange(img, device=DEVICE).float() + 0.5) / img   # row centres
        z = torch.zeros(num_envs, device=DEVICE)
        self.by = z + 0.5
        self.by_prev = z + 0.5        # previous bird y -> rendered as a trail (velocity cue)
        self.vy = z.clone()
        self.pipe_x = z + 1.0
        self.gap_y = self._rand_gap()
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        self.score = z.clone()
        self.cum_score = z.clone()
        self._img = torch.zeros(num_envs, 3, img, img, device=DEVICE)
        self._render()

    def _rand_gap(self):
        return (torch.rand(self.num_envs, generator=self._gen, device=DEVICE)
                * (1 - 2 * self.GH) + self.GH)

    @property
    def state(self):
        return self._img.reshape(self.num_envs, -1)

    def _render(self):
        self._img.zero_()
        N, H, W = self.num_envs, self.H, self.W
        cols = torch.arange(W, device=DEVICE).float().view(1, 1, W)
        rows_y = self._ys.view(1, H, 1)
        # pipe: green band ~2px wide at pipe_x, all rows except the gap
        pipe_col = (self.pipe_x.clamp(0, 1) * (W - 1)).view(N, 1, 1)
        col_mask = (cols - pipe_col).abs() <= 1.5                        # (N,1,W)
        visible = ((self.pipe_x >= 0) & (self.pipe_x <= 1)).view(N, 1, 1)
        gap_mask = (rows_y - self.gap_y.view(N, 1, 1)).abs() <= self.GH  # (N,H,1)
        pipe = col_mask & (~gap_mask) & visible                          # (N,H,W)
        self._img[:, 1] = pipe.float()
        # bird block geometry
        rad = max(1, int(self.bird_r * H))
        bc = int(self.bird_x * (W - 1))
        ri = torch.arange(H, device=DEVICE).view(1, H, 1)
        ci = torch.arange(W, device=DEVICE).view(1, 1, W)
        # trail: PREVIOUS bird position in BLUE -> a single frame now encodes velocity
        tr = (self.by_prev.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
        trail = ((ri - tr).abs() <= rad) & ((ci - bc).abs() <= rad)
        self._img[:, 2] = torch.maximum(self._img[:, 2], trail.float())
        # bird: yellow block at (bird_x, by), drawn on top
        br = (self.by.clamp(0, 1) * (H - 1)).long().view(N, 1, 1)
        bird = ((ri - br).abs() <= rad) & ((ci - bc).abs() <= rad)       # (N,H,W)
        bf = bird.float()
        self._img[:, 0] = torch.maximum(self._img[:, 0], bf)
        self._img[:, 1] = torch.maximum(self._img[:, 1], bf)
        self._img[:, 2] = self._img[:, 2] * (~bird)      # bird overwrites trail (stays yellow)

    def reset(self):
        return self.state

    def step(self, action):
        N = self.num_envs
        z = torch.zeros(N, device=DEVICE)
        old_by = self.by.clone()
        flap = action == self.A_FLAP
        # gravity + flap impulse
        self.vy = self.vy + self.G
        self.vy = torch.where(flap, torch.full_like(self.vy, -self.FLAP), self.vy)
        self.by = self.by + self.vy
        self.by_prev = old_by         # one-step trail -> single frame now encodes velocity
        # scroll pipe
        prev_x = self.pipe_x.clone()
        self.pipe_x = self.pipe_x - self.SCROLL

        # crossing the bird's x-plane this step
        crossing = (prev_x > self.bird_x) & (self.pipe_x <= self.bird_x)
        in_gap = (self.by - self.gap_y).abs() <= (self.GH - self.bird_r)
        passed = crossing & in_gap
        hit_pipe = crossing & ~in_gap
        # ceiling / ground death
        oob = (self.by <= 0.0) | (self.by >= 1.0)
        dead = hit_pipe | oob

        # dense shaping: reward staying near the current gap centre (default off).
        near_gap = (1.0 - (self.by - self.gap_y).abs() * 3.0).clamp(min=0.0)
        reward = (self.survive * (~dead).float() + passed.float() - dead.float()
                  + self.shaping * near_gap * (~dead).float())
        self.score = self.score + passed.float()
        self.cum_score = self.cum_score + passed.float()

        # respawn pipe after it leaves the screen
        gone = self.pipe_x < -0.05
        new_gap = self._rand_gap()
        self.pipe_x = torch.where(gone, torch.ones_like(self.pipe_x), self.pipe_x)
        self.gap_y = torch.where(gone, new_gap, self.gap_y)

        self.steps = self.steps + 1
        truncated = self.steps >= self.max_steps
        terminated = dead
        done = terminated | truncated
        if bool(done.any()):
            self._reset_done(done)
        self._render()
        return self.state, reward, terminated, truncated, done

    def _reset_done(self, done):
        z = torch.zeros(self.num_envs, device=DEVICE)
        self.by = torch.where(done, z + 0.5, self.by)
        self.by_prev = torch.where(done, z + 0.5, self.by_prev)
        self.vy = torch.where(done, z, self.vy)
        self.pipe_x = torch.where(done, z + 1.0, self.pipe_x)
        self.gap_y = torch.where(done, self._rand_gap(), self.gap_y)
        self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        self.score = torch.where(done, z, self.score)

    def metrics(self):
        return dict(mean_cum_score=float(self.cum_score.mean()))
