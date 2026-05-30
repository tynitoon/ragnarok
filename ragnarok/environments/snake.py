"""DeviceVecSnake — a GPU-batched, device-resident Snake with PIXEL obs.

The illustration of "no end, maximize points": eat food (GREEN head, +1 each),
grow, and the score climbs indefinitely until you die (hit a wall or yourself,
-1 + game over). Same device-resident paradigm: N games on the accelerator,
`.state` is a flattened RGB image, `.step(action)` a discrete action per env.

Rung P2 of the game curriculum: endless score-maximization + survival, learned
from pixels by the SAME agent as Pong/Breakout. Actions: 0 up, 1 down, 2 left,
3 right (a 180-degree reversal is ignored).

Batched body representation (no per-env variable-length lists): each cell stores
the global step `t_enter` at which the head last entered it; a cell is part of
the body iff `T - t_enter < length`. Eating food increments `length`, so the
tail lingers one extra step => the snake grows. Stale cells vacate automatically
as T advances.
"""

import torch

from ragnarok.infrastructure.device import DEVICE

NEG = -10_000


class DeviceVecSnake:
    def __init__(self, num_envs, grid=12, tile=4, max_steps=400, init_len=3,
                 survive_bonus=0.01, img=None, seed=0):
        self.num_envs = num_envs
        self.G = grid
        # accept an `img` size for a uniform game interface (derive tile)
        self.tile = max(1, img // grid) if img is not None else tile
        self.H = self.W = grid * tile
        self.img_hw = grid * tile
        self.obs_dim = 3 * self.img_hw * self.img_hw
        self.action_dim = 4
        self.max_steps = max_steps
        self.init_len = init_len
        self.SURV = survive_bonus
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        self._dr = torch.tensor([-1, 1, 0, 0], device=DEVICE)
        self._dc = torch.tensor([0, 0, -1, 1], device=DEVICE)
        self._opp = torch.tensor([1, 0, 3, 2], device=DEVICE)
        self.T = 0
        self.cum_food = torch.zeros(num_envs, device=DEVICE)
        self.cum_deaths = torch.zeros(num_envs, device=DEVICE)
        self.reset()

    def _occupied(self):
        return (self.t_enter > NEG // 2) & \
               ((self.T - self.t_enter) < self.length.view(-1, 1, 1))

    def _spawn_food(self, mask):
        N, G = self.num_envs, self.G
        need = mask.clone()
        ar = torch.arange(N, device=DEVICE)
        for _ in range(16):
            if not bool(need.any()):
                break
            rr = torch.randint(0, G, (N,), generator=self._gen, device=DEVICE)
            rc = torch.randint(0, G, (N,), generator=self._gen, device=DEVICE)
            occ = (self.t_enter[ar, rr, rc] > NEG // 2) & \
                  ((self.T - self.t_enter[ar, rr, rc]) < self.length)
            free = need & ~occ
            self.fr = torch.where(free, rr, self.fr)
            self.fc = torch.where(free, rc, self.fc)
            need = need & ~free
        if bool(need.any()):                       # fallback: place anywhere
            self.fr = torch.where(need, torch.randint(0, G, (N,), generator=self._gen, device=DEVICE), self.fr)
            self.fc = torch.where(need, torch.randint(0, G, (N,), generator=self._gen, device=DEVICE), self.fc)

    def _reset(self, mask):
        N, G = self.num_envs, self.G
        cr, cc = G // 2, G // 2
        if bool(mask.any()):
            self.t_enter[mask] = NEG
            # head + (init_len-1) body cells extending left; moving right
            for k in range(self.init_len):
                self.t_enter[mask, cr, cc - k] = self.T - k
        self.hr = torch.where(mask, torch.full_like(self.hr, cr), self.hr)
        self.hc = torch.where(mask, torch.full_like(self.hc, cc), self.hc)
        self.dir = torch.where(mask, torch.full_like(self.dir, 3), self.dir)   # right
        self.length = torch.where(mask, torch.full_like(self.length, self.init_len), self.length)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self._spawn_food(mask)

    def reset(self):
        N, G = self.num_envs, self.G
        self.t_enter = torch.full((N, G, G), NEG, dtype=torch.long, device=DEVICE)
        self.hr = torch.full((N,), G // 2, dtype=torch.long, device=DEVICE)
        self.hc = torch.full((N,), G // 2, dtype=torch.long, device=DEVICE)
        self.dir = torch.full((N,), 3, dtype=torch.long, device=DEVICE)
        self.length = torch.full((N,), self.init_len, dtype=torch.long, device=DEVICE)
        self.fr = torch.zeros(N, dtype=torch.long, device=DEVICE)
        self.fc = torch.zeros(N, dtype=torch.long, device=DEVICE)
        self.steps = torch.zeros(N, dtype=torch.long, device=DEVICE)
        self._reset(torch.ones(N, dtype=torch.bool, device=DEVICE))
        self._render()
        return self.state

    def _render(self):
        N, G = self.num_envs, self.G
        ar = torch.arange(N, device=DEVICE)
        occ = self._occupied().float()                       # (N,G,G)
        g = torch.zeros(N, 3, G, G, device=DEVICE)
        g[:, 1] = occ * 0.55                                  # body: green
        g[ar, 1, self.hr, self.hc] = 1.0                     # head: bright green
        g[ar, 0, self.fr, self.fc] = 1.0                     # food: red
        img = g.repeat_interleave(self.tile, 2).repeat_interleave(self.tile, 3)
        self._img = img.reshape(N, -1)

    @property
    def state(self):
        return self._img

    def step(self, action):
        N, G = self.num_envs, self.G
        a = action.reshape(N).long()
        ar = torch.arange(N, device=DEVICE)
        # turn (ignore 180-degree reversal)
        valid = a != self._opp[self.dir]
        self.dir = torch.where(valid, a, self.dir)
        nhr = self.hr + self._dr[self.dir]
        nhc = self.hc + self._dc[self.dir]
        wall = (nhr < 0) | (nhr >= G) | (nhc < 0) | (nhc >= G)
        nhr_c, nhc_c = nhr.clamp(0, G - 1), nhc.clamp(0, G - 1)
        age = self.T - self.t_enter[ar, nhr_c, nhc_c]
        seen = self.t_enter[ar, nhr_c, nhc_c] > NEG // 2
        self_coll = (~wall) & seen & (age < self.length - 1)  # stays occupied after move
        dead = wall | self_coll
        ate = (~dead) & (nhr == self.fr) & (nhc == self.fc)
        alive = ~dead

        self.T += 1
        if bool(alive.any()):
            self.t_enter[ar[alive], nhr_c[alive], nhc_c[alive]] = self.T
        self.hr = torch.where(alive, nhr_c, self.hr)
        self.hc = torch.where(alive, nhc_c, self.hc)
        self.length = self.length + ate.long()
        reward = ate.float() - dead.float() + alive.float() * self.SURV
        self.cum_food += ate.float()
        self.cum_deaths += dead.float()
        self._spawn_food(ate)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = dead
        done = terminated | truncated
        self._reset(done)
        self._render()
        return self.state, reward, terminated, truncated, done

    def stats(self):
        return dict(mean_food=float(self.cum_food.mean()),
                    deaths=float(self.cum_deaths.sum()))
