"""Explorer V (docs/ARC4_V83_PROTOCOL.md §1; port of the v74 `Agent.explore`, learner.py:132-188).

Cell = (me on a 16x16 grid, or (-1,-1) when I am not found; lives; floor(4*log2(1+S))), S = cumulative
sensed rise along the cell's trajectory. A better cell has a higher S, then a shorter depth; an
improved cell is re-added fresh (v71 rule: chosen and visits restart). Choice: v71 weights,
vectorised: 1/sqrt(chosen+1), times 4 for cells at the best S when it is > 0. Rollouts of 100 steps,
a uniform button held with repeat 0.95; a lives decrease or the game's end stops the rollout and that
state is not archived. Probes at most once every 8 steps while I am lost. Cap 20,000 cells; when full,
the most-chosen cell below the best S is evicted (ties: the oldest; if none is below the best S, the
most-chosen of all). The run ends when the senses raise Spent (billed cap or evaluator stop).

Every 4th rollout step (from step 4) a promise row is logged (logs.RowLog). 'Me lost' = no position
or a tracker miss at this step (the entry's definition): the row is flagged and its window centred
on the screen; the cell key keeps the tracker's last position (v74). `steer` (promise arms) may replace
the cell and button draws, and receives on_put(ex, slot, ctx) for every added or replaced cell and
on_iter(ex, it) after each rollout (`ex.last_rows`: that rollout's row indices); with steer=None
this is arm V.
"""
import math
import time

import numpy as np

from ragnarok.agent_v83.logs import IterLog, RowLog
from ragnarok.agent_v83.views import fg_mask, log_view

GRID = 16


def s_bin(S):
    return int(math.floor(4.0 * math.log2(1.0 + S)))


def cell_key(me, lives, S, shape):
    if me is None:
        mb = (-1, -1)
    else:
        mb = (int(me[0] * GRID / shape[1]), int(me[1] * GRID / shape[0]))
    return (mb, int(lives), s_bin(S))


class Archive:
    """Vectorised cell store: arrays indexed by slot, key -> slot."""

    def __init__(self, cap=20000):
        self.cap = cap
        n = cap + 1
        self.S = np.zeros(n)
        self.depth = np.zeros(n, np.int64)
        self.chosen = np.zeros(n, np.int64)
        self.visits = np.zeros(n, np.int64)
        self.age = np.zeros(n, np.int64)
        self.handle = np.zeros(n, np.int64)
        self.mx = np.full(n, np.nan)
        self.my = np.full(n, np.nan)
        self.lives = np.zeros(n, np.int64)
        self.used = np.zeros(n, bool)
        self.slot = {}
        self.key_of = {}
        self.free_slots = list(range(n - 1, -1, -1))
        self.clock = 0
        self.evicted = 0

    def __len__(self):
        return len(self.slot)

    def best_S(self):
        return float(self.S[self.used].max()) if self.slot else 0.0

    def better(self, key, S, depth):
        i = self.slot.get(key)
        return i is None or S > self.S[i] or (S == self.S[i] and depth < self.depth[i])

    def put(self, key, h, S, depth, me, lives, env):
        """Add or replace (fresh) the cell; returns its slot."""
        i = self.slot.get(key)
        if i is not None:
            env.free(int(self.handle[i]))
        else:
            if len(self.slot) >= self.cap:
                self._evict(env)
            i = self.free_slots.pop()
            self.slot[key] = i
            self.key_of[i] = key
            self.used[i] = True
        self.clock += 1
        self.S[i], self.depth[i], self.chosen[i], self.visits[i] = S, depth, 0, 1
        self.age[i], self.handle[i], self.lives[i] = self.clock, h, lives
        self.mx[i], self.my[i] = (np.nan, np.nan) if me is None else me
        return i

    def _evict(self, env):
        idx = np.nonzero(self.used)[0]
        best = self.S[idx].max()
        below = idx[self.S[idx] < best]
        pool = below if len(below) else idx
        top = self.chosen[pool].max()
        cand = pool[self.chosen[pool] == top]
        i = int(cand[np.argmin(self.age[cand])])
        env.free(int(self.handle[i]))
        del self.slot[self.key_of.pop(i)]
        self.used[i] = False
        self.free_slots.append(i)
        self.evicted += 1

    def weights(self, score_bias=4.0):
        idx = np.nonzero(self.used)[0]
        best = self.S[idx].max()
        w = np.where((self.S[idx] == best) & (best > 0), score_bias, 1.0) / np.sqrt(self.chosen[idx] + 1.0)
        return idx, w

    def draw(self, rng, n=None):
        idx, w = self.weights()
        p = w / w.sum()
        if n is None:
            return int(idx[rng.choice(len(idx), p=p)])
        return idx[rng.choice(len(idx), size=n, p=p)]


class Explorer:
    def __init__(self, body, steer=None, steps=100, repeat=0.95, probe_every=8, cap=20000, row_every=4):
        self.body, self.steer = body, steer
        self.steps, self.repeat, self.probe_every, self.row_every = steps, repeat, probe_every, row_every
        self.arch = Archive(cap)
        self.rows = RowLog()
        self.iters = IterLog()
        self.last_rows = []

    def pick_cell(self, rng):
        if self.steer is not None:
            return self.steer.pick_cell(self, rng)
        i = self.arch.draw(rng)
        self.arch.chosen[i] += 1
        return i

    def pick_button(self, rng, n_actions, ctx):
        if self.steer is not None:
            return self.steer.pick_button(self, rng, n_actions, ctx)
        return int(rng.integers(n_actions))

    def run(self, env, rng, spent=Exception, max_iters=None):
        """Explore until the senses raise `spent` (or `max_iters`). Returns a summary dict."""
        body, arch, tr = self.body, self.arch, self.body.tracker
        t0 = time.time()
        it = 0
        try:
            obs, lives = env.reset()
            tr.pos, tr.miss = None, 0
            me = body.locate(env, obs)
            i0 = arch.put(cell_key(me, lives, 0.0, obs.shape), env.clone(), 0.0, 0, me, lives, env)
            if self.steer is not None:
                self.steer.on_put(self, i0, dict(obs=obs, prev_obs=None, me=me, prev_me=None, lives=lives,
                                                 lost=me is None or tr.miss > 0))
            while max_iters is None or it < max_iters:
                i = self.pick_cell(rng)
                obs, lives = env.restore(int(arch.handle[i]))
                pos = None if np.isnan(arch.mx[i]) else (float(arch.mx[i]), float(arch.my[i]))
                tr.pos, tr.miss = pos, 0
                S, depth = float(arch.S[i]), int(arch.depth[i])
                lost_steps, since_probe = 0, self.probe_every
                me, prev_obs, prev_me, lost = pos, None, None, pos is None
                ctx = dict(obs=obs, prev_obs=None, me=me, prev_me=None, lives=lives, lost=lost)
                a = self.pick_button(rng, env.n_actions, ctx)
                rises, drops, row_ids = [], [], []
                stop = "length"
                try:
                    for t in range(self.steps):
                        if rng.random() > self.repeat:
                            a = self.pick_button(rng, env.n_actions, ctx)
                        if t >= self.row_every and t % self.row_every == 0:
                            packed, screen, _, _ = log_view(obs, fg_mask(prev_obs, body.bg), body.bg, me, body.my_size,
                                                            lost=lost, prev_me=prev_me)
                            row_ids.append((t, self.rows.add(packed, screen, lost, me, lives,
                                                             body.adapter.to_effect(a), a, it, t, tr.miss)))
                        prev_obs, prev_me = obs, (None if lost else me)
                        obs, rise, nl, end = env.step(a)
                        depth += 1
                        S += rise
                        rises.append(rise > 0)
                        drops.append(nl < lives)
                        if end:
                            stop = "end"
                            break
                        if nl < lives:
                            stop = "drop"
                            break
                        lives = nl
                        me = body.track(obs)
                        since_probe += 1
                        if tr.lost and since_probe >= self.probe_every:
                            since_probe = 0
                            found = body.locate(env, obs)
                            if found is not None:
                                me = found
                        lost = me is None or tr.miss > 0          # the entry's definition (entry.py)
                        if lost:
                            lost_steps += 1
                        ctx = dict(obs=obs, prev_obs=prev_obs, me=me, prev_me=prev_me, lives=lives, lost=lost)
                        key = cell_key(me, lives, S, obs.shape)
                        if arch.better(key, S, depth):
                            j = arch.put(key, env.clone(), S, depth, me, lives, env)
                            if self.steer is not None:
                                self.steer.on_put(self, j, ctx)
                        else:
                            arch.visits[arch.slot[key]] += 1
                finally:
                    self.rows.label(row_ids, rises, drops)
                it += 1
                self.iters.add(env.billed, arch.best_S(), len(arch), body.probe_frames, lost_steps, stop)
                self.last_rows = [r for _, r in row_ids]
                if self.steer is not None:
                    self.steer.on_iter(self, it)
        except spent:
            pass
        return dict(iters=it, cells=len(arch), best_S=arch.best_S(), evicted=arch.evicted, billed=env.billed,
                    probe_calls=body.probe_calls, probe_frames=body.probe_frames, wall_s=time.time() - t0)
