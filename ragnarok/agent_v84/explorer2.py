"""V2, the fresh climber (docs/ARC4_V84_PROTOCOL_EN.md §5).

Cell = (lives, s_bin(S), code_r(frame)), S the cumulative sensed rise along the trajectory; the frame
code resolution r is chosen every 250 expansions so that the number of distinct keys is closest in log
to clip(30 sqrt(n), 64, 8000), and the archive is re-keyed (better cell: higher S, then more lives,
then shorter depth). Selection weight (1 + chosen)^-1/2 * 2^((sbin - sbin_max)/2) * 2^-(L_max - L) *
(2 if added or improved in the last 100 expansions); with probability 0.25 a frontier draw (uniform among
the cells at sbin_max with the most lives). Button processes Q1 (new uniform slot every step), Q2 (held,
repeat 0.95), Q3 (held, repeat 0.75), Q4 (long hold, repeat 0.995, first slot inherited with probability
0.5), chosen by a Thompson bandit on ADV (Beta(1 + ADV, 1 + not ADV), counts discounted 0.995 per
expansion, 20% uniform floor; propensity logged from a separate generator). Rollouts of 100 steps,
extended by 50 at each new archive-best S, up to 400. A life loss archives the post-loss state and play
continues; the rollout stops at its second loss or at game end. No probes after entry: me comes from
Body.track only. ADV: the rollout created a cell above its start cell's sbin, or a new cell in the top
sbin. Cap 20,000 cells, v83 eviction (most chosen below the best sbin, oldest first).
"""
import math

import numpy as np

from ragnarok.agent_v83.explorer import s_bin
from ragnarok.agent_v83.views import fg_mask, log_view
from ragnarok.agent_v84.framecode import RES, codes
from ragnarok.agent_v84.rowlog import DiskRowLog

PROCS = ("Q1", "Q2", "Q3", "Q4")
REPEAT = dict(Q1=0.0, Q2=0.95, Q3=0.75, Q4=0.995)
N_EXP_FEAT = 44


def slot_class(slot_name):
    """(moving, vertical, extra) of a canonical slot name 'sx,sy,ex' or 'privK'."""
    if slot_name.startswith("priv"):
        return 0.0, 0.0, 1.0
    sx, sy, ex = (int(v) for v in slot_name.split(","))
    return float(sx != 0 or sy != 0), float(sy != 0), float(ex)


class Archive2:
    def __init__(self, cap=20000):
        self.cap = cap
        n = cap + 1
        z = lambda dt=np.float64: np.zeros(n, dt)
        self.S, self.sbin, self.lvs, self.depth = z(), z(np.int64), z(np.int64), z(np.int64)
        self.chosen, self.age, self.handle = z(np.int64), z(np.int64), z(np.int64)
        self.last_imp, self.added, self.adv_children = z(np.int64), z(np.int64), z()
        self.parent, self.since_rise, self.after_loss = np.full(n, -1, np.int64), z(np.int64), z(bool)
        self.last_slot, self.change = np.full(n, -1, np.int64), z()
        self.mx, self.my = np.full(n, np.nan), np.full(n, np.nan)
        self.used = np.zeros(n, bool)
        self.codes = [None] * n
        self.fine = [None] * n
        self.key_of = {}
        self.slot = {}
        self.free_slots = list(range(n - 1, -1, -1))
        self.clock = 0
        self.evicted = 0
        self.r = 1                                  # initial resolution (11, 8, 8): Go-Explore's default

    def __len__(self):
        return len(self.slot)

    def key(self, i):
        return (int(self.lvs[i]), int(self.sbin[i]), self.codes[i][self.r])

    def better(self, key, S, lives, depth):
        i = self.slot.get(key)
        if i is None:
            return True
        return (S, lives, -depth) > (self.S[i], self.lvs[i], -self.depth[i])

    def put(self, key, h, S, lives, depth, me, cds, fine, parent, since_rise, after_loss, last_slot, exp_n, env):
        i = self.slot.get(key)
        new = i is None
        if not new:
            env.free(int(self.handle[i]))
        else:
            if len(self.slot) >= self.cap:
                self._evict(env)
            i = self.free_slots.pop()
            self.slot[key] = i
            self.key_of[i] = key
            self.used[i] = True
            self.added[i] = exp_n
            self.chosen[i] = 0
            self.adv_children[i] = 0.0
        self.clock += 1
        self.S[i], self.sbin[i], self.lvs[i], self.depth[i] = S, s_bin(S), lives, depth
        self.age[i], self.handle[i], self.last_imp[i] = self.clock, h, exp_n
        self.parent[i], self.since_rise[i], self.after_loss[i], self.last_slot[i] = parent, since_rise, after_loss, last_slot
        self.mx[i], self.my[i] = (np.nan, np.nan) if me is None else me
        self.codes[i], self.fine[i] = cds, fine
        pf = self.fine[parent] if parent >= 0 and self.used[parent] else None
        self.change[i] = float(np.mean(pf != fine)) if pf is not None else 0.0
        return i, new

    def _drop(self, i, env):
        env.free(int(self.handle[i]))
        del self.slot[self.key_of.pop(i)]
        self.used[i] = False
        self.codes[i] = self.fine[i] = None
        self.free_slots.append(i)

    def _evict(self, env):
        idx = np.nonzero(self.used)[0]
        best = self.sbin[idx].max()
        below = idx[self.sbin[idx] < best]
        pool = below if len(below) else idx
        top = self.chosen[pool].max()
        cand = pool[self.chosen[pool] == top]
        self._drop(int(cand[np.argmin(self.age[cand])]), env)
        self.evicted += 1

    def distinct(self, r):
        idx = np.nonzero(self.used)[0]
        return len({(int(self.lvs[i]), int(self.sbin[i]), self.codes[i][r]) for i in idx})

    def rekey(self, r, env):
        """Re-key on resolution r; colliding cells: the better one stays, the others are dropped."""
        self.r = r
        idx = np.nonzero(self.used)[0]
        best = {}
        for i in idx:
            k = self.key(i)
            j = best.get(k)
            if j is None or (self.S[i], self.lvs[i], -self.depth[i]) > (self.S[j], self.lvs[j], -self.depth[j]):
                best[k] = i
        keep = set(best.values())
        self.slot, self.key_of = {}, {}
        for k, i in best.items():
            self.slot[k] = i
            self.key_of[i] = k
        for i in idx:
            if i not in keep:
                env.free(int(self.handle[i]))
                self.used[i] = False
                self.codes[i] = self.fine[i] = None
                self.free_slots.append(int(i))

    def weights(self, exp_n):
        idx = np.nonzero(self.used)[0]
        sb, lv = self.sbin[idx], self.lvs[idx]
        w = (1.0 + self.chosen[idx]) ** -0.5 * 2.0 ** ((sb - sb.max()) / 2.0) * 2.0 ** (-(lv.max() - lv))
        w = w * np.where(exp_n - self.last_imp[idx] < 100, 2.0, 1.0)
        return idx, w


class Bandit:
    def __init__(self, k=4, discount=0.995, floor=0.2):
        self.a, self.b = np.zeros(k), np.zeros(k)
        self.k, self.discount, self.floor = k, discount, floor

    def pick(self, rng):
        if rng.random() < self.floor:
            return int(rng.integers(self.k))
        return int(np.argmax(rng.beta(1 + self.a, 1 + self.b)))

    def propensity(self, q, rng_p, n=200):
        draws = rng_p.beta(1 + self.a[:, None], 1 + self.b[:, None], size=(self.k, n))
        p_ts = float(np.mean(np.argmax(draws, axis=0) == q))
        return self.floor / self.k + (1 - self.floor) * p_ts

    def update(self, q, adv):
        self.a *= self.discount
        self.b *= self.discount
        self.a[q] += float(adv)
        self.b[q] += float(not adv)


class Explorer2:
    def __init__(self, body, rows_dir, budget, steer=None, cap=20000, row_every=4, frontier=0.25,
                 base_len=100, ext=50, max_len=400, rekey_every=250):
        self.body, self.steer, self.budget = body, steer, budget
        self.arch = Archive2(cap)
        self.rows = DiskRowLog(rows_dir)
        self.row_every, self.frontier = row_every, frontier
        self.base_len, self.ext, self.max_len, self.rekey_every = base_len, ext, max_len, rekey_every
        self.bandit = Bandit()
        self.exp_rows, self.exp_meta = [], []
        self.coarse_count = {}
        self.hist = []                               # per expansion: (adv, losses, new_cells)
        self.best_S, self.best_S_billed, self.best_S_exp = 0.0, 0, 0
        self.any_rise = False
        self.slots = None

    # ---------------------------------------------------------------- helpers
    def _slot_names(self):
        from ragnarok.agent_v83.adapter import SLOTS
        return SLOTS

    def _adapter_sig(self):
        names = self._slot_names()
        cls = [slot_class(names[s]) for s in self.slots]
        n_move = sum(1 for c in cls if c[0])
        spawn = any((not c[0]) and c[2] for c in cls)
        return [float(n_move), float(spawn)]

    def _target_r(self, n):
        target = min(max(30.0 * math.sqrt(max(n, 1)), 64.0), 8000.0)
        counts = [self.arch.distinct(r) for r in range(len(RES))]
        return int(np.argmin([abs(math.log(max(c, 1)) - math.log(target)) for c in counts]))

    def _put(self, key, env, S, lives, depth, me, cds, fine, parent, since_rise, after_loss, last_slot, exp_n):
        h = env.clone()
        i, new = self.arch.put(key, h, S, lives, depth, me, cds, fine, parent, since_rise, after_loss, last_slot, exp_n, env)
        if new:
            self.coarse_count[cds[0]] = self.coarse_count.get(cds[0], 0) + 1
        return i, new

    def _features(self, i, q, slot0, env):
        A = self.arch
        idx = np.nonzero(A.used)[0]
        sb_max, L_max = A.sbin[idx].max(), A.lvs[idx].max()
        ranks = np.argsort(np.argsort(A.depth[idx]))
        drank = float(ranks[np.searchsorted(idx, i)])
        par = A.parent[i]
        pyield = (A.adv_children[par] + 1) / (A.chosen[par] + 2) if par >= 0 and A.used[par] else 0.5
        occ = int(np.sum(A.sbin[idx] == A.sbin[i]))
        cell = [float(A.sbin[i] - sb_max), float(L_max - A.lvs[i]), math.log1p(drank), math.log1p(A.chosen[i]),
                (A.adv_children[i] + 1) / (A.chosen[i] + 2), pyield, math.log1p(max(0, len(self.hist) - A.added[i])),
                math.log1p(A.since_rise[i]), float(A.after_loss[i]), math.log1p(self.coarse_count.get(A.codes[i][0], 1)),
                float(A.change[i]), math.log1p(occ)]
        oh = [float(q == k) for k in range(4)]
        sc = list(slot_class(self._slot_names()[slot0]))
        last = self.hist[-200:]
        adv_rate = float(np.mean([h[0] for h in last])) if last else 0.0
        loss_rate = float(np.mean([h[1] for h in last])) if last else 0.0
        new_rate = float(np.mean([h[2] for h in last])) if last else 0.0
        stagn = len(self.hist) - self.best_S_exp
        ctx = [env.billed / self.budget, adv_rate, loss_rate, math.log1p(stagn), float(self.any_rise),
               math.log1p(max(0, env.billed - self.best_S_billed)), new_rate]
        inter = [o * v for o in oh for v in (cell[1], math.log1p(stagn), loss_rate, cell[0])]
        f = cell + oh + sc + ctx + self._adapter_sig() + inter
        assert len(f) == N_EXP_FEAT
        return np.array(f, np.float32)

    def pick_cell(self, rng, exp_n):
        A = self.arch
        if self.steer is not None:
            return self.steer.pick_cell(self, rng, exp_n)
        if rng.random() < self.frontier:
            idx = np.nonzero(A.used)[0]
            top = idx[A.sbin[idx] == A.sbin[idx].max()]
            top = top[A.lvs[top] == A.lvs[top].max()]
            i = int(top[rng.integers(len(top))])
        else:
            idx, w = A.weights(exp_n)
            i = int(idx[rng.choice(len(idx), p=w / w.sum())])
        A.chosen[i] += 1
        return i

    def pick_process(self, rng, i):
        if self.steer is not None:
            return self.steer.pick_process(self, rng, i)
        return self.bandit.pick(rng)

    def _first_slot(self, q, i, rng):
        last = int(self.arch.last_slot[i])
        if q == 3 and last >= 0 and rng.random() < 0.5:
            return last
        return int(self.slots[rng.integers(len(self.slots))])

    # ---------------------------------------------------------------- run
    def run(self, env, rng, rng_prop, spent=Exception):
        body, A, tr = self.body, self.arch, self.body.tracker
        ad = body.adapter
        self.slots = sorted(ad.buttons)
        exp_n = 0
        try:
            obs, lives = env.reset()
            tr.pos, tr.miss = None, 0
            me = body.track(obs)
            cds, fine = codes(obs)
            self._put((int(lives), 0, cds[A.r]), env, 0.0, int(lives), 0, me, cds, fine, -1, 0, False, -1, 0)
            while True:
                if exp_n and exp_n % self.rekey_every == 0:
                    r = self._target_r(exp_n)
                    if r != A.r:
                        A.rekey(r, env)
                i = self.pick_cell(rng, exp_n)
                q = self.pick_process(rng, i)
                prop = self.bandit.propensity(q, rng_prop)
                obs, lives = env.restore(int(A.handle[i]))
                pos = None if np.isnan(A.mx[i]) else (float(A.mx[i]), float(A.my[i]))
                tr.pos, tr.miss = pos, 0
                S, depth, since_rise = float(A.S[i]), int(A.depth[i]), int(A.since_rise[i])
                start_sbin, cur_parent = int(A.sbin[i]), i
                slot = self._first_slot(q, i, rng)
                feat = self._features(i, q, slot, env)
                me, prev_obs, prev_me, lost, losses = pos, None, None, pos is None, 0
                after_loss = bool(A.after_loss[i])
                L = self.base_len
                rises, drops, adv, new_cells = [], [], False, 0
                t = 0
                try:
                    while t < L:
                        if t and REPEAT[PROCS[q]] < 1.0 and rng.random() > REPEAT[PROCS[q]]:
                            slot = int(self.slots[rng.integers(len(self.slots))])
                        a = ad.to_button(slot)
                        if t >= self.row_every and t % self.row_every == 0:
                            packed, screen, _, _ = log_view(obs, fg_mask(prev_obs, body.bg), body.bg, me, body.my_size,
                                                            lost=lost, prev_me=prev_me)
                            self.rows.add(view=packed, screen=screen, lost=lost,
                                          mx=np.nan if me is None else me[0], my=np.nan if me is None else me[1],
                                          lives=lives, slot=slot, button=a, it=exp_n, t=t, L_t=L, miss=min(255, int(tr.miss)),
                                          rise_lag=255, drop_lag=255, avail=0)
                        prev_obs, prev_me = obs, (None if lost else me)
                        obs, rise, nl, end = env.step(a)
                        t += 1
                        depth += 1
                        S += rise
                        since_rise = 0 if rise > 0 else since_rise + 1
                        if rise > 0:
                            self.any_rise = True
                        rises.append(rise > 0)
                        drops.append(nl < lives)
                        if end:
                            break
                        if nl < lives:
                            losses += 1
                            after_loss = True
                            if losses >= 2:
                                break
                        lives = nl
                        me = body.track(obs)
                        lost = me is None or tr.miss > 0
                        cds, fine = codes(obs)
                        key = (int(lives), s_bin(S), cds[A.r])
                        if A.better(key, S, lives, depth):
                            j, new = self._put(key, env, S, int(lives), depth, me, cds, fine, cur_parent, since_rise,
                                               after_loss, slot, exp_n)
                            sb = int(A.sbin[j])
                            idx = np.nonzero(A.used)[0]
                            if sb > start_sbin or (new and sb == int(A.sbin[idx].max())):
                                adv = True
                            new_cells += int(new)
                            cur_parent = j
                            if self.steer is not None:
                                self.steer.on_put(self, j)
                        if S > self.best_S:
                            self.best_S, self.best_S_billed, self.best_S_exp = S, env.billed, exp_n
                            L = min(self.max_len, L + self.ext)
                finally:
                    self.rows.end_rollout(rises, drops)
                A.adv_children[i] += float(adv)
                self.bandit.update(q, adv)
                self.hist.append((float(adv), float(losses), float(new_cells)))
                self.exp_rows.append(feat)
                self.exp_meta.append((exp_n, i, q, int(adv), int(env.billed), prop, losses, len(rises)))
                exp_n += 1
                if self.steer is not None:
                    self.steer.on_iter(self, exp_n)
        except spent:
            pass
        self.rows.close()
        return dict(expansions=exp_n, cells=len(A), best_S=self.best_S, evicted=A.evicted, billed=env.billed,
                    resolution=list(RES[A.r]), rows=self.rows.n_rows, probe_frames_after_entry=0)
