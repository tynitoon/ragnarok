"""V2s, v84's V2 from a shared start (docs/ARC4_V85_PROTOCOL_EN.md A.6). Same law as V2; not the same bytes.

- Root: s1 (the env has done `start_from`); best_S = S(s1), best_S_billed = billed at run start,
  best_S_exp = 0, any_rise = False.
- Streams per expansion e: P_{e,j} = [seed, 85, e, j, 0] (cell, process, first slot of proposal j);
  X_{e,j} = [seed, 85, e, j, 1] (look-ahead j, used by the evaluator); X_e = [seed, 85, e, 1000] (the
  executed rollout's in-rollout draws); C_e = [seed, 85, e, 2000] (steering coins and steered picks);
  propensity [seed, 1, e].
- `propose(e, j, cell=None)` has no side effects (no `chosen` count, no bandit or archive write).
  V2s executes `propose(e, 0)` on X_e; `chosen` is incremented for the executed cell only.
- `_rollout(..., dry=True)` writes nothing (archive read-only plus a per-candidate key overlay, eviction
  not simulated; local best_S and any_rise); the evaluator wraps it in its sandbox.
- Row logging off; exp_meta and exp_feat kept (features after the restore, as in V2; the rank of a
  cell's depth breaks ties by slot index).
- Injected objects (this module imports no family or evaluation module):
  steer: pick_cell(ex, rng_P, rng_C, e), pick_process(ex, rng_P, rng_C, i), on_put(ex, j), on_iter(ex, e),
         to_state(), from_state(s);
  q5 (process 4, bandit k = 5): first_slot(ex, i, rng), next_slot(ex, t, slot, rng), to_state(), from_state(s);
  chooser(ex, env, e) -> (i, q, slot) replaces propose(e, 0);
  boundary(ex, e) is called at every expansion boundary (checkpoints).
  With none of them the bandit has k = 4.
"""
import math

import numpy as np

from ragnarok.agent_v83.explorer import s_bin
from ragnarok.agent_v84.explorer2 import N_EXP_FEAT, PROCS, REPEAT, Archive2, Bandit, Explorer2, slot_class
from ragnarok.agent_v84.framecode import RES, codes

class Explorer2s(Explorer2):
    def __init__(self, body, budget, seed, steer=None, q5=None, chooser=None, boundary=None, cap=20000,
                 frontier=0.25, base_len=100, ext=50, max_len=400, rekey_every=250):
        self.body, self.budget, self.seed = body, budget, int(seed)
        self.steer, self.q5, self.chooser, self.boundary = steer, q5, chooser, boundary
        self.arch = Archive2(cap)
        self.frontier = frontier
        self.base_len, self.ext, self.max_len, self.rekey_every = base_len, ext, max_len, rekey_every
        self.bandit = Bandit(k=4 if q5 is None else 5)
        self.exp_rows, self.exp_meta = [], []
        self.coarse_count = {}
        self.hist = []
        self.best_S, self.best_S_billed, self.best_S_exp = 0.0, 0, 0
        self.any_rise = False
        self.slots = None
        self.exp_n, self.b0 = 0, 0

    # ---------------------------------------------------------------- streams and proposals
    def stream(self, e, j, k):
        return np.random.default_rng([self.seed, 85, e, j, k])

    def draw_cell(self, rng, e):
        """V2's cell draw, without the `chosen` increment."""
        A = self.arch
        if rng.random() < self.frontier:
            idx = np.nonzero(A.used)[0]
            top = idx[A.sbin[idx] == A.sbin[idx].max()]
            top = top[A.lvs[top] == A.lvs[top].max()]
            return int(top[rng.integers(len(top))])
        idx, w = A.weights(e)
        return int(idx[rng.choice(len(idx), p=w / w.sum())])

    def propose(self, e, j, cell=None):
        rng = self.stream(e, j, 0)
        rc = np.random.default_rng([self.seed, 85, e, 2000]) if self.steer is not None else None
        if cell is not None:
            i = int(cell)
        elif self.steer is not None:
            i = int(self.steer.pick_cell(self, rng, rc, e))
        else:
            i = self.draw_cell(rng, e)
        q = int(self.steer.pick_process(self, rng, rc, i)) if self.steer is not None else self.bandit.pick(rng)
        slot = self.q5.first_slot(self, i, rng) if q == 4 else self._first_slot(q, i, rng)
        return i, q, int(slot)

    # ---------------------------------------------------------------- features
    def _features(self, i, q, slot0, env):
        A = self.arch
        idx = np.nonzero(A.used)[0]
        sb_max, L_max = A.sbin[idx].max(), A.lvs[idx].max()
        d, di = A.depth[idx], A.depth[i]
        drank = float(np.count_nonzero(d < di) + np.count_nonzero((d == di) & (idx < i)))
        par = A.parent[i]
        pyield = (A.adv_children[par] + 1) / (A.chosen[par] + 2) if par >= 0 and A.used[par] else 0.5
        occ = int(np.count_nonzero(A.sbin[idx] == A.sbin[i]))
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
        ctx = [(env.billed - self.b0) / self.budget, adv_rate, loss_rate, math.log1p(stagn), float(self.any_rise),
               math.log1p(max(0, env.billed - self.best_S_billed)), new_rate]
        inter = [o * v for o in oh for v in (cell[1], math.log1p(stagn), loss_rate, cell[0])]
        f = cell + oh + sc + ctx + self._adapter_sig() + inter
        assert len(f) == N_EXP_FEAT
        return np.array(f, np.float32)

    # ---------------------------------------------------------------- one rollout
    def _rollout(self, env, i, q, slot, rng, e, dry=False, restored=None):
        A, body = self.arch, self.body
        tr, ad = body.tracker, body.adapter
        obs, lives = env.restore(int(A.handle[i])) if restored is None else restored
        pos = None if np.isnan(A.mx[i]) else (float(A.mx[i]), float(A.my[i]))
        tr.pos, tr.miss = pos, 0
        S, depth, since_rise = float(A.S[i]), int(A.depth[i]), int(A.since_rise[i])
        start_sbin, cur_parent, after_loss = int(A.sbin[i]), i, bool(A.after_loss[i])
        best_S, any_rise, max_S = self.best_S, self.any_rise, S
        over, top_sb = {}, int(A.sbin[A.used].max())
        L, t, losses, adv, new_cells = self.base_len, 0, 0, False, 0
        while t < L:
            if q == 4:
                if t:
                    slot = int(self.q5.next_slot(self, t, slot, rng))
            elif t and REPEAT[PROCS[q]] < 1.0 and rng.random() > REPEAT[PROCS[q]]:
                slot = int(self.slots[rng.integers(len(self.slots))])
            obs, rise, nl, end = env.step(ad.to_button(slot))
            t += 1
            depth += 1
            S += rise
            max_S = max(max_S, S)
            since_rise = 0 if rise > 0 else since_rise + 1
            if rise > 0:
                any_rise = True
            if end:
                break
            if nl < lives:
                losses += 1
                after_loss = True
                if losses >= 2:
                    break
            lives = nl
            me = body.track(obs)
            cds, fine = codes(obs)
            key = (int(lives), s_bin(S), cds[A.r])
            if dry:
                if A.better(key, S, lives, depth) and (key not in over or (S, lives, -depth) > over[key]):
                    new = key not in A.slot and key not in over
                    over[key] = (S, lives, -depth)
                    sb = s_bin(S)
                    if sb > start_sbin or (new and sb >= top_sb):
                        adv = True
                    top_sb = max(top_sb, sb)
                    new_cells += int(new)
            elif A.better(key, S, lives, depth):
                j, new = self._put(key, env, S, int(lives), depth, me, cds, fine, cur_parent, since_rise,
                                   after_loss, slot, e)
                sb = int(A.sbin[j])
                idx = np.nonzero(A.used)[0]
                if sb > start_sbin or (new and sb == int(A.sbin[idx].max())):
                    adv = True
                new_cells += int(new)
                cur_parent = j
                if self.steer is not None:
                    self.steer.on_put(self, j)
            if S > best_S:
                best_S = S
                if not dry:
                    self.best_S, self.best_S_billed, self.best_S_exp = S, env.billed, e
                L = min(self.max_len, L + self.ext)
        if not dry:
            self.any_rise = any_rise
        return dict(adv=adv, losses=losses, new_cells=new_cells, steps=t, max_S=max_S)

    # ---------------------------------------------------------------- run
    def run(self, env, obs=None, lives=None, S1=None, spent=Exception, resume=False):
        """Fresh: obs, lives, S1 from env.start_from(s1). Resume: after from_state, on an env restored per §13.3."""
        A = self.arch
        try:
            if not resume:
                self.slots = sorted(self.body.adapter.buttons)
                self.b0 = env.billed
                tr = self.body.tracker
                tr.pos, tr.miss = None, 0
                me = self.body.track(obs)
                cds, fine = codes(obs)
                self.best_S, self.best_S_billed, self.best_S_exp, self.any_rise = float(S1), env.billed, 0, False
                self._put((int(lives), s_bin(float(S1)), cds[A.r]), env, float(S1), int(lives), 0, me, cds, fine,
                          -1, 0, False, -1, 0)
            while True:
                e = self.exp_n
                if self.boundary is not None:
                    self.boundary(self, e)
                if e and e % self.rekey_every == 0:
                    r = self._target_r(e)
                    if r != A.r:
                        A.rekey(r, env)
                i, q, slot = self.chooser(self, env, e) if self.chooser is not None else self.propose(e, 0)
                A.chosen[i] += 1
                prop = self.bandit.propensity(q, np.random.default_rng([self.seed, 1, e]))
                restored = env.restore(int(A.handle[i]))
                feat = self._features(i, q, slot, env)
                out = self._rollout(env, i, q, slot, np.random.default_rng([self.seed, 85, e, 1000]), e,
                                    restored=restored)
                adv = out["adv"]
                A.adv_children[i] += float(adv)
                self.bandit.update(q, adv)
                self.hist.append((float(adv), float(out["losses"]), float(out["new_cells"])))
                self.exp_rows.append(feat)
                self.exp_meta.append((e, i, q, int(adv), int(env.billed), prop, out["losses"], out["steps"]))
                self.exp_n = e + 1
                if self.steer is not None:
                    self.steer.on_iter(self, e + 1)
        except spent:
            pass
        return dict(expansions=self.exp_n, cells=len(A), best_S=self.best_S, evicted=A.evicted,
                    billed=env.billed - self.b0, resolution=[int(v) for v in RES[A.r]], rows=0,
                    probe_frames_after_entry=0)

    # ---------------------------------------------------------------- checkpoints (§13.3)
    def to_state(self):
        """A dict for the evaluator's checkpoint writer (it pickles it at once)."""
        return dict(
            arch=self.arch, bandit=self.bandit, hist=self.hist, coarse_count=self.coarse_count, best_S=self.best_S,
            best_S_billed=self.best_S_billed, best_S_exp=self.best_S_exp, any_rise=self.any_rise,
            exp_rows=self.exp_rows, exp_meta=self.exp_meta, exp_n=self.exp_n, b0=self.b0, slots=self.slots,
            tracker=self.body.tracker, steer=None if self.steer is None else self.steer.to_state(),
            q5=None if self.q5 is None else self.q5.to_state())

    def from_state(self, d):
        self.arch, self.bandit, self.hist, self.coarse_count = d["arch"], d["bandit"], d["hist"], d["coarse_count"]
        self.best_S, self.best_S_billed, self.best_S_exp = d["best_S"], d["best_S_billed"], d["best_S_exp"]
        self.any_rise, self.exp_rows, self.exp_meta = d["any_rise"], d["exp_rows"], d["exp_meta"]
        self.exp_n, self.b0, self.slots = d["exp_n"], d["b0"], d["slots"]
        self.body.tracker = d["tracker"]
        if self.steer is not None:
            self.steer.from_state(d["steer"])
        if self.q5 is not None:
            self.q5.from_state(d["q5"])
