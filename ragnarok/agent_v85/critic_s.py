"""Route S online steering, family W "which proposal to bill" (docs/ARC4_V85_PROTOCOL_EN.md §4.4; v84 §6.2,
A.5-A.8). Agent side: this module imports no evaluation module and receives every carried quantity as DATA.

`CriticS` is the steering object injected into Explorer2s (hooks pick_cell, pick_process, on_put, on_iter,
to_state, from_state). The arm modules F_S, X_S, K_S, P_S, U_S, T, X4_S and X12_S differ only in the data
handed to the constructor: the shared geometry (m, sd, Q) and the prior Sigma of one object file, and kappa.

- Critic. A logistic ADV critic on the explorer's 44 expansion features (exp_feat, label = the ADV column of
  exp_meta): x_hat = Q^T ((x - m) / sd), z = b + x_hat^T v, prior v ~ N(0, kappa Sigma), free intercept b.
  MAP exactly as v84 A.6 (`ragnarok/agent_v84/consequence.py`: the loss is a SUM over the expansions seen,
  the penalty (1/(2 kappa)) v^T Sigma^-1 v, whitened x' = x_hat Sigma^(1/2), L-BFGS-B, at most 200
  iterations, warm-started from the previous refit, from 0 at the first).
- Refits. Every 100 expansions (after expansion n - 1, n = 100, 200, ...) on all expansions 0..n-1. A refit
  whose rows lack either class is skipped (intercept-only, no fit). The critic is ACTIVE from the first refit
  that sees both classes and stays active (its data only grow). Inactive: pick_cell returns V2s's own
  `draw_cell(P_{e,0})` and pick_process V2s's own `bandit.pick(P_{e,0})`, with no other draw anywhere, so a W
  arm is byte-identical to V2s until its first two-class refit (G-build test 6).
- P3 steering (active), every draw on C_e = [seed, 85, e, 2000]: a coin; if it is below 0.5, 32 candidate
  cells are drawn i.i.d. with V2's selection weights (`arch.weights(e)`, with replacement, no frontier
  mixture), every (candidate, process) pair (32 x 4) is scored, and one pair is picked with probability
  proportional to exp(1.5 clip(z, -2, 2)); otherwise V2s's own draw (cell and process from P_{e,0}). The first
  slot then comes from P_{e,0} through `propose`, as for every V2s proposal.
- Score of a pair: P(ADV) of the critic on the features that expansion would log if that pair were executed
  (vectorised over the pairs, one call per expansion): the cell's `chosen` count plus one (Explorer2s counts
  the executed cell before taking its features), billed frames = the billed count at the end of the last
  expansion (b0 at expansion 0) plus one restore (`restore_frames`, the 60-frame restore charge, data), and the
  first slot's class (moving, vertical, extra) at its expectation under that process's first-slot law (uniform
  over the adapter's slots; Q4 inherits the cell's last slot with probability 0.5 when it has one). Every
  other feature is exactly Explorer2s._features of that (cell, process).
- z: the robust z-score of the 128 scores, (u - median) / (1.4826 MAD); when MAD = 0 its clipped limit,
  2 sign(u - median).
- No RNG state: every draw is on the expansion-indexed streams P_{e,0} and C_e; to_state holds the fit, the
  activation point and the counters (checkpoints are taken at expansion boundaries).
- Identity (review fix W-2): `ident`, data from the evaluator (the object's content SHA-256, kappa.json's
  SHA-256 and kappa), is stored by to_state and must be equal in from_state: the fit lives in the whitened
  coordinates of one Sigma^(1/2) and one kappa, so a resume onto another object or kappa is refused, as
  SingleProcess refuses another q.

`SingleProcess` is the steering object of arm S (shared by families W and M; v84 §6.1 "single best process":
"V2 with only q*, no bandit"): V2s's own cell draw from P_{e,0}, then always process q* (data). The bandit's
bookkeeping in Explorer2s still runs but never chooses (its logged propensity is not S's).

Interpretations recorded (the frozen text leaves them open): the slot-class expectation above (the text scores
a (cell, process) pair, whose first slot is still random); candidates drawn with replacement, duplicates kept;
the score in probability units, as v84 §6.1's u; the MAD = 0 limit; the coin branch (steered when below 0.5,
as GF and K*map); the restore charge passed as data, not imported.
"""
import math

import numpy as np

from ragnarok.agent_v83.adapter import SLOTS
from ragnarok.agent_v84.consequence import fit, sqrt_psd
from ragnarok.agent_v84.explorer2 import N_EXP_FEAT, slot_class

N_PROC = 4                       # Q1..Q4: a W arm has no fifth process (its bandit keeps k = 4)
SC_COLS = (16, 17, 18)           # first slot's class in Explorer2s._features
REFIT_EVERY = 100
N_CAND = 32
TEMP = 1.5
CLIP = 2.0
MIX = 0.5
MAXITER = 200


def robust_z(u):
    """(u - median) / (1.4826 MAD); when MAD = 0, the clipped limit 2 sign(u - median)."""
    u = np.asarray(u, np.float64)
    med = float(np.median(u))
    s = 1.4826 * float(np.median(np.abs(u - med)))
    if s > 0:
        return (u - med) / s
    return CLIP * np.sign(u - med)


def pick_weights(u, temp=TEMP, clip=CLIP):
    """Probabilities proportional to exp(temp clip(z, -clip, clip)) of the robust z of the scores u."""
    w = np.exp(temp * np.clip(robust_z(u), -clip, clip))
    return w / w.sum()


def adapter_signature(slots):
    """[number of movement slots, whether a spawn slot exists] of the adapter's canonical slots."""
    cls = [slot_class(SLOTS[s]) for s in slots]
    return [float(sum(1 for c in cls if c[0])), float(any((not c[0]) and c[2] for c in cls))]


def pair_features(ex, cells, restore_frames, n_proc=N_PROC):
    """(len(cells), n_proc, 44) float64: the features Explorer2s would log if (cell, process) were executed at
    the current expansion, the first slot's class at its expectation (module docstring)."""
    A = ex.arch
    c = np.asarray(cells, np.int64)
    used = np.flatnonzero(A.used)
    sb_u, d_u = A.sbin[used], A.depth[used]
    sb_max, L_max = sb_u.max(), A.lvs[used].max()
    dc = A.depth[c]
    drank = (np.count_nonzero(d_u[None, :] < dc[:, None], axis=1)
             + np.count_nonzero((d_u[None, :] == dc[:, None]) & (used[None, :] < c[:, None]), axis=1))
    chosen = A.chosen[c] + 1
    par = A.parent[c]
    p0 = np.maximum(par, 0)
    par_ok = (par >= 0) & A.used[p0]
    par_chosen = A.chosen[p0] + (par == c)
    pyield = np.where(par_ok, (A.adv_children[p0] + 1) / (par_chosen + 2), 0.5)
    occ = np.count_nonzero(sb_u[None, :] == A.sbin[c][:, None], axis=1)
    nh = len(ex.hist)
    coarse = np.array([ex.coarse_count.get(A.codes[int(i)][0], 1) for i in c], np.float64)
    gap_s = (A.sbin[c] - sb_max).astype(np.float64)
    gap_l = (L_max - A.lvs[c]).astype(np.float64)
    cell = np.stack([gap_s, gap_l, np.log1p(drank), np.log1p(chosen), (A.adv_children[c] + 1) / (chosen + 2),
                     pyield, np.log1p(np.maximum(0, nh - A.added[c])), np.log1p(A.since_rise[c]),
                     A.after_loss[c].astype(np.float64), np.log1p(coarse), A.change[c], np.log1p(occ)], axis=1)
    last = ex.hist[-200:]
    adv_rate = float(np.mean([h[0] for h in last])) if last else 0.0
    loss_rate = float(np.mean([h[1] for h in last])) if last else 0.0
    new_rate = float(np.mean([h[2] for h in last])) if last else 0.0
    stagn = nh - ex.best_S_exp
    billed = (int(ex.exp_meta[-1][4]) if ex.exp_meta else int(ex.b0)) + int(restore_frames)
    ctx = [(billed - ex.b0) / ex.budget, adv_rate, loss_rate, math.log1p(stagn), float(ex.any_rise),
           math.log1p(max(0, billed - ex.best_S_billed)), new_rate]
    sc_all = np.mean([slot_class(SLOTS[s]) for s in ex.slots], axis=0)
    F = np.zeros((len(c), n_proc, N_EXP_FEAT))
    F[:, :, 0:12] = cell[:, None, :]
    F[:, :, 19:26] = ctx
    F[:, :, 26:28] = adapter_signature(ex.slots)
    ls = math.log1p(stagn)
    for q in range(n_proc):
        F[:, q, 12 + q] = 1.0
        F[:, q, 16:19] = sc_all
        F[:, q, 28 + 4 * q:32 + 4 * q] = np.stack([gap_l, np.full(len(c), ls), np.full(len(c), loss_rate), gap_s], 1)
    if n_proc > 3:                                  # Q4 inherits the cell's last slot with probability 0.5
        ls_c = A.last_slot[c]
        for k in np.flatnonzero(ls_c >= 0):
            F[k, 3, 16:19] = 0.5 * np.asarray(slot_class(SLOTS[int(ls_c[k])])) + 0.5 * sc_all
    return F


class CriticS:
    """Family W's online steering object (module docstring). m, sd: (44,) shared standardisation (sd already
    floored); Q: (44, d) shared basis; Sigma: (d, d) prior shape; kappa: prior scale; restore_frames: the
    billed charge of one restore; ident: the identity of the object and kappa (data, checked on resume)."""

    def __init__(self, m, sd, Q, Sigma, kappa, restore_frames, refit_every=REFIT_EVERY, n_cand=N_CAND,
                 temp=TEMP, clip=CLIP, mix=MIX, maxiter=MAXITER, ident=None):
        self.m = np.asarray(m, np.float64).reshape(N_EXP_FEAT)
        self.sd = np.asarray(sd, np.float64).reshape(N_EXP_FEAT)
        self.Q = np.asarray(Q, np.float64)
        self.Sigma = np.asarray(Sigma, np.float64)
        d = self.Q.shape[1]
        assert self.Q.shape == (N_EXP_FEAT, d) and self.Sigma.shape == (d, d) and np.all(self.sd > 0)
        assert float(kappa) > 0
        self.S12 = sqrt_psd(self.Sigma)
        self.kappa, self.restore_frames = float(kappa), int(restore_frames)
        self.refit_every, self.n_cand, self.temp, self.clip = int(refit_every), int(n_cand), float(temp), float(clip)
        self.mix, self.maxiter = float(mix), int(maxiter)
        self.ident = None if ident is None else tuple(ident)
        self.theta = None                            # (v', b) in whitened coordinates, after a two-class refit
        self.active, self.active_from = False, None
        self.refits = self.skipped = self.coins = self.steered = 0
        self.first_steered = None
        self.pending, self.last_coin = None, False

    # ---------------------------------------------------------------- critic
    def whiten(self, X):
        """x' = Sigma^(1/2) Q^T ((x - m) / sd), row-wise."""
        return (((np.asarray(X, np.float64) - self.m) / self.sd) @ self.Q) @ self.S12

    def logit(self, X):
        """b + x_hat^T v of raw feature rows (requires a fit)."""
        return self.whiten(X) @ self.theta[:-1] + self.theta[-1]

    def slopes(self):
        """(v, b): slopes on x_hat (v = Sigma^(1/2) v') and the intercept."""
        return self.S12 @ self.theta[:-1], float(self.theta[-1])

    def refit(self, ex, n):
        y = np.array([int(r[3]) for r in ex.exp_meta[:n]], np.int64)
        if not (n > 0 and 0 < int(y.sum()) < n):
            self.skipped += 1                         # one class only: intercept-only, nothing to fit
            return False
        X = np.asarray(ex.exp_rows[:n], np.float64).reshape(n, N_EXP_FEAT)
        self.theta = fit(self.whiten(X), y, self.kappa, self.theta, maxiter=self.maxiter)
        self.refits += 1
        if not self.active:
            self.active, self.active_from = True, int(n)
        return True

    # ---------------------------------------------------------------- steering hooks
    def pick_cell(self, ex, rng_p, rng_c, e):
        self.pending, self.last_coin = None, False
        if not self.active or not rng_c.random() < self.mix:
            return ex.draw_cell(rng_p, e)
        idx, w = ex.arch.weights(e)
        cand = idx[rng_c.choice(len(idx), size=self.n_cand, p=w / w.sum())]
        F = pair_features(ex, cand, self.restore_frames)
        u = 1.0 / (1.0 + np.exp(-self.logit(F.reshape(-1, N_EXP_FEAT))))
        k = int(rng_c.choice(len(u), p=pick_weights(u, self.temp, self.clip)))
        i, q = int(cand[k // N_PROC]), k % N_PROC
        self.pending, self.last_coin = (i, q), True
        if self.first_steered is None:
            self.first_steered = int(e)
        return i

    def pick_process(self, ex, rng_p, rng_c, i):
        p, self.pending = self.pending, None
        if p is not None and p[0] == int(i):
            return int(p[1])
        return ex.bandit.pick(rng_p)

    def on_put(self, ex, j):
        pass

    def on_iter(self, ex, n):
        self.coins += int(self.active)
        self.steered += int(self.last_coin)
        self.last_coin = False
        if n % self.refit_every == 0:
            self.refit(ex, n)

    # ---------------------------------------------------------------- captions and checkpoints
    def caption(self):
        return dict(kappa=self.kappa, d=int(self.Q.shape[1]), refit_every=self.refit_every, refits=self.refits,
                    refits_one_class=self.skipped, active=self.active, active_from=self.active_from,
                    active_expansions=self.coins, steered_picks=self.steered, first_steered=self.first_steered,
                    steered_share=self.steered / self.coins if self.coins else None,
                    intercept=None if self.theta is None else float(self.theta[-1]))

    def to_state(self):
        return dict(theta=None if self.theta is None else self.theta.copy(), active=self.active,
                    active_from=self.active_from, refits=self.refits, skipped=self.skipped, coins=self.coins,
                    steered=self.steered, first_steered=self.first_steered, pending=self.pending,
                    last_coin=self.last_coin, ident=self.ident)

    def from_state(self, d):
        saved = d.get("ident")
        assert (None if saved is None else tuple(saved)) == self.ident, \
            "the critic's object or kappa changed since the checkpoint"
        self.theta = None if d["theta"] is None else np.array(d["theta"], np.float64)
        self.active, self.active_from, self.refits, self.skipped = d["active"], d["active_from"], d["refits"], d["skipped"]
        self.coins, self.steered, self.first_steered = d["coins"], d["steered"], d["first_steered"]
        self.pending, self.last_coin = d["pending"], d["last_coin"]


class SingleProcess:
    """Arm S (module docstring): V2s's own cell draw, then always process q (no bandit choice)."""

    def __init__(self, q):
        assert 0 <= int(q) < N_PROC
        self.q = int(q)

    def pick_cell(self, ex, rng_p, rng_c, e):
        return ex.draw_cell(rng_p, e)

    def pick_process(self, ex, rng_p, rng_c, i):
        return self.q

    def on_put(self, ex, j):
        pass

    def on_iter(self, ex, n):
        pass

    def caption(self):
        return dict(q_star=self.q)

    def to_state(self):
        return dict(q=self.q)

    def from_state(self, d):
        assert int(d["q"]) == self.q
