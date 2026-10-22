"""Promise heads and steering (docs/ARC4_V83_PROTOCOL.md §1 'Promesses'; synthesis §2.2), arms F, X, P,
X0 and K*; absent in V.

Two logistic heads over the view features (views.features): GOOD = a sensed rise within 24 steps, BAD
= a lives decrease within 40 steps; rows cut off before their horizon and rows where I am lost are
dropped. Intercepts are free per game. Refit every 250 iterations on a reservoir of at most 120k rows;
loss = mean log-loss + (n0/(n0+n)) (kappa0/2) ||w - w0||^2, n0 = 50,000, kappa0 = 1, w0 the carried
weights (0 for F). The carried object W = both heads' weights + the standardisation (mean, scale); a
fresh arm recomputes its standardisation from its reservoir at each refit; every scale >= SD_FLOOR.
A fresh head without both classes in the reservoir is intercept-only; a carried head keeps its weights
and steers (BAD is intercept-only while no lives decrease was sensed: synthesis §2.2). A frozen arm
(X0) never refits its weights (intercepts only). At a rollout start the button is scored on the
restored cell's archived view (it has the motion channels).

Steering: (a) draw 16 candidate cells with V's weights, pick one with probability proportional to
exp(beta clip(z, -2, 2)), z the robust z-score of u(c) = max over slots of [P(GOOD) - lambda P(BAD)];
(b) at a rollout start and each action change, button b drawn from (1 - alpha) uniform + alpha
softmax_b(logit GOOD(slot b) - lambda logit BAD(slot b)). With w = 0 both are exactly V's laws.
"""
import numpy as np
from scipy.optimize import minimize

from ragnarok.agent_v83.adapter import SLOTS
from ragnarok.agent_v83.logs import NONE, labels
from ragnarok.agent_v83.views import N_FEAT, features, fg_mask, log_view

HORIZON = dict(good=24, bad=40)
HEADS = ("good", "bad")
SD_FLOOR = 0.05            # declared: no feature is scaled by less than this (all arms, carried W included)


def slot_traits(slot):
    """(move (sx, sy), extra bit, private) of a canonical slot index."""
    name = SLOTS[slot]
    if name.startswith("priv"):
        return (0, 0), False, True
    sx, sy, ex = (int(v) for v in name.split(","))
    return (sx, sy), bool(ex), False


def feats(packed, screen, lost, me, lives, slot, shape):
    move, extra, private = slot_traits(slot)
    return features(packed, screen, lost, me, lives, move, extra, private, shape)


def fit_head(X, y, w0, n0=50000, kappa0=1.0, b0=0.0, weights=None, maxiter=200, offset=None):
    """Logistic head with L2-SP pull to w0 (strength n0/(n0+n) kappa0/2); intercept unpenalised;
    `offset` is a fixed per-row logit added (frozen weights)."""
    n, d = X.shape
    off = np.zeros(n) if offset is None else offset
    lam = (n0 / (n0 + n)) * kappa0 / 2.0
    sw = np.full(n, 1.0 / n) if weights is None else weights / weights.sum()
    y = y.astype(np.float64)

    dt = X.dtype if X.dtype == np.float32 else np.float64

    def f(theta):
        w, b = theta[:d], theta[d]
        z = (X @ w.astype(dt)).astype(np.float64) + b + off
        lz = np.logaddexp(0.0, z)
        loss = float(sw @ (lz - y * z)) + lam * float((w - w0) @ (w - w0))
        p = 1.0 / (1.0 + np.exp(-z))
        g = (sw * (p - y)).astype(np.float64)
        gw = (X.T @ g.astype(dt)).astype(np.float64) + 2.0 * lam * (w - w0)
        return loss, np.concatenate([gw, [g.sum()]])

    th0 = np.concatenate([w0, [b0]]).astype(np.float64)
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return r.x[:d], float(r.x[d])


class Prior:
    """The carried object: weights of both heads and the standardisation. Zero weights = no prior."""

    def __init__(self, w=None, mean=None, scale=None, heads=HEADS):
        self.w = {h: (np.zeros(N_FEAT) if w is None or h not in w else np.asarray(w[h], np.float64)) for h in HEADS}
        self.mean = None if mean is None else np.asarray(mean, np.float64)
        self.scale = None if scale is None else np.asarray(scale, np.float64)
        self.heads = tuple(heads)              # heads carried (G-pred: only passing heads); others w0 = 0
        for h in HEADS:
            if h not in self.heads:
                self.w[h] = np.zeros(N_FEAT)


class Promise:
    def __init__(self, prior=None, frozen=False, beta=1.0, alpha=0.5, lam=1.0, n0=50000, kappa0=1.0,
                 refit_every=250, reservoir=120000, n_cand=16, seed=0):
        self.prior = prior or Prior()
        self.frozen = frozen
        self.beta, self.alpha, self.lam, self.n0, self.kappa0 = beta, alpha, lam, n0, kappa0
        self.refit_every, self.cap, self.n_cand = refit_every, reservoir, n_cand
        self.w = {h: self.prior.w[h].copy() for h in HEADS}
        self.b = {h: 0.0 for h in HEADS}
        self.mean, self.scale = self.prior.mean, self.prior.scale
        # a carried head steers from the start (intercept 0 until the first refit); a fresh head once fitted
        self.carried = {h: bool(np.any(self.w[h] != 0)) for h in HEADS}
        self.active = dict(self.carried)
        self.drop_seen = False                        # a lives decrease sensed after some logged row
        self._cur = None                              # the cell of the current rollout (pick_cell)
        self._ucache = {}                             # cell slot -> (refits, u)
        self.X = np.zeros((reservoir, N_FEAT), np.float32)
        self.lag = {h: np.zeros(reservoir, np.uint8) for h in HEADS}
        self.avail = np.zeros(reservoir, np.uint8)
        self.t = np.zeros(reservoir, np.uint8)
        self.n = 0                                    # rows held
        self.seen = 0                                 # rows offered (reservoir sampling)
        self.rng = np.random.default_rng(seed)
        self.refits = 0
        self.cell_view = {}
        self.shape = None

    # ---- data ----------------------------------------------------------------------------------
    def add_rows(self, rows):
        """rows: RowLog.get dicts (view, screen, lost, mx, my, lives, slot, rise_lag, drop_lag, avail, t)."""
        for r in rows:
            if r["drop_lag"] != NONE:
                self.drop_seen = True
            if r["lost"]:
                continue
            me = (float(r["mx"]), float(r["my"]))
            x = feats(r["view"], r["screen"], False, me, int(r["lives"]), int(r["slot"]), self.shape)
            self.seen += 1
            if self.n < self.cap:
                j = self.n
                self.n += 1
            else:
                j = int(self.rng.integers(self.seen))
                if j >= self.cap:
                    continue
            self.X[j] = x
            self.lag["good"][j], self.lag["bad"][j], self.avail[j] = r["rise_lag"], r["drop_lag"], r["avail"]
            self.t[j] = r["t"]

    def refit(self):
        if self.n == 0:
            return
        Z = self.X[: self.n].copy()                   # float32 copy, standardised in place
        if self.prior.mean is None:                   # fresh arm: its own standardisation, recomputed at each refit
            self.mean = Z.mean(0, dtype=np.float64)
            self.scale = np.maximum(Z.std(0, dtype=np.float64), SD_FLOOR)
        Z -= self.mean.astype(np.float32)
        Z /= self.scale.astype(np.float32)
        for h in HEADS:
            y = labels(self.lag[h][: self.n], self.avail[: self.n], self.t[: self.n], HORIZON[h], stops_on_event=(h == "bad"))
            ok = y >= 0
            if not ok.any() or not (y[ok] == 1).any() or (y[ok] == 1).all():
                if self.carried[h] and (h == "good" or self.drop_seen):
                    self.active[h] = True       # a carried head keeps its w (w0 or last fit), b and steering
                    continue
                self.active[h] = False          # intercept-only: a fresh head, or BAD while no lives decrease was sensed
                continue
            if self.frozen:
                w = self.prior.w[h]
                _, b = fit_head(Z[ok][:, :0], y[ok], np.zeros(0), n0=0.0, b0=self.b[h], offset=Z[ok] @ w)
                self.w[h], self.b[h] = w, b
            else:
                self.w[h], self.b[h] = fit_head(Z[ok], y[ok], self.prior.w[h], self.n0, self.kappa0, self.b[h])
            self.active[h] = True
        self.refits += 1
        self._ucache = {}

    # ---- scores --------------------------------------------------------------------------------
    def _logits(self, F):
        Z = (F - self.mean) / self.scale if self.mean is not None else F
        out = {}
        for h in HEADS:
            out[h] = Z @ self.w[h] + self.b[h] if self.active[h] else np.zeros(len(F))
        return out

    def slot_scores(self, packed, screen, lost, me, lives, slots):
        F = np.stack([feats(packed, screen, lost, me, lives, s, self.shape) for s in slots])
        lg = self._logits(F)
        pg, pb = 1 / (1 + np.exp(-lg["good"])), 1 / (1 + np.exp(-lg["bad"]))
        return lg["good"] - self.lam * lg["bad"], pg - self.lam * pb

    # ---- steering (explorer hooks) --------------------------------------------------------------
    def pick_cell(self, ex, rng):
        arch = ex.arch
        cand = arch.draw(rng, self.n_cand)
        if not any(self.active.values()):
            i = int(cand[int(rng.integers(len(cand)))])
        else:
            slots = sorted(ex.body.adapter.buttons)
            u = np.array([self._cell_u(int(c), slots) for c in cand])
            med = np.median(u)
            mad = 1.4826 * np.median(np.abs(u - med))
            z = np.zeros_like(u) if mad <= 1e-12 else (u - med) / mad
            p = np.exp(self.beta * np.clip(z, -2, 2))
            i = int(cand[rng.choice(len(cand), p=p / p.sum())])
        arch.chosen[i] += 1
        self._cur = i
        return i

    def _cell_u(self, i, slots):
        c = self._ucache.get(i)
        if c is not None and c[0] == self.refits:
            return c[1]
        v = self.cell_view.get(i)
        if v is None:
            return 0.0
        _, util = self.slot_scores(*v, slots)
        self._ucache[i] = (self.refits, float(util.max()))
        return self._ucache[i][1]

    def on_put(self, ex, i, ctx):
        self.shape = ctx["obs"].shape[:2]
        self.cell_view[i] = self._view(ex.body, ctx)
        self._ucache.pop(i, None)

    def on_iter(self, ex, it):
        self.add_rows([ex.rows.get(r) for r in ex.last_rows])
        if it % self.refit_every == 0:
            self.refit()

    def _view(self, body, ctx):
        prev_fg = None if ctx["prev_obs"] is None else fg_mask(ctx["prev_obs"], body.bg)
        packed, screen, lost, _ = log_view(ctx["obs"], prev_fg, body.bg, ctx["me"], body.my_size,
                                           lost=ctx["lost"], prev_me=ctx["prev_me"])
        return packed, screen, lost, ctx["me"], ctx["lives"]

    def button_probs(self, ex, n_actions, ctx):
        if not any(self.active.values()):
            return np.full(n_actions, 1.0 / n_actions)
        ad = ex.body.adapter
        self.shape = ctx["obs"].shape[:2]
        slots = sorted(ad.buttons)
        # rollout start (no previous frame): the restored cell's archived view, built with its own previous frame
        v = self.cell_view.get(self._cur) if ctx["prev_obs"] is None else None
        logit, _ = self.slot_scores(*(v if v is not None else self._view(ex.body, ctx)), slots)
        by_slot = dict(zip(slots, logit))
        lb = np.array([by_slot[ad.to_effect(b)] for b in range(n_actions)])
        sm = np.exp(lb - lb.max())
        p = (1 - self.alpha) / n_actions + self.alpha * sm / sm.sum()
        return p / p.sum()

    def pick_button(self, ex, rng, n_actions, ctx):
        if not any(self.active.values()):
            return int(rng.integers(n_actions))
        return int(rng.choice(n_actions, p=self.button_probs(ex, n_actions, ctx)))
