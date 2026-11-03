"""Family M, "how to move" (docs/ARC4_V85_PROTOCOL_EN.md §4.4; Part B item 12). Agent side: the Q5 process glue
(prior libraries, online re-mining, merge), the frozen miner, the designer library DM, the P_on generator and the
X_frozen switch. The Q5 law itself is agent_v85.q5_v85.Q5 (one law for K*mac and family M: draws proportional to
count with a 10% uniform mixture, chained to the end of the rollout); this module only feeds it libraries.

Representation (§4.4; audit M8). A macro is a run-length sequence ((slot, steps), ...) of canonical slots, each
slot held 4 steps (one logged row), 8-64 steps long in multiples of 4. Priv slots map to NOOP when a macro is
built; Q5 resolves an unavailable slot to its movement class without extra, then to NOOP. A mined macro is 8 rows
= 32 steps. Its key is its 8 canonical slots in base 18 (row 0 least significant); two windows are the same macro
iff their keys are equal (the run-length form is a function of the 8 slots).

Miner (frozen in §4.4), on per-source tables. A source is one game (its two G1b runs pooled) offline, and the run
itself online. Fields read: slot, it, t, rise_lag and the ADV flag only.
- Candidates: in ADV rollouts, each row with rise_lag < 4 (a sensed rise within its 4 steps) that has 7 earlier
  rows in the same rollout gives the 8-row window that ends with it: rows t-28..t, i.e. steps t-28..t+3, with the
  rise inside the last row. This is the row analogue of K*mac's 32-step window ending at the rise (choice). Rows
  exist at t = 4, 8, ... (t >= 4), as in the v84 row log, so a row with a rise but fewer than 7 earlier rows in its
  rollout gives no candidate. One candidate per such row (several rises inside one row give one candidate).
- Baseline: one random 8-row window per non-ADV rollout; a rollout with fewer than 8 rows gives none.
- Per source g: c_adv,g = the number of candidate windows equal to the macro, N_adv,g = the number of candidate
  windows; c_base,g and N_base,g the same over the baseline windows.
- Lift = the mean over sources, each with weight 1, of [(c_adv,g + 1)/(N_adv,g + 1)] / [(c_base,g + 1)/(N_base,g
  + 1)]: the equal-game mean of the per-game ratio (reading). Every source enters, including those where the macro
  never occurs.
- Keep: in decreasing lift (ties: larger total c_adv, then smaller key), a macro is kept iff it has c_adv,g >= 3 in
  at least 2 sources, its run-length edit distance to every macro already kept is >= 3, and, if one source
  dominates it, fewer than 4 kept macros are already dominated by that source. Stop at 32.
- RLE edit distance: the edit distance of the run-length-encoded strings, i.e. the Levenshtein distance between
  the row sequences the two macros encode (one symbol per 4-step row; substitution, insertion and deletion each
  cost 1), which is the usual meaning of "edit distance of run-length-encoded strings" (reading). Two other
  readings were rejected. Levenshtein on (slot, steps) tokens puts any two macros of at most 2 runs within
  distance 2, so the library could hold at most one hold or two-run macro. Per-step strings put two macros that
  differ in a single row at distance 4, so the threshold 3 would not even exclude one-row variants.
- Dominated: one source holds strictly more than half of the macro's total c_adv (choice).
- One source only (the run's own rollouts online; K_mine, one game's own G1b rows): the two cross-game clauses
  have no object there. "c_adv >= 3 in at least 2 games" is applied as c_adv >= 3 in at least min(2, G) = 1
  source, and the per-game cap is not applied. Read literally, with G = 1 every macro is dominated by the one
  source (so at most 4 would be kept) and none has c_adv >= 3 in 2 games (so none would be kept). Online mining
  and K_mine would then always be empty, F_M would equal V2s for ever and K_mine would equal F_M, against §4.4
  ("F_M: an empty library plus mining. It is identical to V2s until 250 expansions"; "K_mine ... the same-game
  anchor") (interpretation, recorded; needs ratification).

Weights and online mining. Q5 draws in proportion to count. Every prior library (carried X, K_mine, DM, P, X4,
X12) enters a run with the pseudo-count n0 = 10 per macro, so a prior alone is drawn uniformly. Every 250
expansions (after expansions 250, 500, ... have been executed, in the `after` hook of the 250th, so the library
changes at the boundary of expansion 250, 500, ...), the run re-mines all its own executed rollouts so far as one
source. The merged library is the sum of two count tables: the prior {macro: 10} and the re-mine's kept macros
{macro: the run's own c_adv}. A macro in both gets 10 + c_adv. Prior macros come first, in their order, then the
newly kept ones in lift order (reading of "prior pseudo-counts n0 = 10 merge with the run's own counts"). With an
empty prior (F_M) the library stays empty, and the bandit keeps k = 4, until a re-mine keeps a macro. The X_frozen
switch (mining=False) keeps the prior library: no log, no re-mine. With the same prior, X_frozen and X_on are
identical until expansion 250.

Online rows ("every 4th executed slot"). The run's own rows are rebuilt from each executed rollout. The slot at
t = 4, 8, ... (t < steps, as the v84 row log) is the slot of the ALE action played (adapter.to_effect, exact for
every slot the explorer plays), with priv mapped to NOOP. rise_lag < 4 iff a sensed rise occurs in steps t..t+3,
read from the run's own senses by a pass-through of the env (SensedSteps, installed by Explorer2m.run). The ADV
flag is the rollout's own. explorer2s.py is not modified, so the closure of V2s and of the instruments is
unchanged. The baseline window of non-ADV rollout e is drawn from a new stream M_e = [seed, 85, e, 3000] (choice;
the A.6 streams are untouched, so F_M is byte-identical to V2s for its first 250 expansions). The mining state
(counts, kept keys, number of re-mines) is part of Q5's to_state, which is checkpointed with the explorer
(§13.3). The per-rollout rise buffer is empty at every expansion boundary.

DM, the designer library (frozen in §4.4), 33 macros (build note 6). The 8 directions are the movement classes (sx, sy) != (0, 0)
(screen axes: sy = +1 is down), in the CANON order. "Extra" is the slot "0,0,1"; "jump" is the run's direction
with extra; NOOP is "0,0,0".
- holds of 8 and of 32 steps in the 8 directions (16);
- 16-step holds with extra in the 8 directions (8);
- extra 4 on, 4 off (NOOP), for 32 steps (1);
- alternations of 8: left 8 then right 8, up 8 then down 8 (2; "of 8" read as 8-step segments, one cycle);
- two diagonal zigzags: up-right 8 then down-right 8, up-left 8 then down-left 8 (2; a zigzag alternates between
  two diagonals, with 8-step segments like the alternations);
- NOOP for 16 steps (1); extra held for 32 steps (1);
- right-run-then-jump and left-run-then-jump (8 + 8): two 16-step macros, run 8 then the same direction with
  extra 8 (2). §4.4 states 32 macros but its list makes 33; build note 6 (3 November 2026, before any v85 data) keeps
  every listed macro unchanged and treats the stated total as an arithmetic slip.

P_on generator. For each carried macro, in order, one window is drawn in proportion to its window count among the
non-ADV windows whose 8-row slot multiset equals the macro's: same length (32 steps) and same slot frequency
(reading of "matched on length and slot frequency"). Sequences already drawn are excluded. If no window is left
with that multiset, the draw is among the windows at the smallest L1 distance between slot-count vectors. Draws
use inverse CDFs on uniforms supplied by the caller (the evaluator draws them from [9803, 1]). A placebo macro may
coincide with a carried one: a pure hold has only one sequence with its multiset.
"""
import numpy as np

from ragnarok.agent_v83.adapter import CANON, SLOTS
from ragnarok.agent_v85.explorer2s import Explorer2s
from ragnarok.agent_v85.q5_v85 import NOOP, Q5

HOLD = 4                    # steps per logged row; every macro slot is held 4 steps
ROWS = 8                    # rows per mined window: 32 steps
N_LIB = 32                  # library size (top 32)
MIN_C = 3                   # c_adv >= 3 ...
MIN_SOURCES = 2             # ... in at least 2 sources (games)
MIN_EDIT = 3                # pairwise run-length edit distance >= 3
CAP_DOM = 4                 # at most 4 kept macros dominated by one source
N0 = 10                     # prior pseudo-count per macro
EVERY = 250                 # online re-mining period (expansions)
MINER_STREAM = 3000         # M_e = [seed, 85, e, 3000]: the online baseline window of non-ADV rollout e
BASE = len(CANON)           # 18 canonical slots; priv slots (>= 18) map to NOOP
POW = BASE ** np.arange(ROWS, dtype=np.int64)
HPOW = (ROWS + 1) ** np.arange(BASE, dtype=np.int64)        # slot-count vector -> class key (9^17 < 2^63)


# ---------------------------------------------------------------- representation
def canon(s):
    """Canonical slot: priv (or anything outside CANON) -> NOOP. Works on ints and arrays."""
    a = np.asarray(s, np.int64)
    out = np.where((a >= 0) & (a < BASE), a, NOOP)
    return int(out) if out.ndim == 0 else out


def encode(rows):
    """(n, 8) canonical slots -> (n,) int64 keys (row 0 least significant)."""
    return np.asarray(rows, np.int64).reshape(-1, ROWS) @ POW


def decode(keys):
    """(n,) int64 keys -> (n, 8) canonical slots."""
    k = np.asarray(keys, np.int64).reshape(-1, 1)
    return (k // POW) % BASE


def rle(slots, hold=HOLD):
    """Run-length macro ((slot, steps), ...) of a sequence of slots, each held `hold` steps."""
    out = []
    for s in slots:
        s = int(s)
        if out and out[-1][0] == s:
            out[-1][1] += hold
        else:
            out.append([s, hold])
    return tuple((s, n) for s, n in out)


def macro_of(key):
    return rle(decode([key])[0])


def unroll(macro, hold=HOLD):
    """The per-row slots of a macro (each run of n steps is n / hold rows)."""
    out = []
    for s, n in macro:
        assert n % hold == 0, macro
        out += [int(s)] * (int(n) // hold)
    return out


def steps(macro):
    return int(sum(n for _, n in macro))


def edit_distance(a, b):
    """RLE edit distance: the Levenshtein distance between the row sequences that two run-length macros encode
    (each run of n steps = n / 4 rows of its slot; unit costs)."""
    a, b = unroll(a), unroll(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def slot_counts(rows):
    """(n, 8) canonical slots -> (n, 18) slot-count vectors."""
    rows = np.asarray(rows, np.int64)
    return np.stack([(rows == s).sum(1) for s in range(BASE)], 1)


def lookup(keys, counts, query):
    """Counts of `query` keys in a sorted unique key table (0 when absent)."""
    keys, counts, query = np.asarray(keys, np.int64), np.asarray(counts, np.int64), np.asarray(query, np.int64)
    if not len(keys):
        return np.zeros(len(query), np.int64)
    i = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return np.where(keys[i] == query, counts[i], 0)


def table(keys):
    """Sorted unique keys and counts of a list of keys."""
    k, c = np.unique(np.asarray(keys, np.int64), return_counts=True)
    return k.astype(np.int64), c.astype(np.int64)


def source(adv_keys, base_keys):
    """A miner source from its candidate keys and baseline keys (with repetitions)."""
    ak, ac = table(adv_keys)
    bk, bc = table(base_keys)
    return dict(adv_k=ak, adv_c=ac, n_adv=int(len(adv_keys)), base_k=bk, base_c=bc, n_base=int(len(base_keys)))


# ---------------------------------------------------------------- the frozen miner
def mine(sources, n_lib=N_LIB):
    """The §4.4 miner on per-source tables (module docstring).

    sources: [dict(adv_k, adv_c, n_adv, base_k, base_c, n_base)], keys sorted unique int64.
    Returns (keys, rows, summary): the kept keys in selection order, one row per kept macro (macro, lift, c_adv,
    c_adv per source, sources with c_adv >= 3, dominant source index or None) and a summary."""
    G = len(sources)
    need = min(MIN_SOURCES, G)
    summary = dict(sources=G, need_sources=need, cap_applied=G >= MIN_SOURCES, eligible=0,
                   n_adv=[int(s["n_adv"]) for s in sources], n_base=[int(s["n_base"]) for s in sources])
    if not G:
        return [], [], summary
    hits = np.concatenate([np.asarray(s["adv_k"], np.int64)[np.asarray(s["adv_c"]) >= MIN_C] for s in sources])
    uk, nk = np.unique(hits, return_counts=True)
    keys = uk[nk >= need].astype(np.int64)
    summary["eligible"] = int(len(keys))
    if not len(keys):
        return [], [], summary
    C = np.stack([lookup(s["adv_k"], s["adv_c"], keys) for s in sources]).astype(np.float64)
    B = np.stack([lookup(s["base_k"], s["base_c"], keys) for s in sources]).astype(np.float64)
    na = np.array([float(s["n_adv"]) for s in sources])[:, None]
    nb = np.array([float(s["n_base"]) for s in sources])[:, None]
    lift = (((C + 1.0) / (na + 1.0)) / ((B + 1.0) / (nb + 1.0))).mean(0)
    tot = C.sum(0)
    share = C.max(0) / tot
    dom = np.where(share > 0.5, C.argmax(0), -1)
    order = np.lexsort((keys, -tot, -lift))
    kept, kept_m, rows, per = [], [], [], {}
    for j in order:
        m = macro_of(keys[j])
        if any(edit_distance(m, o) < MIN_EDIT for o in kept_m):
            continue
        g = int(dom[j])
        if G >= MIN_SOURCES and g >= 0 and per.get(g, 0) >= CAP_DOM:
            continue
        if g >= 0:
            per[g] = per.get(g, 0) + 1
        kept.append(int(keys[j]))
        kept_m.append(m)
        rows.append(dict(key=int(keys[j]), macro=[list(t) for t in m], lift=float(lift[j]), c_adv=int(tot[j]),
                         c_adv_by_source=[int(v) for v in C[:, j]], c_base_by_source=[int(v) for v in B[:, j]],
                         sources_ge3=int((C[:, j] >= MIN_C).sum()), dominant=None if g < 0 else g))
        if len(kept) == n_lib:
            break
    summary["kept"] = len(kept)
    summary["dominated_per_source"] = {str(k): v for k, v in sorted(per.items())}
    return kept, rows, summary


# ---------------------------------------------------------------- the designer library DM (frozen in §4.4)
def _slot(sx, sy, ex=0):
    return SLOTS.index(f"{sx},{sy},{ex}")


DIRS = [(sx, sy) for sx in (0, 1, -1) for sy in (0, 1, -1) if (sx, sy) != (0, 0)]


def designer_library():
    """The 33 macros of DM, in the §4.4 order (module docstring; build note 6)."""
    E, N = _slot(0, 0, 1), NOOP
    R, L, D, U = _slot(1, 0), _slot(-1, 0), _slot(0, 1), _slot(0, -1)
    lib = [((_slot(sx, sy), n),) for n in (8, 32) for sx, sy in DIRS]                 # holds of 8 and 32 (16)
    lib += [((_slot(sx, sy, 1), 16),) for sx, sy in DIRS]                               # 16 with extra (8)
    lib.append(((E, 4), (N, 4)) * 4)                                                    # burst, 32 steps
    lib += [((L, 8), (R, 8)), ((U, 8), (D, 8))]                                         # alternations of 8
    lib += [((_slot(1, -1), 8), (_slot(1, 1), 8)), ((_slot(-1, -1), 8), (_slot(-1, 1), 8))]   # zigzags
    lib += [((N, 16),), ((E, 32),)]                                                     # NOOP 16, extra 32
    lib += [((R, 8), (_slot(1, 0, 1), 8)), ((L, 8), (_slot(-1, 0, 1), 8))]             # run-then-jump, 8 + 8
    return lib


DM = designer_library()


# ---------------------------------------------------------------- the P_on generator
def placebo_library(targets, pool_k, pool_c, u):
    """P: for each target (carried) macro, one non-ADV window matched on length and slot multiset (module
    docstring). pool_k / pool_c: the distinct non-ADV 8-row window keys (sorted) and their window counts; u: one
    uniform in [0, 1) per target. Returns (macros, rows)."""
    pool_k, pool_c = np.asarray(pool_k, np.int64), np.asarray(pool_c, np.float64)
    hk = np.empty(len(pool_k), np.int64)
    for i in range(0, len(pool_k), 1 << 18):                    # slot-multiset class of every pooled sequence
        hk[i:i + (1 << 18)] = slot_counts(decode(pool_k[i:i + (1 << 18)])) @ HPOW
    cls, inv = np.unique(hk, return_inverse=True)
    inv = inv.reshape(-1)
    Hc = (cls.reshape(-1, 1) // HPOW) % (ROWS + 1)              # (classes, 18) slot counts
    free = np.ones(len(pool_k), bool)
    out, rows = [], []
    big = np.iinfo(np.int64).max
    for m, ui in zip(targets, u):
        m = tuple((int(s), int(n)) for s, n in m)
        assert steps(m) == ROWS * HOLD, ("P matches 8-row (32-step) macros", m)
        h = slot_counts([[canon(s) for s in unroll(m)]])[0]
        left = np.bincount(inv[free], minlength=len(cls))
        d = np.where(left > 0, np.abs(Hc - h).sum(1), big)
        if not len(d) or d.min() == big:
            break
        idx = np.flatnonzero(free & (d == d.min())[inv])
        cdf = np.cumsum(pool_c[idx]) / pool_c[idx].sum()
        j = int(idx[min(int(np.searchsorted(cdf, float(ui), side="right")), len(idx) - 1)])
        free[j] = False
        p = macro_of(pool_k[j])
        out.append(p)
        rows.append(dict(target=[list(t) for t in m], macro=[list(t) for t in p], key=int(pool_k[j]),
                         window_count=int(pool_c[j]), distance=int(d.min()), pool_windows=int(pool_c[idx].sum()),
                         pool_sequences=int(len(idx))))
    return out, rows


# ---------------------------------------------------------------- Q5 glue: prior, online re-mining, X_frozen
def prior_library(macros, n0=N0):
    """A prior library for Q5: every macro with the pseudo-count n0."""
    return [(tuple((int(s), int(n)) for s, n in m), float(n0)) for m in macros]


class SensedSteps:
    """Pass-through of the run's env that keeps the sensed rise (r > 0) of every step since the last take()."""

    def __init__(self, env):
        self.env = env
        self.rises = bytearray()

    @property
    def billed(self):
        return self.env.billed

    def step(self, a):
        out = self.env.step(a)
        self.rises.append(1 if out[1] > 0 else 0)
        return out

    def restore(self, h):
        return self.env.restore(h)

    def clone(self):
        return self.env.clone()

    def free(self, h):
        return self.env.free(h)

    def take(self):
        r = bytes(self.rises)
        self.rises = bytearray()
        return r


class Explorer2m(Explorer2s):
    """Explorer2s run on SensedSteps(env), so that the `after` hook can read each executed rollout's sensed rises
    (family M's online miner). Same law and same draws as Explorer2s."""

    def run(self, env, obs=None, lives=None, S1=None, spent=Exception, resume=False):
        return super().run(SensedSteps(env), obs, lives, S1, spent=spent, resume=resume)


def rollout_rows(slots_played, rises):
    """Rows of one executed rollout: (canonical slot at t = 4, 8, ... < steps, rise within steps t..t+3)."""
    n = len(slots_played)
    ts = range(HOLD, n, HOLD)
    return [canon(slots_played[t]) for t in ts], [bool(any(rises[t:t + HOLD])) for t in ts]


class MinedQ5(Q5):
    """Q5 over a prior library (pseudo-count n0 per macro) with online re-mining (mining=True) or the X_frozen switch
    (mining=False). `after` is Explorer2m's after-rollout hook."""

    def __init__(self, prior=(), mining=True, every=None, n0=N0):
        self.prior = [tuple((int(s), int(n)) for s, n in m) for m in prior]
        self.mining, self.n0 = bool(mining), float(n0)
        self.every = EVERY if every is None else int(every)
        self.c_adv, self.c_base = {}, {}
        self.n_adv = self.n_base = 0
        self.kept, self.mines, self.seen = [], 0, 0
        super().__init__(prior_library(self.prior, self.n0))

    # ------------------------------------------------------------ the run's own rows
    def observe(self, e, adv, slots_played, rises, seed):
        """Log one executed rollout (expansion e): candidates if ADV, else one baseline window from M_e."""
        rows, flags = rollout_rows(slots_played, rises)
        self.seen += 1
        if adv:
            for k in range(ROWS - 1, len(rows)):
                if flags[k]:
                    key = int(encode(rows[k - ROWS + 1:k + 1])[0])
                    self.c_adv[key] = self.c_adv.get(key, 0) + 1
                    self.n_adv += 1
        elif len(rows) >= ROWS:
            rng = np.random.default_rng([int(seed), 85, int(e), MINER_STREAM])
            s0 = int(rng.integers(len(rows) - ROWS + 1))
            key = int(encode(rows[s0:s0 + ROWS])[0])
            self.c_base[key] = self.c_base.get(key, 0) + 1
            self.n_base += 1

    def own_source(self):
        ak = np.array(sorted(self.c_adv), np.int64)
        bk = np.array(sorted(self.c_base), np.int64)
        return dict(adv_k=ak, adv_c=np.array([self.c_adv[k] for k in ak.tolist()], np.int64), n_adv=self.n_adv,
                    base_k=bk, base_c=np.array([self.c_base[k] for k in bk.tolist()], np.int64), n_base=self.n_base)

    def remine(self):
        """Re-mine the run's own rollouts (one source) and merge with the prior pseudo-counts."""
        kept, _, _ = mine([self.own_source()])
        self.kept, self.mines = kept, self.mines + 1
        merged = {}
        for m in self.prior:
            merged[m] = merged.get(m, 0.0) + self.n0
        for key in kept:
            m = macro_of(key)
            merged[m] = merged.get(m, 0.0) + float(self.c_adv[key])
        if merged:
            self.set_library(list(merged.items()))
        else:
            assert not len(self), "a non-empty library never becomes empty"

    def after(self, ex, env, e, out):
        rises = env.take()
        if not self.mining:
            return
        n = int(out["steps"])
        played = out["acts"]
        assert len(rises) == n == len(played), (len(rises), n, len(played))
        ad = ex.body.adapter
        self.observe(e, bool(out["adv"]), [ad.to_effect(a) for a in played], rises, ex.seed)
        if (e + 1) % self.every == 0:
            self.remine()

    # ------------------------------------------------------------ captions and checkpoints (§13.3)
    def caption(self):
        return dict(mining=self.mining, prior_size=len(self.prior), library_size=len(self), mines=self.mines,
                    kept_last=len(self.kept), rollouts_logged=self.seen, n_adv=self.n_adv, n_base=self.n_base,
                    distinct_adv=len(self.c_adv), every=self.every, n0=self.n0)

    def to_state(self):
        return dict(lib=list(self.lib), prior=list(self.prior), mining=self.mining, every=self.every, n0=self.n0,
                    c_adv=dict(self.c_adv), c_base=dict(self.c_base), n_adv=self.n_adv, n_base=self.n_base,
                    kept=list(self.kept), mines=self.mines, seen=self.seen)

    def from_state(self, d):
        self.prior = [tuple((int(s), int(n)) for s, n in m) for m in d["prior"]]
        self.mining, self.every, self.n0 = bool(d["mining"]), int(d["every"]), float(d["n0"])
        self.c_adv, self.c_base = dict(d["c_adv"]), dict(d["c_base"])
        self.n_adv, self.n_base = int(d["n_adv"]), int(d["n_base"])
        self.kept, self.mines, self.seen = list(d["kept"]), int(d["mines"]), int(d["seen"])
        self.set_library(d["lib"])


def family_hooks(prior=(), mining=True):
    """The arm hooks of a family-M arm: MinedQ5 as process 5, Explorer2m, and the after-rollout hook."""
    q5 = MinedQ5(prior, mining=mining)
    return dict(q5=q5, explorer=Explorer2m, kwargs=dict(after=q5.after))
