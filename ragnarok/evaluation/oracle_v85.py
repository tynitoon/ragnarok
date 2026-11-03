"""v85 evaluator instruments (docs/ARC4_V85_PROTOCOL_EN.md §4.3, §7.2, A.7). Development only; never agents.

- O_m (gate row O_4): V2s's own proposals j = 0..m-1, each played once in the unbilled sandbox on X_{e,j};
  the lexicographic best (ties to the lowest j) is executed for real on a fresh stream X_e. `after`
  records the executed outcome, scored like a look-ahead, for the rank correlation with the winner's
  look-ahead score; `caption` gives O-TIES (share of all-tied expansions, flagged above 70%) and that
  correlation (Spearman on the lexicographic ranks of the score tuples).
- O_4-dice (caption row, ODice): as O_4, but the winner is re-executed on X_{e,w}; its ALE actions,
  emulated frames (after the restore) and max S must equal its look-ahead's, else DiceMismatch aborts the run.
- D (plumbing control, `d_script`): the explorer's first rollout plays the start's own lineage actions
  (ALE indices) open-loop from s1 with ordinary puts, up to the first step with cum >= M_{k_ref}; the
  evaluator stop then ends the run (Explorer2s(script=...), agent seed E_s).
- K*map (gate row): steering toward cells matching the start's own future lineage (KMap, lineage_map).
- K*mac (gate row, `kmac_library`): from the OTHER start's lineage above its s1, the 32-step windows ending
  at each sensed rise (a rise at lineage step t gives steps t-31..t; a rise with t < 31 has no full window
  above s1 and is skipped: choice); the actions at window steps 0, 4, ..., 28 mapped to canonical slots
  through THIS start's cached adapter (`slot_of`, the inverse of `to_button`), each held 4 steps; unmapped
  actions and priv slots -> NOOP; run-length encoded; unique sequences weighted by count. Played by
  agent_v85.q5_v85.Q5 (§4.4) with in-rollout draws from X_e. Reported: round-trip fidelity (`fidelity`:
  share of sampled actions a with to_button(slot(a)) == a; also `fidelity_held`, the share of all window
  steps whose held button equals the lineage action, and `fidelity_slot`, the share whose slot survives),
  the share of priv- or NOOP-mapped sampled steps, the library size.
- K*mac-exact (caption row): the start's own lineage, the same windows, every step's slot held 1 step
  (priv slots of this start's own adapter are kept: they are playable and exact; choice).
- Placebos (G1-P, §7.2): GF4 (among the same 4 proposals, the one whose cell has the highest S, then more
  lives, then shallower, then lowest j; no look-ahead; executed on X_e); GF, K*map's placebo (§4.4: with
  probability 0.5, a C_e coin, the cell with the largest S, ties to the deepest, then uniform on C_e;
  otherwise V2s's own draw; process from the bandit on P); K~mac (another unit's cross-start library:
  partners from a derangement drawn with default_rng([9801, 2]), permutations of the given unit list
  redrawn until no unit is fixed; the partner's library at the same start, else at its other start, else
  empty: choice).
"""
import hashlib
import json
import pickle

import numpy as np
from scipy.stats import spearmanr

from ragnarok.agent_v83.adapter import SLOTS
from ragnarok.agent_v83.explorer import s_bin
from ragnarok.agent_v84.framecode import codes
from ragnarok.agent_v85.q5_v85 import NOOP, Q5   # noqa: F401  (Q5 plays K*mac, K*mac-exact and K~mac)
from ragnarok.evaluation.senses_v85 import SenseEnv85

WINDOW, HOLD = 32, 4
OTHER = {"A": "B", "B": "A"}


def _lex_rank_corr(pairs):
    """Spearman correlation of paired score tuples, each side ranked lexicographically; None if undefined."""
    if len(pairs) < 3:
        return None
    cols = []
    for k in (0, 1):
        order = {v: r for r, v in enumerate(sorted({p[k] for p in pairs}))}
        cols.append([order[p[k]] for p in pairs])
    if len(set(cols[0])) < 2 or len(set(cols[1])) < 2:
        return None
    return float(spearmanr(cols[0], cols[1])[0])


class DiceMismatch(RuntimeError):
    """O_4-dice: the re-executed winner differs from its look-ahead; the run is aborted (counted)."""


class OChooser:
    """O_m: candidates j = 0..m-1 from propose(e, j), each played in env.sandbox() on X_{e,j}; score
    (max(0, max S - best_S), ADV against archive + overlay, keys new to archive + overlay), compared
    lexicographically, ties to the lowest j. The explorer executes the winner on X_e. Wire as
    Explorer2s(chooser=o, after=o.after) to get the rank correlation (the choice does not depend on it)."""

    def __init__(self, m=4):
        self.m = m
        self.log = []                     # per expansion: (e, winner, scores, all tied)
        self.outcomes = []                # per executed rollout: (e, its outcome scored like a look-ahead)
        self._b0 = None

    @staticmethod
    def _score(b0, out):
        return (max(0.0, out["max_S"] - b0), int(out["adv"]), int(out["new_cells"]))

    def _look(self, ex, env, e):
        saved = pickle.dumps(ex.body.tracker)
        best, scores, outs = None, [], []
        for j in range(self.m):
            p = ex.propose(e, j)
            try:
                with env.sandbox():
                    out = ex._rollout(env, *p, ex.stream(e, j, 1), e, dry=True)
            finally:
                ex.body.tracker = pickle.loads(saved)
            s = self._score(ex.best_S, out)
            scores.append(s)
            outs.append(out)
            if best is None or s > best[0]:
                best = (s, j, p)
        self.log.append((e, best[1], scores, len(set(scores)) == 1))
        self._b0 = ex.best_S
        return best[1], best[2], outs[best[1]]

    def __call__(self, ex, env, e):
        return self._look(ex, env, e)[1]

    def after(self, ex, env, e, out):
        self.outcomes.append((e, self._score(self._b0, out)))

    def caption(self):
        n = len(self.log)
        ties = float(np.mean([t for *_, t in self.log])) if n else None
        win = {e: sc[w] for e, w, sc, _ in self.log}
        pairs = [(win[e], s) for e, s in self.outcomes if e in win]
        return dict(m=self.m, expansions=n, o_ties=ties, O_TIES=bool(ties is not None and ties > 0.7),
                    rank_corr=_lex_rank_corr(pairs), n_pairs=len(pairs))

    def to_state(self):
        return dict(log=self.log, outcomes=self.outcomes)

    def from_state(self, d):
        self.log, self.outcomes = d["log"], d.get("outcomes", [])


class ODice(OChooser):
    """O_4-dice: as O_4, but the winner is re-executed on its own look-ahead stream X_{e,w}; actions, frames
    and max S are checked against the look-ahead after the rollout. Wire as Explorer2s(chooser=o, after=o.after)."""

    def __init__(self, m=4):
        super().__init__(m)
        self.checked = 0
        self._want = None

    def __call__(self, ex, env, e):
        w, p, out = self._look(ex, env, e)
        self._want = (e, w, out["acts"], out["frames"], out["max_S"])
        return (*p, ex.stream(e, w, 1))

    def after(self, ex, env, e, out):
        super().after(ex, env, e, out)
        we, w, acts, frames, max_S = self._want
        if (e, out["acts"], out["frames"], out["max_S"]) != (we, acts, frames, max_S):
            got = out["acts"]
            k = next((t for t, (x, y) in enumerate(zip(acts, got)) if x != y), min(len(acts), len(got)))
            raise DiceMismatch(f"expansion {e}, winner {w}: look-ahead {len(acts)} steps, {frames} frames, "
                               f"max S {max_S}; executed {len(got)} steps, {out['frames']} frames, "
                               f"max S {out['max_S']}; first differing step {k}")
        self.checked += 1

    def caption(self):
        return dict(super().caption(), dice_checked=self.checked)

    def to_state(self):
        return dict(super().to_state(), checked=self.checked)

    def from_state(self, d):
        super().from_state(d)
        self.checked = d["checked"]


class GF4:
    """O_4's placebo: among the same m proposals, the one whose cell has the highest S, then more lives,
    then shallower, then lowest j; no look-ahead; the explorer executes it on X_e."""

    def __init__(self, m=4):
        self.m = m
        self.log = []                     # per expansion: (e, winner, distinct cells among the proposals)

    def __call__(self, ex, env, e):
        A = ex.arch
        props = [ex.propose(e, j) for j in range(self.m)]
        w = max(range(self.m), key=lambda j: (A.S[props[j][0]], A.lvs[props[j][0]], -A.depth[props[j][0]], -j))
        self.log.append((e, w, len({p[0] for p in props})))
        return props[w]

    def caption(self):
        n = len(self.log)
        return dict(m=self.m, expansions=n,
                    share_not_first=float(np.mean([w > 0 for _, w, _ in self.log])) if n else None,
                    share_one_cell=float(np.mean([c == 1 for *_, c in self.log])) if n else None)

    def to_state(self):
        return dict(log=self.log)

    def from_state(self, d):
        self.log = d["log"]


def d_script(s1, lad, k_ref):
    """D's open-loop actions: the start's own lineage above s1 up to its first step with cum >= M_{k_ref}."""
    j = np.flatnonzero(np.asarray(s1["cum"], np.float64) >= lad[k_ref])
    assert len(j), "the lineage does not reach M_k_ref"
    return np.asarray(s1["actions"][:int(j[0]) + 1], np.uint8)


def lineage_map(game, seed, s1):
    """K*map's map: (s_bin(cum), coarse code) every 4th step of the start's own lineage above s1,
    consecutive duplicates removed. Replayed on SenseEnv85(game, seed) from s1 (reset, NOOP, restore)."""
    env = SenseEnv85(game, seed=seed)
    env.start_from(s1["blob"], s1["S"])
    out = []
    for t, a in enumerate(s1["actions"], 1):
        obs = env.step(int(a))[0]
        if t % 4 == 0:
            item = (s_bin(env._cum), codes(obs)[0][0])
            if not out or out[-1] != item:
                out.append(item)
    return out


class KMap:
    """K*map steering (injected into V2s): a map index per archive slot, set on each put (largest map
    index with equal s_bin and coarse-code Hamming distance <= 4 of 48 blocks, else -1). Each expansion: if
    a C_e coin is below 0.5 and some cell has an index >= 0, a uniform C_e pick among the cells with the
    largest index; otherwise V2s's own draw. Process and first slot come from P as in propose(e, 0, cell)."""

    def __init__(self, items, cap=20000, max_dist=4):
        self.sb = np.array([it[0] for it in items], np.int64)
        self.cd = np.frombuffer(b"".join(it[1] for it in items), np.uint8).reshape(len(items), 48) \
            if items else np.zeros((0, 48), np.uint8)
        self.max_dist = max_dist
        self.idx = np.full(cap + 1, -1, np.int64)
        self.coins = self.found = 0
        self._last = (False, False)

    def on_put(self, ex, j):
        A = ex.arch
        ok = (self.sb == A.sbin[j]) & ((self.cd != np.frombuffer(A.codes[j][0], np.uint8)).sum(1) <= self.max_dist)
        hit = np.flatnonzero(ok)
        self.idx[j] = int(hit[-1]) if len(hit) else -1

    def pick_cell(self, ex, rng_p, rng_c, e):
        A = ex.arch
        coin = rng_c.random() < 0.5
        if coin:
            used = np.flatnonzero(A.used)
            mi = self.idx[used]
            if len(mi) and mi.max() >= 0:
                cand = used[mi == mi.max()]
                self._last = (True, True)
                return int(cand[rng_c.integers(len(cand))])
        self._last = (coin, False)
        return ex.draw_cell(rng_p, e)

    def pick_process(self, ex, rng_p, rng_c, i):
        return ex.bandit.pick(rng_p)

    def on_iter(self, ex, e):
        self.coins += int(self._last[0])
        self.found += int(self._last[1])

    @property
    def coverage(self):
        return self.found / self.coins if self.coins else None

    def caption(self):
        """MAP-WEAK if coverage < 30%; an undefined coverage (no coin) also counts as weak (choice: it can
        only block NO-ROOM-K*map, never help it)."""
        c = self.coverage
        return dict(map_size=int(len(self.sb)), coins=self.coins, found=self.found, coverage=c,
                    MAP_WEAK=bool(c is None or c < 0.30))

    def to_state(self):
        return dict(idx=self.idx.copy(), coins=self.coins, found=self.found, last=self._last)

    def from_state(self, d):
        self.idx, self.coins, self.found, self._last = d["idx"].copy(), d["coins"], d["found"], d["last"]


class GF:
    """K*map's placebo, greedy frontier (§4.4, §7.2): with probability 0.5 (a C_e coin) the cell with the
    largest S, ties to the deepest, then uniform (C_e); otherwise V2s's own draw. Process from the bandit."""

    def __init__(self):
        self.coins = 0
        self._last = False

    def on_put(self, ex, j):
        pass

    def pick_cell(self, ex, rng_p, rng_c, e):
        A = ex.arch
        self._last = bool(rng_c.random() < 0.5)
        if self._last:
            used = np.flatnonzero(A.used)
            top = used[A.S[used] == A.S[used].max()]
            top = top[A.depth[top] == A.depth[top].max()]
            return int(top[rng_c.integers(len(top))])
        return ex.draw_cell(rng_p, e)

    def pick_process(self, ex, rng_p, rng_c, i):
        return ex.bandit.pick(rng_p)

    def on_iter(self, ex, e):
        self.coins += int(self._last)

    def caption(self):
        return dict(greedy_picks=self.coins)

    def to_state(self):
        return dict(coins=self.coins, last=self._last)

    def from_state(self, d):
        self.coins, self._last = d["coins"], d["last"]


# ---------------------------------------------------------------- K*mac, K*mac-exact, K~mac libraries
def lib_sha256(lib):
    return hashlib.sha256(json.dumps([[list(map(list, m)), w] for m, w in lib]).encode()).hexdigest()


def kmac_library(s1_lineage, adapter, hold=HOLD, keep_priv=False):
    """(library, report) from a lineage above its s1 (dict with `actions`, `cum`, `S`), mapped through
    `adapter`. K*mac: the other start's s1 and this start's adapter, hold 4. K*mac-exact: this start's own
    s1 and adapter, hold 1, keep_priv=True. Library: [(((slot, steps), ...), count), ...] in order of first
    appearance."""
    acts = np.asarray(s1_lineage["actions"], np.int64)
    cum = np.asarray(s1_lineage["cum"], np.float64)
    prev = np.concatenate([[float(s1_lineage["S"])], cum[:-1]])
    rises = np.flatnonzero(cum > prev)
    ends = rises[rises >= WINDOW - 1]
    counts, kinds = {}, {"unmapped": 0, "priv": 0, "noop": 0, "ok": 0}
    rt = held = held_n = 0
    for t in ends:
        w = acts[t - WINDOW + 1:t + 1]
        slots = []
        for a in w[::hold]:
            s = adapter.slot_of.get(int(a))
            if s is None:
                kind, s = "unmapped", NOOP
            elif SLOTS[s].startswith("priv") and not keep_priv:
                kind, s = "priv", NOOP
            else:
                kind = "noop" if s == NOOP else "ok"
            kinds[kind] += 1
            rt += int(adapter.to_button(s) == int(a))
            slots.append(int(s))
        played = np.repeat([adapter.to_button(s) for s in slots], hold)
        held += int(np.count_nonzero(played == w))
        held_n += len(w)
        rle = []
        for s in slots:
            if rle and rle[-1][0] == s:
                rle[-1][1] += hold
            else:
                rle.append([s, hold])
        key = tuple((s, n) for s, n in rle)
        counts[key] = counts.get(key, 0) + 1
    lib = list(counts.items())
    n = sum(kinds.values())
    rep = dict(hold=hold, keep_priv=keep_priv, rises=int(len(rises)), windows=int(len(ends)),
               rises_without_window=int(len(rises) - len(ends)), size=len(lib), sampled_steps=n,
               fidelity=rt / n if n else None, fidelity_held=held / held_n if held_n else None,
               fidelity_slot=(kinds["ok"] + kinds["noop"]) / n if n else None,
               share_priv_or_noop=(kinds["priv"] + kinds["unmapped"] + kinds["noop"]) / n if n else None,
               share_priv=kinds["priv"] / n if n else None, share_unmapped=kinds["unmapped"] / n if n else None,
               sha256=lib_sha256(lib))
    return lib, rep


def tilde_partners(units):
    """K~mac partners {unit: partner}: a derangement of the unit list drawn with default_rng([9801, 2])."""
    units = list(units)
    assert len(units) >= 2
    rng = np.random.default_rng([9801, 2])
    while True:
        p = rng.permutation(len(units))
        if not np.any(p == np.arange(len(units))):
            return {u: units[int(k)] for u, k in zip(units, p)}


def tilde_library(game, start, units, libs):
    """K~mac's library at (game, start): its partner's cross-start (K*mac) library at the same start, else
    at the partner's other start, else empty. libs: {(unit, start): library, or None if that (unit, start)
    has no K*mac row}."""
    h = tilde_partners(units)[game]
    for s in (start, OTHER[start]):
        if libs.get((h, s)) is not None:
            return h, s, libs[(h, s)]
    return h, None, []


def tilde_kmac(game, start, units, load_s1, load_body):
    """K~mac through the job's loaders (load_s1(unit, start) -> s1 or None; load_body(unit, start) -> Body):
    (partner, partner start or None, library, report or None). The partner's K*mac row at start s exists when
    both of its starts are valid: its other start's lineage mapped through its start-s adapter."""
    h = tilde_partners(units)[game]
    for s in (start, OTHER[start]):
        lin = load_s1(h, OTHER[s])
        if lin is not None and load_s1(h, s) is not None:
            lib, rep = kmac_library(lin, load_body(h, s).adapter)
            return h, s, lib, rep
    return h, None, [], None
