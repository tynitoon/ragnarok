"""v85 evaluator instruments (docs/ARC4_V85_PROTOCOL_EN.md A.7). Development only; never agents.

Done here: O_m (choose among V2s's own proposals after an unbilled look-ahead, then execute on a fresh
stream) and K*map (where to restart: steer toward cells matching the start's own lineage).
Still to write (Part B step 5): D, O_4-dice, K*mac and K*mac-exact, placebos GF4, GF and K~mac.
"""
import pickle

import numpy as np

from ragnarok.agent_v83.explorer import s_bin
from ragnarok.agent_v84.framecode import codes
from ragnarok.evaluation.senses_v85 import SenseEnv85


class OChooser:
    """O_m: candidates j = 0..m-1 from propose(e, j), each played in env.sandbox() on X_{e,j}; score
    (max(0, max S - best_S), ADV against archive + overlay, keys new to archive + overlay), compared
    lexicographically, ties to the lowest j. The explorer executes the winner on X_e."""

    def __init__(self, m=4):
        self.m = m
        self.log = []                     # per expansion: (e, winner, scores, all tied)

    def __call__(self, ex, env, e):
        saved = pickle.dumps(ex.body.tracker)
        best, scores = None, []
        for j in range(self.m):
            p = ex.propose(e, j)
            try:
                with env.sandbox():
                    out = ex._rollout(env, *p, ex.stream(e, j, 1), e, dry=True)
            finally:
                ex.body.tracker = pickle.loads(saved)
            s = (max(0.0, out["max_S"] - ex.best_S), int(out["adv"]), int(out["new_cells"]))
            scores.append(s)
            if best is None or s > best[0]:
                best = (s, j, p)
        self.log.append((e, best[1], scores, len(set(scores)) == 1))
        return best[2]

    def to_state(self):
        return dict(log=self.log)

    def from_state(self, d):
        self.log = d["log"]


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

    def to_state(self):
        return dict(idx=self.idx.copy(), coins=self.coins, found=self.found, last=self._last)

    def from_state(self, d):
        self.idx, self.coins, self.found, self._last = d["idx"].copy(), d["coins"], d["found"], d["last"]
