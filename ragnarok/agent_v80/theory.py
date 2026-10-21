"""The agent's theory of a game (docs/ARC4_V80_DESIGN.md §3), fitted from its own lab traces.

Readable by construction: every part is a small table or a named rule, exportable as JSON.
  types    : objects grouped by (dominant colour, size bin, aspect bin); >= MIN_SIGHT sightings
  motion   : per type, a rule from MOTIONS chosen by held-out 8-step prediction error
  me_card  : per slot, my mean displacement per step and its spread; my bounds on the screen
  mine     : per slot, whether pressing it spawns an object near me (sibling evidence), with the
             spawned object's type, offset, velocity and cooldown
  contacts : per type, outcomes after touching ME and after being touched by MINE:
             counts of {death, score, vanish, nothing}, interventions and observations apart
  gauges   : objects with a fixed anchor and a varying extent; per slot drift of the fill; risk of
             my death against the fill
  births   : per type, rate of appearance per step and an empirical list of (x, y, vx, vy)
  leaf     : my coarse cell x gauge quartile -> my mean events over the next 32 steps (base policy)
Inputs are lab traces (lab.py): screen-derived rows, my position, my slot, my HUD events and my
death signal. Nothing from RAM or the true score.
"""
import json

import numpy as np

from ragnarok.agent_v80.lab import COLS

I = {c: i for i, c in enumerate(COLS)}
MIN_SIGHT = 30
MOTIONS = ("static", "cv", "cv_wrap", "cv_bounce", "toward_me")
OUTCOMES = ("death", "score", "vanish", "nothing")
W, H = 160, 210
CELL = 16


def type_key(r):
    w, h = max(1.0, r[I["x1"]] - r[I["x0"]]), max(1.0, r[I["y1"]] - r[I["y0"]])
    return (int(r[I["colour"]]), int(round(np.log2(max(1.0, r[I["area"]])))), int(round(np.log2(w / h))))


def me_row(rows, me, radius=6.0):
    """Index of the row that is me (nearest visible row to my tracked position), or None."""
    if rows.shape[0] == 0 or not np.isfinite(me[0]):
        return None
    vis = rows[:, I["missed"]] == 0
    d = np.abs(rows[:, I["cx"]] - me[0]) + np.abs(rows[:, I["cy"]] - me[1])
    d[~vis] = 1e9
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def overlap(a, b, pad=1.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def predict_motion(rule, x, y, vx, vy, k, bounds, me=None, speed=0.0):
    if rule == "static":
        return x, y
    if rule == "toward_me" and me is not None:
        dx, dy = me[0] - x, me[1] - y
        n = max(1e-6, np.hypot(dx, dy))
        step = min(n, speed * k)
        return x + dx / n * step, y + dy / n * step
    nx, ny = x + vx * k, y + vy * k
    if rule == "cv_wrap":
        nx = nx % W
    elif rule == "cv_bounce":
        x0, x1, y0, y1 = bounds
        for _ in range(4):
            if nx < x0:
                nx = 2 * x0 - nx
            elif nx > x1:
                nx = 2 * x1 - nx
            if ny < y0:
                ny = 2 * y0 - ny
            elif ny > y1:
                ny = 2 * y1 - ny
    return nx, ny


class Theory:
    def __init__(self, slots, base):
        self.slots = [int(s) for s in slots]
        self.base = int(base)
        self.types = {}          # key -> dict(n, rule, speed, bounds, w, h, mine)
        self.me_card = {}        # slot -> (dx, dy, sdx, sdy)
        self.me_bounds = (0, W, 0, H)
        self.mine = {}           # slot -> dict(rate, key, off, v, cooldown) when the slot spawns
        self.contacts = {}       # key -> {"me": {"int": counts, "obs": counts}, "mine": {...}}
        self.gauges = []         # dict(anchor, axis, beta{slot}, lo, hi, risk[bins])
        self.births = {}         # key -> dict(rate, samples[(x, y, vx, vy)])
        self.leaf = {}           # (cx, cy, gq) -> (sum, n)
        self.leaf_mean = 0.0
        self.log = []

    # ------------------------------------------------------------------ fitting
    def fit(self, sibsets, hold=8):
        traces = [tr for s in sibsets for tr in s.values()]
        self._fit_types(traces)
        self._fit_me(traces, hold)
        self._fit_gauges(traces, hold)
        self._fit_mine(sibsets, hold)
        self._fit_motion(traces)
        self._fit_contacts(sibsets, hold)
        self._fit_births(traces)
        self._fit_leaf(traces, hold)
        return self

    def _rows_not_me(self, tr, t):
        rows = tr["objs"][t]
        k = me_row(rows, tr["me"][t])
        keep = np.ones(rows.shape[0], bool)
        if k is not None:
            keep[k] = False
        return rows[keep]

    def _fit_types(self, traces):
        cnt = {}
        for tr in traces:
            for t in range(len(tr["objs"])):
                for r in self._rows_not_me(tr, t):
                    if r[I["missed"]] == 0:
                        k = type_key(r)
                        cnt[k] = cnt.get(k, 0) + 1
        self.types = {k: dict(n=n, rule="cv", speed=0.0, bounds=(0, W, 0, H), mine=False)
                      for k, n in cnt.items() if n >= MIN_SIGHT}

    def key_of(self, r):
        k = type_key(r)
        return k if k in self.types else None

    def _fit_me(self, traces, hold):
        d = {s: [] for s in self.slots}
        xs, ys = [], []
        for tr in traces:
            me = tr["me"]
            for t in range(min(hold, len(me) - 1)):
                if np.isfinite(me[t, 0]) and np.isfinite(me[t + 1, 0]):
                    d[int(tr["slot"][t])].append(me[t + 1] - me[t])
            ok = np.isfinite(me[:, 0])
            xs += me[ok, 0].tolist(); ys += me[ok, 1].tolist()
        for s, v in d.items():
            v = np.array(v) if v else np.zeros((1, 2))
            self.me_card[s] = (float(np.median(v[:, 0])), float(np.median(v[:, 1])),
                               float(np.std(v[:, 0])), float(np.std(v[:, 1])))
        if xs:
            self.me_bounds = (float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys)))

    def _fit_gauges(self, traces, hold):
        """A gauge: a track whose anchor (left or top edge) stays put while its extent varies."""
        seq = {}
        for ti, tr in enumerate(traces):
            for t, rows in enumerate(tr["objs"]):
                for r in rows:
                    if r[I["missed"]] == 0:
                        seq.setdefault((ti, int(r[I["id"]])), []).append((t, r))
        cand = {}
        for (ti, k), s in seq.items():
            if len(s) < 8:
                continue
            R = np.array([r for _, r in s])
            for axis, a0, a1 in (("x", I["x0"], I["x1"]), ("y", I["y0"], I["y1"])):
                ext = R[:, a1] - R[:, a0]
                if np.std(R[:, a0]) < 0.6 and np.std(ext) >= 2.0:
                    anchor = (axis, int(round(np.median(R[:, I["x0"]]))), int(round(np.median(R[:, I["y0"]]))))
                    cand.setdefault(anchor, []).append((ti, k))
        self.gauges = []
        for anchor, members in cand.items():
            if len(members) < 5:
                continue
            axis = anchor[0]
            a0, a1 = (I["x0"], I["x1"]) if axis == "x" else (I["y0"], I["y1"])
            drift = {s: [] for s in self.slots}
            fills, deaths = [], []
            for ti, k in members:
                tr = traces[ti]
                f = {t: r[a1] - r[a0] for t, r in seq[(ti, k)]}
                for t in range(len(tr["slot"])):
                    if t in f and t + 1 in f:
                        drift[int(tr["slot"][t])].append(f[t + 1] - f[t])
                    if t in f:
                        fills.append(f[t]); deaths.append(bool(tr["death"][t:t + 8].any()))
            if not fills:
                continue
            lo, hi = float(np.min(fills)), float(np.max(fills))
            edges = np.linspace(lo, hi + 1e-6, 5)
            fb = np.clip(np.digitize(fills, edges) - 1, 0, 3)
            risk = [float((np.sum(np.array(deaths)[fb == q]) + 0.5) / (np.sum(fb == q) + 1.0)) for q in range(4)]
            self.gauges.append(dict(anchor=anchor, axis=axis, lo=lo, hi=hi, edges=edges.tolist(), risk=risk,
                                    beta={s: (float(np.mean(v)) if v else 0.0) for s, v in drift.items()}))

    def gauge_fill(self, rows):
        """Current fill of each gauge (or None) from one frame's rows."""
        out = []
        for g in self.gauges:
            axis, ax, ay = g["anchor"]
            a0, a1 = (I["x0"], I["x1"]) if axis == "x" else (I["y0"], I["y1"])
            best = None
            for r in rows:
                if r[I["missed"]] == 0 and abs(r[I["x0"]] - ax) <= 1 and abs(r[I["y0"]] - ay) <= 1:
                    best = float(r[a1] - r[a0])
            out.append(best)
        return out

    def _is_gauge_row(self, r):
        return any(abs(r[I["x0"]] - g["anchor"][1]) <= 1 and abs(r[I["y0"]] - g["anchor"][2]) <= 1 for g in self.gauges)

    def _fit_mine(self, sibsets, hold, radius=16.0):
        """Objects born near me during the hold, per slot; a slot spawns if its birth rate exceeds
        the lowest slot's by > 0.05 per step (sibling comparison: same root, different action)."""
        births = {s: [] for s in self.slots}
        steps = {s: 0 for s in self.slots}
        for sib in sibsets:
            for s, tr in sib.items():
                seen = set(int(r[I["id"]]) for r in tr["objs"][0])
                for t in range(1, min(hold + 1, len(tr["objs"]))):
                    steps[s] += 1
                    me = tr["me"][t - 1]
                    for r in tr["objs"][t]:
                        i = int(r[I["id"]])
                        if i in seen:
                            continue
                        seen.add(i)
                        if np.isfinite(me[0]) and abs(r[I["cx"]] - me[0]) + abs(r[I["cy"]] - me[1]) <= radius:
                            v = None                    # its velocity two steps later, if still tracked
                            if t + 2 < len(tr["objs"]):
                                nxt = tr["objs"][t + 2]
                                hit = nxt[(nxt[:, I["id"]] == i) & (nxt[:, I["missed"]] == 0)]
                                if hit.shape[0]:
                                    v = (float(hit[0, I["vx"]]), float(hit[0, I["vy"]]))
                            births[s].append((t, r, me, v))
                for t in range(min(hold + 1, len(tr["objs"])), len(tr["objs"])):
                    seen |= set(int(r[I["id"]]) for r in tr["objs"][t])
        rate = {s: len(b) / max(1, steps[s]) for s, b in births.items()}
        floor = min(rate.values()) if rate else 0.0
        self.mine = {}
        for s, b in births.items():
            if rate[s] - floor <= 0.05 or not b:
                continue
            keys = [type_key(r) for _, r, _, _ in b]
            key = max(set(keys), key=keys.count)
            ts = sorted(t for t, _, _, _ in b)
            vs = [v for _, _, _, v in b if v is not None]
            gaps = np.diff(ts) if len(ts) > 1 else np.array([hold])
            self.mine[s] = dict(rate=float(rate[s] - floor), key=list(key),
                                off=[float(np.median([r[I["cx"]] - m[0] for _, r, m, _ in b])),
                                     float(np.median([r[I["cy"]] - m[1] for _, r, m, _ in b]))],
                                v=[float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs]))] if vs else [0.0, 0.0],
                                cooldown=int(max(1, np.min(gaps[gaps > 0]) if np.any(gaps > 0) else hold)))
            if key in self.types:
                self.types[key]["mine"] = True

    def _fit_motion(self, traces, k=8):
        """Per type, the rule with the lowest median k-step error on track segments (prequential:
        the velocity used is the one observed at the segment start)."""
        seg = {}
        for tr in traces:
            byid = {}
            for t, rows in enumerate(tr["objs"]):
                for r in rows:
                    if r[I["missed"]] == 0:
                        byid.setdefault(int(r[I["id"]]), {})[t] = r
            for i, d in byid.items():
                for t0, r0 in d.items():
                    if t0 + k in d and (t0 - 1) in d:
                        key = self.key_of(r0)
                        if key is not None:
                            me = tr["me"][t0 + k] if np.isfinite(tr["me"][t0 + k, 0]) else None
                            seg.setdefault(key, []).append((r0, d[t0 + k], me))
        for key, ty in self.types.items():
            S = seg.get(key, [])
            if len(S) < 5:
                ty["rule"], ty["err"] = "cv", None
                continue
            xs = np.array([[a[I["cx"]], a[I["cy"]]] for a, _, _ in S])
            ty["bounds"] = (float(xs[:, 0].min()), float(xs[:, 0].max()), float(xs[:, 1].min()), float(xs[:, 1].max()))
            ty["speed"] = float(np.median([np.hypot(a[I["vx"]], a[I["vy"]]) for a, _, _ in S]))
            errs = {}
            for rule in MOTIONS:
                e = []
                for a, b, me in S:
                    px, py = predict_motion(rule, a[I["cx"]], a[I["cy"]], a[I["vx"]], a[I["vy"]], k, ty["bounds"],
                                            me=me, speed=ty["speed"])
                    e.append(abs(px - b[I["cx"]]) + abs(py - b[I["cy"]]))
                errs[rule] = float(np.median(e))
            ty["rule"] = min(MOTIONS, key=lambda m: (errs[m], MOTIONS.index(m)))
            ty["err"] = errs
            ty["w"] = float(np.median([a[I["x1"]] - a[I["x0"]] for a, _, _ in S]))
            ty["h"] = float(np.median([a[I["y1"]] - a[I["y0"]] for a, _, _ in S]))

    def _fit_contacts(self, sibsets, hold, window=3):
        """Outcome after a contact, per type. A contact within the causal cone (the first
        hold+window steps of a branch, near me) whose occurrence differs between sibling branches
        is an intervention; any other contact is an observation."""
        self.contacts = {k: {"me": {"int": np.zeros(4), "obs": np.zeros(4)},
                             "mine": {"int": np.zeros(4), "obs": np.zeros(4)}} for k in self.types}
        for sib in sibsets:
            touched = {}
            for s, tr in sib.items():
                touched[s] = self._contacts_in(tr, window)
            for s, events in touched.items():
                others = set().union(*[set((e[0], e[1]) for e in touched[o]) for o in touched if o != s]) if len(touched) > 1 else set()
                for kind, key, t, outcome in events:
                    if key not in self.contacts:
                        continue
                    cone = t <= hold + window and (kind, key) not in others
                    self.contacts[key][kind]["int" if cone else "obs"][OUTCOMES.index(outcome)] += 1

    def _contacts_in(self, tr, window):
        ev, done = [], set()
        n = len(tr["objs"])
        mine_ids = set()
        for t in range(n - 1):
            rows = tr["objs"][t]
            km = me_row(rows, tr["me"][t])
            for r in rows:                                  # my shots: objects of a spawned type near me at birth
                k = self.key_of(r)
                if k is not None and self.types[k].get("mine"):
                    mine_ids.add(int(r[I["id"]]))
            if km is not None:
                mb = rows[km, I["x0"]:I["y1"] + 1]
                for j, r in enumerate(rows):
                    if j == km or r[I["missed"]] or int(r[I["id"]]) in mine_ids or self._is_gauge_row(r):
                        continue
                    key = self.key_of(r)
                    if key is None or ("me", int(r[I["id"]])) in done:
                        continue
                    if overlap(mb, r[I["x0"]:I["y1"] + 1]):
                        done.add(("me", int(r[I["id"]])))
                        ev.append(("me", key, t, self._outcome(tr, t, int(r[I["id"]]), window)))
            for m in [r for r in rows if int(r[I["id"]]) in mine_ids and r[I["missed"]] == 0]:
                for r in rows:
                    if int(r[I["id"]]) in mine_ids or r[I["missed"]] or self._is_gauge_row(r):
                        continue
                    if km is not None and int(r[I["id"]]) == int(rows[km, I["id"]]):
                        continue
                    key = self.key_of(r)
                    if key is None or ("mine", int(r[I["id"]])) in done:
                        continue
                    if overlap(m[I["x0"]:I["y1"] + 1], r[I["x0"]:I["y1"] + 1]):
                        done.add(("mine", int(r[I["id"]])))
                        ev.append(("mine", key, t, self._outcome(tr, t, int(r[I["id"]]), window, mine=True)))
        return ev

    def _outcome(self, tr, t, oid, window, mine=False):
        end = min(len(tr["score"]), t + window + 1)
        if not mine and tr["death"][t:end].any():
            return "death"
        if tr["score"][t:end].any():
            return "score"
        later = [rows for rows in tr["objs"][t + 1:t + window + 2]]
        if later and all(oid not in set(int(x) for x in rows[rows[:, I["missed"]] == 0, I["id"]]) for rows in later):
            return "vanish"
        return "nothing"

    def contact_probs(self, key, kind):
        c = self.contacts.get(key, {}).get(kind)
        if c is None:
            return np.array([0.0, 0.0, 0.0, 1.0])
        n = c["int"] + c["obs"] + 0.25                  # Dirichlet(0.25) per outcome
        return n / n.sum()

    def _fit_births(self, traces):
        steps = 0
        b = {}
        for tr in traces:
            seen = set(int(r[I["id"]]) for r in tr["objs"][0])
            for t in range(1, len(tr["objs"])):
                steps += 1
                for r in tr["objs"][t]:
                    i = int(r[I["id"]])
                    if i in seen:
                        continue
                    seen.add(i)
                    k = self.key_of(r)
                    if k is not None and not self.types[k].get("mine"):
                        b.setdefault(k, []).append((float(r[I["cx"]]), float(r[I["cy"]]), float(r[I["vx"]]), float(r[I["vy"]])))
        self.births = {k: dict(rate=len(v) / max(1, steps), samples=v[:200]) for k, v in b.items()}

    def cell(self, me, gq):
        return (int(me[0]) // CELL, int(me[1]) // CELL, gq)

    def gauge_q(self, fills):
        if not self.gauges or fills is None or not fills or fills[0] is None:
            return -1
        g = self.gauges[0]
        return int(np.clip(np.digitize(fills[0], np.array(g["edges"])) - 1, 0, 3))

    def _fit_leaf(self, traces, hold, n=32, dp=2.0):
        tot = []
        for tr in traces:
            L = len(tr["score"])
            for t in range(hold, L - n + 1):
                me = tr["me"][t]
                if not np.isfinite(me[0]):
                    continue
                v = float(tr["score"][t:t + n].sum()) - dp * float(tr["death"][t:t + n].any())
                key = self.cell(me, self.gauge_q(self.gauge_fill(tr["objs"][t])))
                s = self.leaf.get(key, (0.0, 0))
                self.leaf[key] = (s[0] + v, s[1] + 1)
                tot.append(v)
        self.leaf_mean = float(np.mean(tot)) if tot else 0.0

    def leaf_value(self, me, gq, min_n=10):
        s = self.leaf.get(self.cell(me, gq))
        if s is None or s[1] < min_n:
            return self.leaf_mean
        return s[0] / s[1]

    # ------------------------------------------------------------------ reading
    def export(self):
        """The readable rulebook (no colours or slot ids are needed to read it; they are kept as
        per-game keys only)."""
        return json.dumps(dict(
            base=self.base, me_card=self.me_card, me_bounds=self.me_bounds, mine=self.mine,
            types={str(k): dict(n=v["n"], rule=v["rule"], speed=round(v["speed"], 2), mine=v.get("mine", False),
                                contacts_me=self.contact_probs(k, "me").round(3).tolist(),
                                contacts_mine=self.contact_probs(k, "mine").round(3).tolist())
                   for k, v in self.types.items()},
            gauges=[dict(anchor=g["anchor"], lo=g["lo"], hi=g["hi"], risk=[round(x, 3) for x in g["risk"]],
                         beta={str(s): round(b, 2) for s, b in g["beta"].items()}) for g in self.gauges],
            births={str(k): round(v["rate"], 4) for k, v in self.births.items()},
            leaf_mean=round(self.leaf_mean, 3)), indent=1)
