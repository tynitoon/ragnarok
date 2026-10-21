"""Imagination with the theory (docs/ARC4_V80_DESIGN.md §3): play ahead in the head, no emulator.

From the current frame's tracked objects, my position and the gauges, a particle plays `hold`
steps of a candidate slot then the base slot, with the theory's rules: my displacement card, my
shots (spawned when the slot spawns, after a cooldown), each type's motion rule, births of new
objects, contacts (outcomes drawn from the contact ledger) and the gauge risk. A particle returns
(score events, died, step of death). Decisions compare slots with common random numbers.
"""
import numpy as np

from ragnarok.agent_v80.lab import COLS
from ragnarok.agent_v80.theory import I, OUTCOMES, me_row, predict_motion

D_DEATH, D_SCORE, D_VANISH = OUTCOMES.index("death"), OUTCOMES.index("score"), OUTCOMES.index("vanish")


class Scene:
    """What the imagination starts from: arrays of objects (x, y, vx, vy, w, h, type index, mine,
    alive), my position and size, gauge fills."""

    def __init__(self, th, rows, me, me_wh=(6.0, 6.0)):
        self.keys = list(th.types)
        k = me_row(rows, me) if rows.shape[0] else None
        objs = []
        for j, r in enumerate(rows):
            if j == k or r[I["missed"]] > 0 or th._is_gauge_row(r):
                continue
            key = th.key_of(r)
            if key is None:
                continue
            objs.append([r[I["cx"]], r[I["cy"]], r[I["vx"]], r[I["vy"]], r[I["x1"]] - r[I["x0"]], r[I["y1"]] - r[I["y0"]],
                         self.keys.index(key), float(th.types[key].get("mine", False)), 1.0])
        self.objs = np.array(objs, np.float64).reshape(-1, 9)
        if k is not None:
            me_wh = (float(rows[k, I["x1"]] - rows[k, I["x0"]]), float(rows[k, I["y1"]] - rows[k, I["y0"]]))
        self.me = np.array(me, np.float64) if np.isfinite(me[0]) else None
        self.me_wh = me_wh
        self.fills = th.gauge_fill(rows)


def _overlap(ax, ay, aw, ah, X, Y, Wd, Hd, pad=1.0):
    return (np.abs(X - ax) * 2 <= Wd + aw + 2 * pad) & (np.abs(Y - ay) * 2 <= Hd + ah + 2 * pad)


def rollout(th, sc, first, hold, horizon, rng, dp=2.0):
    """One imagined future. Returns (events, died, death_step, me_end, gauge_q_end)."""
    if sc.me is None:
        return 0.0, False, -1, None, -1
    O = sc.objs.copy()
    me = sc.me.copy()
    mw, mh = sc.me_wh
    fills = list(sc.fills)
    keys = sc.keys
    probs_me = [th.contact_probs(k, "me") for k in keys]
    probs_mine = [th.contact_probs(k, "mine") for k in keys]
    rules = [th.types[k]["rule"] for k in keys]
    bounds = [th.types[k]["bounds"] for k in keys]
    speeds = [th.types[k]["speed"] for k in keys]
    x0, x1, y0, y1 = th.me_bounds
    g = th.gauges[0] if th.gauges and fills and fills[0] is not None else None
    base_risk = min(g["risk"]) if g else 0.0
    events, cool = 0.0, 0
    for t in range(hold + horizon):
        s = first if t < hold else th.base
        dx, dy, sdx, sdy = th.me_card.get(s, (0.0, 0.0, 0.0, 0.0))
        me[0] = np.clip(me[0] + dx + 0.25 * sdx * rng.standard_normal(), x0, x1)
        me[1] = np.clip(me[1] + dy + 0.25 * sdy * rng.standard_normal(), y0, y1)
        cool -= 1
        m = th.mine.get(s)
        if m is not None and cool <= 0 and tuple(m["key"]) in th.types and rng.random() < min(1.0, m["rate"] * m["cooldown"]):
            ki = keys.index(tuple(m["key"]))
            ty = th.types[tuple(m["key"])]
            O = np.vstack([O, [me[0] + m["off"][0], me[1] + m["off"][1], m["v"][0], m["v"][1],
                               ty.get("w", 2.0), ty.get("h", 4.0), ki, 1.0, 1.0]])
            cool = m["cooldown"]
        for j in range(O.shape[0]):                       # motion by each type's rule, one step
            if O[j, 8] < 1:
                continue
            ki = int(O[j, 6])
            nx, ny = predict_motion(rules[ki], O[j, 0], O[j, 1], O[j, 2], O[j, 3], 1, bounds[ki], me=me, speed=speeds[ki])
            if rules[ki] in ("cv", "cv_wrap", "cv_bounce"):
                O[j, 2], O[j, 3] = nx - O[j, 0], ny - O[j, 1]
            O[j, 0], O[j, 1] = nx, ny
        for key, b in th.births.items():                  # new objects
            if key in th.types and b["samples"] and rng.random() < b["rate"]:
                x, y, vx, vy = b["samples"][int(rng.integers(len(b["samples"])))]
                ty = th.types[key]
                O = np.vstack([O, [x, y, vx, vy, ty.get("w", 4.0), ty.get("h", 4.0), keys.index(key), 0.0, 1.0]])
        alive = O[:, 8] > 0
        mine = O[:, 7] > 0
        hit = alive & ~mine & _overlap(me[0], me[1], mw, mh, O[:, 0], O[:, 1], O[:, 4], O[:, 5])
        for j in np.nonzero(hit)[0]:                      # something touches me
            o = rng.choice(4, p=probs_me[int(O[j, 6])])
            if o == D_DEATH:
                return events, True, t, me, -1
            if o == D_SCORE:
                events += 1
            if o in (D_SCORE, D_VANISH):
                O[j, 8] = 0
        for mj in np.nonzero(alive & mine & (O[:, 8] > 0))[0]:   # my shots touch things
            tgt = (O[:, 8] > 0) & ~mine & _overlap(O[mj, 0], O[mj, 1], O[mj, 4], O[mj, 5], O[:, 0], O[:, 1], O[:, 4], O[:, 5])
            for j in np.nonzero(tgt)[0]:
                p = probs_mine[int(O[j, 6])].copy(); p[D_DEATH] = 0; p = p / p.sum()
                o = rng.choice(4, p=p)
                if o == D_SCORE:
                    events += 1
                if o in (D_SCORE, D_VANISH):
                    O[j, 8] = 0; O[mj, 8] = 0
                    break
        O = O[(O[:, 8] > 0) & (O[:, 0] > -20) & (O[:, 0] < 180) & (O[:, 1] > -20) & (O[:, 1] < 230)]
        if g is not None:                                 # the gauge: drift by slot, risk by fill
            fills[0] = float(np.clip(fills[0] + g["beta"].get(s, 0.0), g["lo"], g["hi"]))
            q = int(np.clip(np.digitize(fills[0], np.array(g["edges"])) - 1, 0, 3))
            extra = max(0.0, g["risk"][q] - base_risk)
            if rng.random() < 1 - (1 - extra) ** (1 / 8):
                return events, True, t, me, q
    gq = th.gauge_q(fills) if g is not None else -1
    return events, False, -1, me, gq


def imagine(th, sc, slot, rng_seeds, hold=8, horizon=24, dp=2.0, leaf=True):
    """Particles for one slot: arrays (value, events, died)."""
    out = []
    for sd in rng_seeds:
        ev, died, _, me, gq = rollout(th, sc, slot, hold, horizon, np.random.default_rng(int(sd)), dp)
        v = ev - dp * died
        if leaf and not died and me is not None:
            v += th.leaf_value(me, gq)
        out.append((v, ev, float(died)))
    return np.array(out)


def decide(th, sc, rng, K=16, hold=8, horizon=24, tau=0.25, agree=0.8):
    """Deviate from the base only when the imagined gain is clear and shared by the particles."""
    seeds = rng.integers(2 ** 31, size=K)
    V = {s: imagine(th, sc, s, seeds, hold, horizon)[:, 0] for s in th.slots}
    base = V[th.base]
    best = max(th.slots, key=lambda s: (V[s].mean(), s == th.base))
    if best != th.base and V[best].mean() - base.mean() >= tau and np.mean(V[best] > base) >= agree:
        return best, V
    return th.base, V
