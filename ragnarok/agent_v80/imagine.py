"""Imagination with the theory (docs/ARC4_V80_DESIGN.md §3): play ahead in the head, no emulator.

From the current frame's tracked objects, my position and the gauges, a particle plays `hold`
steps of a candidate slot then the base slot, with the theory's rules: my displacement card, my
shots (spawned when the slot spawns, after a cooldown), each type's motion rule, births of new
objects, contacts and the gauge risk. Faithful to how the lab counted things (review v80):
  * a contact is resolved ONCE per object (one outcome drawn from the ledger, as the ledger counts
    one outcome per contact), my shots likewise;
  * each random mechanism (my motion noise, shots, births, contacts, gauge) has its own stream with
    a fixed number of draws per step, so particles with the same seed are paired across slots
    (common random numbers) and differ only through the consequences of the slot.
A particle returns (score events, died, step of death, my end position, gauge quartile at the end).
"""
import numpy as np

from ragnarok.agent_v80.theory import I, OUTCOMES, me_row, predict_motion

D_DEATH, D_SCORE, D_VANISH, D_DISP = (OUTCOMES.index(o) for o in ("death", "score", "vanish", "displaced"))
NC = 13   # x y vx vy w h type mine alive u_me u_mine done_me done_mine


class Scene:
    """What the imagination starts from: object rows (see NC), my position and size, gauge fills."""

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
                         self.keys.index(key), float(th.is_mine_row(r)), 1.0])
        self.objs = np.array(objs, np.float64).reshape(-1, 9)
        if k is not None:
            me_wh = (float(rows[k, I["x1"]] - rows[k, I["x0"]]), float(rows[k, I["y1"]] - rows[k, I["y0"]]))
        self.me = np.array(me, np.float64) if np.isfinite(me[0]) else None
        self.me_wh = me_wh
        self.fills = th.gauge_fill(rows)
        self.no_me = k is None


def _overlap(ax, ay, aw, ah, X, Y, Wd, Hd, pad=1.0):
    return (np.abs(X - ax) * 2 <= Wd + aw + 2 * pad) & (np.abs(Y - ay) * 2 <= Hd + ah + 2 * pad)


def _pick(p, u):
    return min(int(np.searchsorted(np.cumsum(p), u, side="right")), len(p) - 1)


class Model:
    """Per-theory constants the particles need, computed once per decision."""

    def __init__(self, th, keys):
        self.keys = keys
        self.p_me = [th.contact_probs(k, "me") for k in keys]
        pm = []
        for k in keys:
            p = th.contact_probs(k, "mine").copy()
            p[D_DEATH] = 0.0                               # my shot cannot kill me
            pm.append(p / p.sum())
        self.p_mine = pm
        self.disp = [th.displacement(k) for k in keys]
        self.rules = [th.types[k]["rule"] for k in keys]
        self.bounds = [th.types[k]["bounds"] for k in keys]
        self.speeds = [th.types[k]["speed"] for k in keys]
        self.wh = [(th.types[k].get("w", 4.0), th.types[k].get("h", 4.0)) for k in keys]
        self.births = [(keys.index(k), b) for k, b in th.births.items() if k in th.types and b["samples"]]


def rollout(th, md, sc, first, hold, horizon, sd, dp=2.0):
    """One imagined future. Returns (events, died, death_step, me_end, gauge_q_end)."""
    if sc.me is None:
        return 0.0, False, -1, None, -1
    r_me, r_shot, r_birth, r_con, r_gauge, r_zone = (np.random.default_rng(s) for s in np.random.SeedSequence(int(sd)).spawn(6))
    O = np.zeros((sc.objs.shape[0], NC))
    O[:, :9] = sc.objs
    O[:, 9:11] = r_con.random((O.shape[0], 2))
    me = sc.me.copy()
    mw, mh = sc.me_wh
    fills = list(sc.fills)
    x0, x1, y0, y1 = th.me_bounds
    g = th.gauges[0] if th.gauges and fills and fills[0] is not None else None
    base_risk = min(g["risk"]) if g else 0.0
    events, cool = 0.0, 0
    for t in range(hold + horizon):
        s = first if t < hold else th.base
        dx, dy, sdx, sdy = th.me_card.get(s, (0.0, 0.0, 0.0, 0.0))
        e = r_me.standard_normal(2)
        me[0] = np.clip(me[0] + dx + 0.25 * sdx * e[0], x0, x1)
        me[1] = np.clip(me[1] + dy + 0.25 * sdy * e[1], y0, y1)
        cool -= 1
        us = r_shot.random(3)                              # drawn every step, used or not
        m = th.mine.get(s)
        if m is not None and cool <= 0 and tuple(m["key"]) in th.types and us[0] < min(1.0, m["rate"] * m["cooldown"]):
            ki = md.keys.index(tuple(m["key"]))
            w, h = md.wh[ki]
            O = np.vstack([O, [me[0] + m["off"][0], me[1] + m["off"][1], m["v"][0], m["v"][1], w, h, ki, 1.0, 1.0,
                               us[1], us[2], 0.0, 0.0]])
            cool = m["cooldown"]
        for j in range(O.shape[0]):                       # motion by each type's rule, one step
            if O[j, 8] < 1:
                continue
            ki = int(O[j, 6])
            nx, ny = predict_motion(md.rules[ki], O[j, 0], O[j, 1], O[j, 2], O[j, 3], 1, md.bounds[ki], me=me,
                                    speed=md.speeds[ki])
            if md.rules[ki] in ("cv", "cv_wrap", "cv_bounce"):
                O[j, 2], O[j, 3] = nx - O[j, 0], ny - O[j, 1]
                if md.rules[ki] == "cv_wrap":
                    O[j, 2] = O[j, 2] if abs(O[j, 2]) < 80 else O[j, 2] - np.sign(O[j, 2]) * 160
            O[j, 0], O[j, 1] = nx, ny
        for ki, b in md.births:                           # new objects; 4 draws per type per step
            ub = r_birth.random(4)
            if ub[0] < b["rate"]:
                x, y, vx, vy = b["samples"][int(ub[1] * len(b["samples"]))]
                w, h = md.wh[ki]
                O = np.vstack([O, [x, y, vx, vy, w, h, ki, 0.0, 1.0, ub[2], ub[3], 0.0, 0.0]])
        alive = O[:, 8] > 0
        mine = O[:, 7] > 0
        hit = alive & ~mine & (O[:, 11] == 0) & _overlap(me[0], me[1], mw, mh, O[:, 0], O[:, 1], O[:, 4], O[:, 5])
        for j in np.nonzero(hit)[0]:                      # something touches me: resolved once
            O[j, 11] = 1.0
            o = _pick(md.p_me[int(O[j, 6])], O[j, 9])
            if o == D_DEATH:
                return events, True, t, me, -1
            if o == D_SCORE:
                events += 1
            if o in (D_SCORE, D_VANISH):
                O[j, 8] = 0
            if o == D_DISP:
                me[0] = np.clip(me[0] + md.disp[int(O[j, 6])][0], x0, x1)
                me[1] = np.clip(me[1] + md.disp[int(O[j, 6])][1], y0, y1)
        for mj in np.nonzero(alive & mine & (O[:, 8] > 0))[0]:   # my shots touch things: resolved once
            tgt = (O[:, 8] > 0) & ~mine & (O[:, 12] == 0) & _overlap(O[mj, 0], O[mj, 1], O[mj, 4], O[mj, 5],
                                                                      O[:, 0], O[:, 1], O[:, 4], O[:, 5])
            for j in np.nonzero(tgt)[0]:
                O[j, 12] = 1.0
                o = _pick(md.p_mine[int(O[j, 6])], O[j, 10])
                if o == D_SCORE:
                    events += 1
                if o in (D_SCORE, D_VANISH):
                    O[j, 8] = 0
                    O[mj, 8] = 0
                    break
        O = O[(O[:, 8] > 0) & (O[:, 0] > -20) & (O[:, 0] < 180) & (O[:, 1] > -20) & (O[:, 1] < 230)]
        uz = r_zone.random(2)                             # places that give points or kill
        pz_s, pz_d = th.zone_event(me)
        if uz[0] < pz_d:
            return events, True, t, me, -1
        if uz[1] < pz_s:
            events += 1
            if "score" in th.relocate:
                me[0], me[1] = th.relocate["score"]
        ug = r_gauge.random()
        if g is not None:                                 # the gauge: drift by slot, excess risk by fill
            fills[0] = float(np.clip(fills[0] + g["beta"].get(s, 0.0), g["lo"], g["hi"]))
            q = int(np.clip(np.digitize(fills[0], np.array(g["edges"])) - 1, 0, 3))
            extra = max(0.0, g["risk"][q] - base_risk)
            if ug < 1 - (1 - extra) ** (1 / 8):
                return events, True, t, me, q
    gq = th.gauge_q(fills) if g is not None else -1
    return events, False, -1, me, gq


def imagine(th, sc, slot, rng_seeds, hold=8, horizon=24, dp=2.0, leaf=True, md=None):
    """Particles for one slot: array rows (value, events, died)."""
    md = md or Model(th, sc.keys)
    out = []
    for sd in rng_seeds:
        ev, died, _, me, gq = rollout(th, md, sc, slot, hold, horizon, int(sd), dp)
        v = ev - dp * died
        if leaf and not died and me is not None:
            v += th.leaf_value(me, gq)
        out.append((v, ev, float(died)))
    return np.array(out)


def decide(th, sc, rng, K=16, hold=8, horizon=24, tau=0.25, agree=0.8):
    """Deviate from the base only when the imagined gain is clear and shared: mean gain >= tau and,
    pairing particles by seed, the deviation is at least as good as the base in >= `agree` of them."""
    seeds = rng.integers(2 ** 31, size=K)
    md = Model(th, sc.keys)
    V = {s: imagine(th, sc, s, seeds, hold, horizon, md=md)[:, 0] for s in th.slots}
    base = V[th.base]
    best = max(th.slots, key=lambda s: (V[s].mean(), s == th.base))
    if best != th.base and V[best].mean() - base.mean() >= tau and np.mean(V[best] >= base) >= agree:
        return best, V
    return th.base, V
