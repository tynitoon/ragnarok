"""Novelty drive with savestates as an organ (ARC4_VISION 2.3 NOVELTY, 2.5 E4).

Cells are a coarse, game-agnostic summary of the screen (downsampled grayscale, few intensity
levels — the Go-Explore cell rule; Ecoffet et al. 2021). The archive keeps one savestate per
cell. Each iteration returns to a cell chosen by 1/sqrt(visits) and plays from there with
action repeats; every emulated frame and load is charged by the env. No reward, no RAM.
"""
import numpy as np


def cell_of(rgb, w=11, h=8, levels=8):
    g = rgb.mean(-1)
    H, W = g.shape
    g = g[: H - H % h, : W - W % w].reshape(h, H // h, w, W // w).mean((1, 3))
    q = np.floor(g / 256.0 * levels).astype(np.uint8)
    return q.tobytes()


class Archive:
    def __init__(self):
        self.cells = {}          # key -> dict(state, visits, chosen, depth, events)

    def add(self, key, state, depth, events):
        c = self.cells.get(key)
        better = c is None or events > c["events"] or (events == c["events"] and depth < c["depth"])
        if c is None:
            self.cells[key] = dict(state=state, visits=1, chosen=0, depth=depth, events=events)
            return True
        c["visits"] += 1
        if better:
            c.update(state=state, depth=depth, events=events)
        return False

    def choose(self, rng, score_bias=4.0):
        """Novelty (1/sqrt(times chosen)) times a score drive: cells at the best score so far are
        preferred by `score_bias` (G2, active once the reader found a score counter)."""
        keys = list(self.cells)
        best = max(c["events"] for c in self.cells.values())
        w = np.array([(score_bias if self.cells[k]["events"] == best and best > 0 else 1.0) /
                      np.sqrt(self.cells[k]["chosen"] + 1) for k in keys])
        k = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        self.cells[k]["chosen"] += 1
        return k


def me_cell(me, hud_key, shape, grid=16):
    if me is None:
        mb = (-1, -1)
    else:
        mb = (int(me[0] * grid / shape[1]), int(me[1] * grid / shape[0]))
    return (mb, hud_key)


def explore_me(env, perc, iters, rng, steps=100, repeat=0.95, refit_every=200, log=None):
    """Novelty over (my position on a 16x16 grid, the counters' displayed states). Counter
    regions are refit from the agent's own play every `refit_every` iterations; the archive is
    then re-keyed from each cell's stored frame and position."""
    arch = Archive()
    obs = env.reset()
    perc.tracker.pos = None
    me = perc.locate(env, obs)
    arch.add(me_cell(me, perc.hud_key(obs), obs.shape), env.clone(), 0, 0)
    first = arch.cells[next(iter(arch.cells))]
    first["frame"] = (obs, me)
    first["traj"] = b""                         # actions from reset: the demonstration to this cell
    for it in range(iters):
        if refit_every and it and it % refit_every == 0:
            perc.refit_counters(arch)
            old = arch.cells
            arch = Archive()
            for c in old.values():
                o, m = c["frame"]
                k = me_cell(m, perc.hud_key(o), o.shape)
                if k not in arch.cells or c["depth"] < arch.cells[k]["depth"]:
                    arch.cells[k] = dict(c, visits=c["visits"], chosen=c["chosen"])
            if log:
                log(it, len(arch.cells), len(perc.hud.regions))
        k = arch.choose(rng)
        c = arch.cells[k]
        env.restore(c["state"])
        perc.cf.ring = []                       # a restore is a discontinuity, not a change
        perc.hud.reset_tracking()               # monotonicity is judged within one rollout
        perc.tracker.pos = c["frame"][1]
        perc.tracker.miss = 0
        depth, events = c["depth"], c["events"]
        traj = bytearray(c.get("traj", b""))
        s_idx = perc.counters.index(perc.roles["score"]) if perc.roles.get("score") in (perc.counters or []) else None
        l_idx = perc.counters.index(perc.roles["lives"]) if perc.roles.get("lives") in (perc.counters or []) else None
        a = int(rng.integers(env.n_actions))
        for _ in range(steps):
            if rng.random() > repeat:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            traj.append(a)
            perc.observe(obs)
            depth += 1
            if over:
                break
            me = perc.locate(env, obs)
            hk = perc.hud_key(obs)
            ev = perc.hud.step(obs)
            if s_idx is not None and s_idx in ev:
                events += 1
            if l_idx is not None and l_idx in ev:          # a death ends the rollout (G1 veto)
                break
            key = me_cell(me, hk, obs.shape)
            cc = arch.cells.get(key)
            if cc is None or events > cc["events"] or (events == cc["events"] and depth < cc["depth"]):
                if cc is not None:
                    del arch.cells[key]
                arch.add(key, env.clone(), depth, events)
                arch.cells[key]["frame"] = (obs, me)
                arch.cells[key]["traj"] = bytes(traj)
            else:
                cc["visits"] += 1
    return arch


def explore(env, iters, rng, steps=100, repeat=0.95, on_step=None):
    """Go-Explore phase 1 with the agent's own cells. `on_step(obs, t_global)` lets callers
    (HUD reader, scorer) watch the frames. Returns the archive."""
    arch = Archive()
    obs = env.reset()
    arch.add(cell_of(obs), env.clone(), 0, 0)
    t = 0
    for _ in range(iters):
        k = arch.choose(rng)
        c = arch.cells[k]
        env.restore(c["state"])
        depth, events = c["depth"], c["events"]
        a = int(rng.integers(env.n_actions))
        for _ in range(steps):
            if rng.random() > repeat:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            depth += 1
            t += 1
            ev = on_step(obs, t) if on_step else 0
            events += ev or 0
            if over:
                break
            arch.add(cell_of(obs), env.clone(), depth, events)
    return arch
