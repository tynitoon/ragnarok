"""Who am I and what do my buttons do — from savestate branches only (grants G4, G5).

From one savestate, every action is held for `horizon` agent steps and compared with
NOOP from the same state. The pixels that differ between the two branches and are not
background are the controllable thing: in the action branch it is where I went, in the
NOOP branch where I would have been. The difference of their centroids is the action's
effect. Only screen bytes and the agent's own background estimate are used.
"""
import numpy as np
from scipy import ndimage

NOOP = 0


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _main_component(mask):
    """Keep the largest 8-connected component (drops downstream echoes such as a deflected ball)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, index=range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def probe(env, background, horizon=2):
    """Branch every action vs NOOP from the current state. Returns per-action dicts and the
    NOOP-branch position of the controllable thing (or None). Restores the state after."""
    s = env.clone()
    frames = []
    for a in range(env.n_actions):
        env.restore(s)
        for _ in range(horizon):
            obs, _ = env.step(a)
        frames.append(obs)
    env.restore(s)
    noop = frames[NOOP]
    out, me_pts = [], []
    for a, fa in enumerate(frames):
        if a == NOOP:
            out.append(dict(dx=0.0, dy=0.0, spawn=False, moved=False))
            continue
        diff = (fa != noop).any(-1)
        fg_a = _main_component(diff & (fa != background).any(-1))
        fg_n = _main_component(diff & (noop != background).any(-1))
        ca, cn = _centroid(fg_a), _centroid(fg_n)
        if ca is not None and cn is not None:
            out.append(dict(dx=ca[0] - cn[0], dy=ca[1] - cn[1], spawn=False, moved=True))
            me_pts.append(cn)
        elif ca is not None:             # something appears only when the button is pressed
            out.append(dict(dx=0.0, dy=0.0, spawn=True, moved=False))
        else:
            out.append(dict(dx=0.0, dy=0.0, spawn=False, moved=False))
    me = None
    if me_pts:
        me = (float(np.median([p[0] for p in me_pts])), float(np.median([p[1] for p in me_pts])))
    return out, me


EFFECTS = ["none", "+x", "-x", "+y", "-y", "spawn"]


def effect_table(probes, min_px=0.5):
    """Aggregate probes into one canonical effect per action (median displacement, spawn rate)."""
    n_actions = len(probes[0])
    table = []
    for a in range(n_actions):
        dxs = [p[a]["dx"] for p in probes if p[a]["moved"]]
        dys = [p[a]["dy"] for p in probes if p[a]["moved"]]
        spawn = np.mean([p[a]["spawn"] for p in probes])
        mx = float(np.median(dxs)) if dxs else 0.0
        my = float(np.median(dys)) if dys else 0.0
        if max(abs(mx), abs(my)) >= min_px:
            eff = ("+x" if mx > 0 else "-x") if abs(mx) >= abs(my) else ("+y" if my > 0 else "-y")
        elif spawn > 0.2:
            eff = "spawn"
        else:
            eff = "none"
        table.append(dict(effect=eff, dx=mx, dy=my, spawn_rate=float(spawn), n_moved=len(dxs)))
    return table
