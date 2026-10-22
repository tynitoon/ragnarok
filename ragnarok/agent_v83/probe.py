"""Who am I and what do my buttons do, from savestate branches only (port of agent_v71.me_buttons.probe
to the v83 senses: step returns (obs, rise, lives, end); the probe's handle is freed afterwards).

From one savestate, every button is held for `horizon` steps and compared with NOOP from the same
state. The pixels that differ between the two branches and are not background are the controllable
thing: in the button branch where I went, in the NOOP branch where I would have been.
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


def branches(env, horizon):
    """The frame after holding each button for `horizon` steps from the current state; state restored."""
    s = env.clone()
    frames = []
    for a in range(env.n_actions):
        env.restore(s)
        for _ in range(horizon):
            obs = env.step(a)[0]
        frames.append(obs)
    env.restore(s)
    env.free(s)
    return frames


def probe(env, background, horizon=2):
    """Per-button dicts and the NOOP-branch position of the controllable thing (or None)."""
    frames = branches(env, horizon)
    noop = frames[NOOP]
    out, me_pts = [], []
    colours = set()
    reveal_mask = np.zeros(noop.shape[:2], bool)
    reveal_val = np.zeros_like(noop)
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
            px = noop[fg_n]
            colours.update(((px[:, 0].astype(int) << 16) | (px[:, 1].astype(int) << 8) | px[:, 2]).tolist())
            # where I stand under NOOP, the button branch (I moved away) shows the scenery behind me
            behind = (fa != noop).any(-1) & _main_component(diff & (noop != fa).any(-1)) & ~fg_a
            reveal_mask |= behind
            reveal_val[behind] = fa[behind]
        elif ca is not None:             # something appears only when the button is pressed
            out.append(dict(dx=0.0, dy=0.0, spawn=True, moved=False))
        else:
            out.append(dict(dx=0.0, dy=0.0, spawn=False, moved=False))
    me = None
    if me_pts:
        me = (float(np.median([p[0] for p in me_pts])), float(np.median([p[1] for p in me_pts])))
    probe.last_reveal = (reveal_mask, reveal_val)
    probe.last_colours = colours          # my colours under NOOP (per-game appearance, never carried)
    return out, me
