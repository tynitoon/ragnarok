"""Cheap per-step tracking of 'me' once the probes have found it.

After a probe localises the controllable thing, each step picks the foreground component
(pixels differing from the agent's own background estimate) nearest to the last position
with a similar size. If nothing plausible is found for `lost_after` steps, the caller re-probes.
"""
import numpy as np
from scipy import ndimage


class MeTracker:
    def __init__(self, background, max_jump=24, lost_after=8):
        self.bg = background
        self.pos = None
        self.size = None
        self.max_jump = max_jump
        self.lost_after = lost_after
        self.miss = 0

    def seed(self, pos, frame):
        self.pos = pos
        lab, n = ndimage.label((frame != self.bg).any(-1))
        if n:
            x, y = int(round(pos[0])), int(round(pos[1]))
            l = lab[min(max(y, 0), lab.shape[0] - 1), min(max(x, 0), lab.shape[1] - 1)]
            if l:
                self.size = float((lab == l).sum())
        self.miss = 0

    def update(self, frame):
        if self.pos is None:
            return None
        fg = (frame != self.bg).any(-1)
        lab, n = ndimage.label(fg)
        if n == 0:
            self.miss += 1
            return None
        idx = np.arange(1, n + 1)
        com = ndimage.center_of_mass(fg, lab, idx)
        sizes = ndimage.sum(fg, lab, idx)
        best, bd = None, 1e9
        for (cy, cx), s in zip(com, sizes):
            d = abs(cx - self.pos[0]) + abs(cy - self.pos[1])
            if self.size:
                d += 4.0 * abs(np.log((s + 1) / (self.size + 1)))
            if d < bd:
                best, bd = (cx, cy, s), d
        if best is None or bd > self.max_jump:
            self.miss += 1
            return None
        self.pos = (best[0], best[1])
        self.miss = 0
        return self.pos

    @property
    def lost(self):
        return self.pos is None or self.miss >= self.lost_after
