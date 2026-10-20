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
        self.colours = None          # set by the probe: track me by my own colours, not vs background
        self.pos = None
        self.size = None
        self.max_jump = max_jump
        self.lost_after = lost_after
        self.miss = 0

    def seed(self, pos, frame):
        self.pos = pos
        lab, n = ndimage.label(self._mask(frame, self.bg))
        if n:
            # nearest component to the probed centre (sprites are sparse; the centre may be a hole)
            best, bd = None, 1e9
            for i, sl in enumerate(ndimage.find_objects(lab)):
                m = lab[sl] == (i + 1)
                ys, xs = np.nonzero(m)
                d = abs(xs.mean() + sl[1].start - pos[0]) + abs(ys.mean() + sl[0].start - pos[1])
                if d < bd:
                    best, bd = float(m.sum()), d
            if best is not None and bd < 12:
                self.size = best
        self.miss = 0

    def update(self, frame, win=28):
        """Search only a window around the last position (me moves a few px per step)."""
        if self.pos is None:
            return None
        H, W = frame.shape[:2]
        x0, x1 = max(0, int(self.pos[0]) - win), min(W, int(self.pos[0]) + win + 1)
        y0, y1 = max(0, int(self.pos[1]) - win), min(H, int(self.pos[1]) + win + 1)
        fg = self._mask(frame[y0:y1, x0:x1], self.bg[y0:y1, x0:x1])
        lab, n = ndimage.label(fg)
        if n == 0:
            self.miss += 1
            return None
        objs = ndimage.find_objects(lab)
        best, bd = None, 1e9
        for i, sl in enumerate(objs):
            m = lab[sl] == (i + 1)
            s = float(m.sum())
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean() + sl[0].start + y0, xs.mean() + sl[1].start + x0
            d = abs(cx - self.pos[0]) + abs(cy - self.pos[1])
            if self.size:
                d += 4.0 * abs(np.log((s + 1) / (self.size + 1)))
            if d < bd:
                best, bd = (cx, cy, s), d
        if best is None or bd > self.max_jump:
            self.miss += 1
            return None
        self.pos = (best[0], best[1])
        if self.size is None:                    # first sighting: learn my size
            self.size = best[2]
        self.miss = 0
        return self.pos

    def reacquire(self, frame):
        """Live mode (no savestates): search the whole frame for the component whose size best
        matches the learned size of me, nearest to the last known position."""
        if self.size is None:
            return None
        fg = self._mask(frame, self.bg)
        lab, n = ndimage.label(fg)
        best, bd = None, 1e9
        for i, sl in enumerate(ndimage.find_objects(lab)):
            m = lab[sl] == (i + 1)
            s = float(m.sum())
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
            d = 40.0 * abs(np.log((s + 1) / (self.size + 1)))
            if self.pos is not None:
                d += 0.1 * (abs(cx - self.pos[0]) + abs(cy - self.pos[1]))
            if d < bd:
                best, bd = (cx, cy), d
        if best is not None and bd < 30:
            self.pos, self.miss = best, 0
        return self.pos

    def _mask(self, frame, bg):
        if self.colours:
            packed = (frame[..., 0].astype(np.int64) << 16) | (frame[..., 1].astype(np.int64) << 8) | frame[..., 2]
            return np.isin(packed, self._cols) & (frame != bg).any(-1)
        return (frame != bg).any(-1)

    def set_colours(self, colours):
        """Keep my colours that are rare in the background (a colour shared with the scenery is useless)."""
        bgp = (self.bg[..., 0].astype(np.int64) << 16) | (self.bg[..., 1].astype(np.int64) << 8) | self.bg[..., 2]
        vals, cnt = np.unique(bgp, return_counts=True)
        common = set(vals[cnt > 0.02 * bgp.size].tolist())
        keep = [c for c in colours if c not in common]
        if keep:
            self.colours = set(keep)
            self._cols = np.array(sorted(self.colours), np.int64)

    @property
    def lost(self):
        return self.pos is None or self.miss >= self.lost_after
