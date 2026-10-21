"""Objects on the screen, from pixels only (agent side, docs/ARC4_V80_DESIGN.md §3).

A frame becomes a list of objects: 8-connected components of the foreground (pixels that differ
from the agent's own background estimate), the agent's counter and gauge boxes masked, components
larger than `max_area` of the screen dropped (playfields, not objects). Each object keeps its
centre, box, area and colour set. A tracker links objects through time (Hungarian assignment on
predicted positions, a gate in pixels, a few steps of coasting) and gives each track a velocity.
Nothing here reads RAM, the score or the emulator: screen bytes and the agent's own estimates only.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

EIGHT = np.ones((3, 3), bool)


def pack(rgb):
    return (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]


class Parser:
    """Screen -> objects. `boxes`: (y0, y1, x0, x1) regions the agent treats as displays."""

    def __init__(self, bg, boxes=(), max_area=0.05, max_obj=48, min_px=2):
        self.bg = bg
        H, W = bg.shape[:2]
        self.keep = np.ones((H, W), bool)
        for y0, y1, x0, x1 in boxes:
            self.keep[y0:y1, x0:x1] = False
        self.max_px = max_area * H * W
        self.max_obj = max_obj
        self.min_px = min_px

    def parse(self, obs):
        fg = (obs != self.bg).any(-1) & self.keep
        lab, n = ndimage.label(fg, structure=EIGHT)
        if n == 0:
            return []
        packed = pack(obs)
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab)):
            m = lab[sl] == i + 1
            a = int(m.sum())
            if a < self.min_px or a > self.max_px:
                continue
            ys, xs = np.nonzero(m)
            cols, cnt = np.unique(packed[sl][m], return_counts=True)
            out.append(dict(cx=float(xs.mean() + sl[1].start), cy=float(ys.mean() + sl[0].start),
                            x0=sl[1].start, x1=sl[1].stop, y0=sl[0].start, y1=sl[0].stop, area=a,
                            colour=int(cols[np.argmax(cnt)]), colours=frozenset(int(c) for c in cols)))
        if len(out) > self.max_obj:                  # keep the largest: the rest is debris
            out.sort(key=lambda o: -o["area"])
            out = out[:self.max_obj]
        return out


class Tracker:
    """Objects -> tracks. Each track: id, position, velocity (px/step), size, colour, age, missed
    steps. Prediction for matching is constant velocity unless a `predict(track) -> (x, y)` is
    given (the theory's own motion rule, once it has one)."""

    def __init__(self, gate=16.0, coast=4):
        self.gate = gate
        self.coast = coast
        self.tracks = {}
        self.next_id = 0

    def reset(self):
        self.tracks = {}

    def _cost(self, t, o, px, py):
        d = abs(o["cx"] - px) + abs(o["cy"] - py)
        d += 4.0 * abs(np.log((o["area"] + 1) / (t["area"] + 1)))
        d += 0.0 if o["colour"] == t["colour"] else 6.0
        return d

    def step(self, objs, predict=None):
        """Returns the list of (track_id, object) matched or born this step."""
        ids = list(self.tracks)
        preds = []
        for k in ids:
            t = self.tracks[k]
            preds.append(predict(t) if predict else (t["cx"] + t["vx"], t["cy"] + t["vy"]))
        out, used_o, used_t = [], set(), set()
        if ids and objs:
            C = np.array([[self._cost(self.tracks[k], o, *preds[i]) for o in objs] for i, k in enumerate(ids)])
            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                if C[r, c] > self.gate:
                    continue
                k, o = ids[r], objs[c]
                t = self.tracks[k]
                vx, vy = o["cx"] - t["cx"], o["cy"] - t["cy"]
                if t["missed"]:                      # spread the jump over the steps it was unseen
                    vx, vy = vx / (t["missed"] + 1), vy / (t["missed"] + 1)
                t.update(cx=o["cx"], cy=o["cy"], vx=vx, vy=vy, area=o["area"], colour=o["colour"],
                         box=(o["x0"], o["y0"], o["x1"], o["y1"]), age=t["age"] + 1, missed=0, born=False)
                out.append((k, o)); used_o.add(c); used_t.add(k)
        for k in ids:
            if k not in used_t:
                t = self.tracks[k]
                t["missed"] += 1
                t["cx"], t["cy"] = t["cx"] + t["vx"], t["cy"] + t["vy"]     # coast
                if t["missed"] > self.coast:
                    del self.tracks[k]
        for c, o in enumerate(objs):
            if c not in used_o:
                k = self.next_id; self.next_id += 1
                self.tracks[k] = dict(id=k, cx=o["cx"], cy=o["cy"], vx=0.0, vy=0.0, area=o["area"],
                                      colour=o["colour"], box=(o["x0"], o["y0"], o["x1"], o["y1"]),
                                      age=0, missed=0, born=True)
                out.append((k, o))
        return out

    def visible(self):
        return [t for t in self.tracks.values() if t["missed"] == 0]


def mark_me(tracks, me, radius=6.0):
    """The track nearest to my own tracked position (the v71 probe-and-colour tracker), if close."""
    if me is None or not tracks:
        return None
    d = [abs(t["cx"] - me[0]) + abs(t["cy"] - me[1]) for t in tracks]
    i = int(np.argmin(d))
    return tracks[i]["id"] if d[i] <= radius else None
