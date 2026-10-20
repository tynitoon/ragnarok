"""Counters on the screen, found without OCR, labels or RAM (grants G1, G2).

A display counter is a set of pixels that rarely change and, when they do, keep the new
value: unlike sprites passing over a pixel, a digit that changes stays changed. Regions of
such pixels are candidate counters. Each region's content is hashed per frame; a change of
hash that holds for `hold` steps is a counter EVENT.

Roles without reading values (G2 / G1 as innate valences):
  score  = the counter region with the most distinct states (it keeps counting);
  lives  = a counter region with few states (<= 6) that changes rarely; its change is a death.
Values (the odometer reading) are a later refinement; the drives need only events.
"""
import numpy as np
from scipy import ndimage


def _pack(rgb):
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]


class CounterFinder:
    """Streaming statistics over ordinary play; `fit()` then turns them into regions."""

    def __init__(self, shape, lag=4, bin_len=50, max_bins=1024):
        self.lag = lag
        self.ring = []
        self.n_change = np.zeros(shape, np.int32)
        self.n_persist = np.zeros(shape, np.int32)
        self.t = 0
        self.bin_len = bin_len
        self.bins = np.zeros((max_bins,) + tuple(shape), np.uint8)   # persistent changes per time bin

    def update(self, rgb):
        p = _pack(rgb)
        self.ring.append(p)
        if len(self.ring) > self.lag + 2:
            self.ring.pop(0)
        if len(self.ring) == self.lag + 2:
            prev, cur, later = self.ring[0], self.ring[1], self.ring[-1]
            ch = cur != prev
            if ch.mean() < 0.3:          # scene-wide flashes / blank frames are not counter changes
                self.n_change += ch
                pers = ch & (later == cur)
                self.n_persist += pers
                b = self.t // self.bin_len
                if b < self.bins.shape[0]:
                    self.bins[b] += pers
        self.t += 1

    def fit(self, max_rate=0.03, min_px=4, max_area=0.05, corr_min=0.3, pix_corr_min=0.3, min_persist=5, min_shared=3):
        """Counter pixels: changes are rare and mostly persistent. Regions larger than
        `max_area` of the screen are playfields (bricks, invaders), not displays."""
        T = max(1, self.t)
        # only persistent changes count: a blinking digit flips often but changes value rarely
        hud = (self.n_persist >= min_persist) & (self.n_persist < max_rate * T)
        H, W = hud.shape
        # Group candidate pixels that are near each other AND change at the same moments: one
        # counter's pixels change together at its events; a moving formation changes at its own
        # steps. Union-find over neighbour offsets, edge iff binned change series correlate.
        nb = min(self.bins.shape[0], T // self.bin_len + 1)
        A = self.bins[:nb] > 0                    # bins where the pixel changed persistently
        X = self.bins[:nb].astype(np.float32)
        X -= X.mean(0, keepdims=True)
        norm = np.sqrt((X * X).sum(0)) + 1e-6
        parent = np.arange(H * W)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        ys, xs = np.nonzero(hud)
        cand = set(zip(ys.tolist(), xs.tolist()))
        for dy, dx in [(0, 1), (0, 2), (0, 3), (0, 4), (1, -1), (1, 0), (1, 1), (2, 0)]:
            for y, x in zip(ys.tolist(), xs.tolist()):
                y2, x2 = y + dy, x + dx
                if (y2, x2) not in cand:
                    continue
                c = float((X[:, y, x] * X[:, y2, x2]).sum() / (norm[y, x] * norm[y2, x2]))
                shared = int(((A[:, y, x]) & (A[:, y2, x2])).sum())
                if c > pix_corr_min and shared >= min_shared:
                    a, b = find(y * W + x), find(y2 * W + x2)
                    if a != b:
                        parent[a] = b
        groups = {}
        for y, x in zip(ys.tolist(), xs.tolist()):
            groups.setdefault(find(y * W + x), []).append((y, x))
        def merge_lines(gl):
            box = [[min(p[0] for p in g), max(p[0] for p in g) + 1, min(p[1] for p in g), max(p[1] for p in g) + 1] for g in gl]
            ser = [X[:, [p[0] for p in g], [p[1] for p in g]].sum(1) for g in gl]
            merged = True
            gap = 5
            while merged:
                merged = False
                for i in range(len(gl)):
                    for j in range(i + 1, len(gl)):
                        if gl[i] is None or gl[j] is None:
                            continue
                        bi, bj = box[i], box[j]
                        ov = min(bi[1], bj[1]) - max(bi[0], bj[0])
                        near = (ov >= 0.5 * min(bi[1] - bi[0], bj[1] - bj[0]) and
                                bi[2] - gap < bj[3] and bj[2] - gap < bi[3])
                        if not near:
                            continue
                        si, sj = ser[i], ser[j]
                        c = float((si * sj).sum() / (np.sqrt((si * si).sum() * (sj * sj).sum()) + 1e-6))
                        if c > corr_min:
                            gl[i] = gl[i] + gl[j]; gl[j] = None
                            box[i] = [min(bi[0], bj[0]), max(bi[1], bj[1]), min(bi[2], bj[2]), max(bi[3], bj[3])]
                            ser[i] = si + sj
                            merged = True
            return [g for g in gl if g is not None]

        # second pass: merge groups on one text line whose change series correlate
        gl = merge_lines(list(groups.values()))
        groups = {k: g for k, g in enumerate(gl)}
        # third pass: a group too large to be a display is split by clustering its pixels' change
        # series (k-means), then by space within each cluster (text lines vs moving formations)
        out_groups = []
        for pts in groups.values():
            py = np.array([p[0] for p in pts]); px = np.array([p[1] for p in pts])
            area = (py.max() - py.min() + 1) * (px.max() - px.min() + 1)
            if area <= max_area * H * W:
                out_groups.append(pts)
                continue
            S = X[:, py, px].T
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)
            rng = np.random.default_rng(0)
            k = min(6, len(pts))
            C = S[rng.choice(len(pts), k, replace=False)]
            for _ in range(15):
                lab = np.argmax(S @ C.T, axis=1)
                for j in range(k):
                    if (lab == j).any():
                        c = S[lab == j].mean(0)
                        C[j] = c / (np.linalg.norm(c) + 1e-6)
            for j in range(k):
                m = np.zeros((H, W), bool)
                m[py[lab == j], px[lab == j]] = True
                sl, nsl = ndimage.label(ndimage.binary_dilation(m, structure=np.ones((3, 7), bool)))
                for kk in range(1, nsl + 1):
                    sel = (sl == kk) & m
                    yy, xx = np.nonzero(sel)
                    out_groups.append(list(zip(yy.tolist(), xx.tolist())))
        out_groups = merge_lines(out_groups)       # re-join line fragments produced by the split
        regions = []
        for pts in out_groups:
            py = [p[0] for p in pts]; px = [p[1] for p in pts]
            y0, y1, x0, x1 = min(py), max(py) + 1, min(px), max(px) + 1
            if len(pts) >= min_px and (y1 - y0) * (x1 - x0) <= max_area * H * W:
                regions.append((y0, y1, x0, x1))
        return sorted(regions)


class HudReader:
    def __init__(self, regions, hold=2):
        self.regions = regions
        self.hold = hold
        self.stable = [None] * len(regions)
        self.pending = [None] * len(regions)
        self.count = [0] * len(regions)
        self.states = [set() for _ in regions]
        self.events = [0] * len(regions)
        self._game_events = [0] * len(regions)
        self._game_seen = [set() for _ in regions]
        self.novel_in_game = [0] * len(regions)
        self.raw_changes = [0] * len(regions)
        self._prev_crop = [None] * len(regions)
        self.before = [None] * len(regions)
        self._raw_last = [None] * len(regions)
        self.steps = 0
        self.per_game = []
        self.role = {}

    def _hash(self, rgb, r):
        y0, y1, x0, x1 = r
        return hash(rgb[y0:y1, x0:x1].tobytes())

    def _blink_hash(self, rgb, i):
        """Hash of the pixelwise max of this and the previous crop: a blinking glyph reads as itself."""
        y0, y1, x0, x1 = self.regions[i]
        crop = rgb[y0:y1, x0:x1]
        # SHAPE, not colour: pixels differing from the crop's dominant colour (colour cycling and
        # palette animations of a display do not change what it shows)
        flat = crop.reshape(-1, 3)
        packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
        vals, counts = np.unique(packed, return_counts=True)
        mask = (packed != vals[np.argmax(counts)]).reshape(crop.shape[:2])
        prev = self._prev_crop[i]
        self._prev_crop[i] = mask
        if prev is not None and prev.shape == mask.shape:
            mask = mask | prev
        return hash(np.packbits(mask).tobytes())

    def step(self, rgb):
        """Returns the list of region indices whose content changed stably at this step."""
        out = []
        self.steps += 1
        for i, r in enumerate(self.regions):
            h = self._blink_hash(rgb, i)
            last = self._raw_last[i]
            # a 2-cycle (blink) is not a change: count only hashes differing from the last two
            if last is not None and h != last[0] and h != last[1]:
                self.raw_changes[i] += 1
            self._raw_last[i] = (h, last[0] if last else None)
            if self.stable[i] is None:
                self.stable[i] = h
                self.states[i].add(h)
                self._game_seen[i].add(h)
                continue
            if h == self.stable[i]:
                self.pending[i], self.count[i] = None, 0
                continue
            if h == self.pending[i]:
                self.count[i] += 1
            else:
                self.pending[i], self.count[i] = h, 1
            if self.count[i] >= self.hold:
                if h == self.before[i]:           # A -> B -> A: a blink, not a count
                    self.before[i], self.stable[i] = self.stable[i], h
                    self.pending[i], self.count[i] = None, 0
                    continue
                self.before[i] = self.stable[i]
                self.stable[i] = h
                self.states[i].add(h)
                self.events[i] += 1
                self._game_events[i] += 1
                self.novel_in_game[i] += int(h not in self._game_seen[i])
                self._game_seen[i].add(h)
                self.pending[i], self.count[i] = None, 0
                out.append(i)
        return out

    def reset_tracking(self):
        """Call at a full game restart (the console's game over -> new game, which the agent
        perceives as every counter snapping back). Not an event; closes a per-game tally."""
        if any(c is not None for c in self.stable):
            self.per_game.append(list(self._game_events))
        self._game_events = [0] * len(self.regions)
        self._game_seen = [set() for _ in self.regions]
        self.before = [None] * len(self.regions)
        self._prev_crop = [None] * len(self.regions)
        self.stable = [None] * len(self.regions)
        self.pending = [None] * len(self.regions)
        self.count = [0] * len(self.regions)

    def assign_roles(self, min_games=3, max_rate=0.05):
        """lives = a counter that changes, per game, a nearly constant number of times (2-9)
        and has few states; score = among the others, the counter with the most states.
        Needs >= `min_games` completed games; else only a score is guessed."""
        role = {}
        # a counter is MONOTONE within a game: almost every change shows a state not yet seen in
        # that game (scores count up, lives count down); a sprite passing over a region revisits
        # the same images
        mono = [self.novel_in_game[i] / self.events[i] if self.events[i] else 0.0
                for i in range(len(self.regions))]
        # a display changes rarely as a whole; a lane of cars or a ball's playfield changes all the time
        rate = [self.raw_changes[i] / max(1, self.steps) for i in range(len(self.regions))]
        is_counter = [self.events[i] >= 3 and mono[i] >= 0.9 and rate[i] < max_rate for i in range(len(self.regions))]
        self.rate = rate
        self.monotone = mono
        games = np.array(self.per_game, float) if len(self.per_game) >= min_games else None
        lives = None
        if games is not None and games.size:
            for i in range(len(self.regions)):
                if not is_counter[i]:
                    continue
                c = games[:, i]
                m = c.mean()
                if 2 <= m <= 9 and c.std() <= 0.25 * m and len(self.states[i]) <= 10:
                    if lives is None or c.std() < games[:, lives].std():
                        lives = i
        if lives is not None:
            role[lives] = "lives"
        rest = [i for i in range(len(self.regions)) if i != lives and is_counter[i]]
        if rest:
            role[max(rest, key=lambda i: self.events[i])] = "score"
        self.role = role
        return role
