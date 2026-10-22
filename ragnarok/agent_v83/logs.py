"""Explorer logs. RowLog: one promise row every 4th rollout step (packed view, slot, button, and the
senses' outcome lags: steps to the first rise and to the first lives decrease from that step within the
rollout, 255 = none, the steps available after it, the rollout step t and the tracker's miss count).
`labels` gives a horizon H a label without selecting rows on their outcome. IterLog: per iteration
billed frames, best S, cells, probe frames, lost steps, how the rollout stopped."""
import numpy as np

from ragnarok.agent_v83.views import WIN

NONE = 255
STOPS = ("length", "drop", "end")


class _Chunks:
    def __init__(self, fields, chunk=8192):
        self.fields, self.chunk = fields, chunk
        self.done = {k: [] for k in fields}
        self.cur = None
        self.n = 0

    def _new(self):
        self.cur = {k: np.zeros((self.chunk,) + shp, dt) for k, (dt, shp) in self.fields.items()}

    def add(self, **vals):
        j = self.n % self.chunk
        if j == 0:
            if self.cur is not None:
                for k in self.fields:
                    self.done[k].append(self.cur[k])
            self._new()
        for k, v in vals.items():
            self.cur[k][j] = v
        self.n += 1
        return self.n - 1

    def set(self, i, k, v):
        c, j = divmod(i, self.chunk)
        arr = self.cur[k] if c == len(self.done[k]) else self.done[k][c]
        arr[j] = v

    def arrays(self):
        out = {}
        for k in self.fields:
            parts = list(self.done[k])
            if self.cur is not None:
                parts.append(self.cur[k])
            out[k] = np.concatenate(parts)[: self.n] if parts else np.zeros((0,) + self.fields[k][1], self.fields[k][0])
        return out


class RowLog(_Chunks):
    def __init__(self):
        super().__init__(dict(view=(np.uint8, (4 * WIN * WIN // 8,)), screen=(np.uint8, (8, 8)),
                              lost=(np.bool_, ()), mx=(np.float32, ()), my=(np.float32, ()),
                              lives=(np.int16, ()), slot=(np.int16, ()), button=(np.int16, ()),
                              it=(np.int32, ()), rise_lag=(np.uint8, ()), drop_lag=(np.uint8, ()),
                              avail=(np.uint8, ()), t=(np.uint8, ()), miss=(np.uint8, ())))

    def add(self, packed, screen, lost, me, lives, slot, button, it, t, miss):
        mx, my = (np.nan, np.nan) if me is None else me
        return super().add(view=packed, screen=screen, lost=lost, mx=mx, my=my, lives=lives, slot=slot,
                           button=button, it=it, t=t, miss=min(255, int(miss)),
                           rise_lag=NONE, drop_lag=NONE, avail=0)

    def get(self, i):
        c, j = divmod(i, self.chunk)
        src = self.cur if c == len(self.done["it"]) else {k: v[c] for k, v in self.done.items()}
        return {k: src[k][j] for k in self.fields}

    def label(self, row_ids, rises, drops):
        """Rows at rollout steps t; rises[j]/drops[j] are the senses after the j-th step's button."""
        if not row_ids:
            return
        r = np.nonzero(rises)[0]
        d = np.nonzero(drops)[0]
        n = len(rises)
        for t, i in row_ids:
            ri = r[r >= t]
            di = d[d >= t]
            self.set(i, "rise_lag", min(NONE, int(ri[0] - t)) if len(ri) else NONE)
            self.set(i, "drop_lag", min(NONE, int(di[0] - t)) if len(di) else NONE)
            self.set(i, "avail", max(0, n - t))


def labels(lag, avail, t, horizon, steps=100, stops_on_event=False):
    """1 / 0 / -1 (censored). A row is kept only if its window [t, t+H) fits the planned rollout
    (t + H <= steps) and was fully observed (avail >= H), or, for an event that itself stops the
    rollout (the lives decrease: stops_on_event=True), if that event ended the window. Keeping a row
    never depends on its outcome otherwise (review A-1)."""
    lag, avail, t = (np.asarray(v, int) for v in (lag, avail, t))
    hit = lag < horizon
    keep = (t + horizon <= steps) & ((avail >= horizon) | (hit & stops_on_event))
    return np.where(keep, hit.astype(int), -1)


class IterLog(_Chunks):
    def __init__(self):
        super().__init__(dict(billed=(np.int64, ()), best_S=(np.float32, ()), cells=(np.int32, ()),
                              probe_frames=(np.int64, ()), lost_steps=(np.int16, ()), stop=(np.int8, ())),
                         chunk=4096)

    def add(self, billed, best_S, cells, probe_frames, lost_steps, stop):
        return super().add(billed=billed, best_S=best_S, cells=cells, probe_frames=probe_frames,
                           lost_steps=lost_steps, stop=STOPS.index(stop))
