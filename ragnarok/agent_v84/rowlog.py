"""V2 row logs (docs/ARC4_V84_PROTOCOL_EN.md §5 Logs). One promise row every 4th rollout step, in the v83
format with t and avail as uint16 and L_t (the planned rollout length at step t); written to disk in
chunks of 8,192 labelled rows. Labels rule: a row is kept for horizon H iff t + H <= L_t and (avail >= H
or (hit and stops_on_event)); keeping never depends on the outcome otherwise."""
import pathlib

import numpy as np

from ragnarok.agent_v83.views import WIN

NONE = 255
CHUNK = 8192
FIELDS = dict(view=(np.uint8, (4 * WIN * WIN // 8,)), screen=(np.uint8, (8, 8)), lost=(np.bool_, ()),
              mx=(np.float32, ()), my=(np.float32, ()), lives=(np.int16, ()), slot=(np.int16, ()),
              button=(np.int16, ()), it=(np.int32, ()), rise_lag=(np.uint8, ()), drop_lag=(np.uint8, ()),
              avail=(np.uint16, ()), t=(np.uint16, ()), L_t=(np.uint16, ()), miss=(np.uint8, ()))


def labels_L(lag, avail, t, L_t, horizon, stops_on_event=False):
    """1 / 0 / -1 (censored) with the planned length L_t in place of v83's fixed 100 steps."""
    lag, avail, t, L_t = (np.asarray(v, np.int64) for v in (lag, avail, t, L_t))
    hit = lag < horizon
    keep = (t + horizon <= L_t) & ((avail >= horizon) | (hit & stops_on_event))
    return np.where(keep, hit.astype(np.int64), -1)


class DiskRowLog:
    def __init__(self, directory):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending = []           # rows of the current rollout (dicts)
        self.ready = []             # labelled rows waiting for a chunk
        self.n_chunks = 0
        self.n_rows = 0

    def add(self, **row):
        self.pending.append(row)

    def end_rollout(self, rises, drops):
        """rises[j] / drops[j]: the senses after the j-th step's button. Fill lags and avail, move to ready."""
        r = np.nonzero(rises)[0]
        d = np.nonzero(drops)[0]
        n = len(rises)
        for row in self.pending:
            t = int(row["t"])
            ri, di = r[r >= t], d[d >= t]
            row["rise_lag"] = min(NONE, int(ri[0] - t)) if len(ri) else NONE
            row["drop_lag"] = min(NONE, int(di[0] - t)) if len(di) else NONE
            row["avail"] = max(0, n - t)
            self.ready.append(row)
        self.pending = []
        if len(self.ready) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.ready:
            return
        arr = {k: np.array([row[k] for row in self.ready], dtype=dt) for k, (dt, _) in FIELDS.items()}
        np.savez_compressed(self.dir / f"rows_{self.n_chunks:04d}.npz", **arr)
        self.n_chunks += 1
        self.n_rows += len(self.ready)
        self.ready = []

    def close(self, rises=None, drops=None):
        if self.pending and rises is not None:
            self.end_rollout(rises, drops)
        self.flush()
        return self.n_rows
