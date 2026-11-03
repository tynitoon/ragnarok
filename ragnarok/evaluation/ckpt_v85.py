"""v85 run infrastructure (docs/ARC4_V85_PROTOCOL_EN.md §2, A.5, §13.2-13.4, A.15). Evaluator side.

- `Checkpointer`: the Explorer2s boundary hook. At an expansion boundary, once at least `every_s` seconds
  (<= 300, §13.3) have passed since the last write, it pickles straight into `<path>.tmp` (streaming, never
  an in-memory copy), fsyncs and `os.replace`s it onto `<path>`. Content (§13.3): explorer.to_state()
  (archive, bandit, hist, coarse_count, best_S triple, any_rise, exp_meta and exp_feat, Body tracker, steer
  and Q5 state), SenseEnv85.to_state(), Trace85.to_state(), the state of every other injected hook object
  that has to_state (the chooser, an `after` object), and the job's own meta. No RNG state: every stream
  is indexed by expansion.
- `load` / `restore`: resume = fresh SenseEnv85 built with a fresh Trace85, env.from_state (reset + NOOP
  inside, after-step context as in a live run), trace.from_state, explorer.from_state, the hook objects'
  from_state, then `ex.run(env, spent=Spent, resume=True)`; the next expansion restores a handle. A kill
  mid-sandbox or mid-write loses only the work since the last checkpoint, which is redone identically (a
  torn `.tmp` is never read).
- `closure_hash` (§13.4): SHA-256 over the sorted (path, file SHA-256) pairs of every project module in
  sys.modules (files under the project root), plus the job script. Project modules (ragnarok.*,
  scripts.*) loaded from outside the root are listed as foreign.
- `peak_mem_mb`, `write_json` (tmp + replace), CLAIM_HEADER and SENSES (§2, verbatim header).
"""
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

CLAIM_HEADER = (
    "Every v85 claim concerns a SEARCH WITH BILLED REWINDS (savestates; 60 frames per restore; every emulated "
    "frame of the agent billed). It starts from a savestate shared by all arms: a state, chosen by the evaluator, "
    "at the game's first matched rung M1 on an earlier V2 run's lineage that reached the top rung. The root's "
    "sensed rise is set to that lineage's own sensed rise and is identical for all arms. It uses the senses "
    "granted on 22 October 2026: pixels, its own buttons, max(0, r_step) per step, lives. Evaluator instruments "
    "(D, O, K*) additionally use unbilled look-ahead or future-path data, reported as oracle frames; they are not "
    "agents. It is not live play without savestates.")
SENSES = dict(
    granted="2026-10-22",
    agent_inputs=["pixels", "own buttons", "max(0, r_step) per step", "lives"],
    root_S="S(s1): the reference lineage's own sensed rise, identical for all arms",
    never_read=["RAM", "true score", "game names", "downloaded weights"],
    ram_read_by_any_v85_component=False,
    instruments="evaluator only (ragnarok/evaluation/), never agents: unbilled look-ahead or future-path data, "
                "reported as oracle frames")
VERSION = 1


def _replace(src, dst, tries=20):
    for t in range(tries):                     # Windows: a scanner may hold the target for a moment
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if t == tries - 1:
                raise
            time.sleep(0.25)


class Checkpointer:
    """Explorer2s boundary hook: `ck(ex, e)`. `attach` it to the run's env, trace and hook objects first."""

    def __init__(self, path, every_s=300.0, key=None):
        assert 0 <= every_s <= 300, "§13.3: a checkpoint at least every 5 min of wall"
        self.path, self.every_s, self.key = pathlib.Path(path), float(every_s), key
        self.t_last = time.monotonic()
        self.writes, self.max_bytes, self.max_s, self.last_exp = 0, 0, 0.0, None
        self.env = self.trace = self.meta = None
        self.objs = {}

    def attach(self, env, trace, objs=None, meta=None):
        """objs: {name: hook object with to_state/from_state} besides the explorer's own steer and q5."""
        self.env, self.trace, self.objs, self.meta = env, trace, dict(objs or {}), meta
        return self

    def __call__(self, ex, e):
        if time.monotonic() - self.t_last >= self.every_s:
            self.write(ex, e)

    def write(self, ex, e):
        t0 = time.perf_counter()
        state = dict(version=VERSION, key=self.key, exp=int(e), explorer=ex.to_state(), env=self.env.to_state(),
                     trace=self.trace.to_state(), hooks={n: o.to_state() for n, o in self.objs.items()},
                     meta=self.meta() if callable(self.meta) else self.meta)
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        del state
        _replace(tmp, self.path)
        self.writes += 1
        self.max_bytes = max(self.max_bytes, self.path.stat().st_size)
        self.max_s = max(self.max_s, time.perf_counter() - t0)
        self.last_exp, self.t_last = int(e), time.monotonic()

    def stats(self):
        return dict(writes=self.writes, max_mb=round(self.max_bytes / 2 ** 20, 3), max_s=round(self.max_s, 3),
                    every_s=self.every_s, last_exp=self.last_exp)

    def remove(self):
        """After the run's JSON is written: the job's own rolling checkpoint (§13.2 disk budget) goes."""
        for p in (self.path, self.path.with_name(self.path.name + ".tmp")):
            if p.exists():
                p.unlink()


def load(path, key=None):
    """The last complete checkpoint at `path`, or None. A key mismatch (another run's file) raises."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        st = pickle.load(f)
    if st.get("version") != VERSION:
        raise ValueError(f"checkpoint version {st.get('version')} != {VERSION}: {path}")
    if key is not None and st.get("key") != key:
        raise ValueError(f"checkpoint {path} belongs to another run: {st.get('key')} != {key}")
    return st


def restore(st, env, trace, ex, objs=None):
    """§13.3 resume on a FRESH SenseEnv85 constructed with the fresh `trace` (in that order: the env's
    constructor opens a base on the trace, which trace.from_state then overwrites); objs as in attach."""
    env.from_state(st["env"])
    trace.from_state(st["trace"])
    ex.from_state(st["explorer"])
    for n, o in (objs or {}).items():
        o.from_state(st["hooks"][n])


# ---------------------------------------------------------------- code identity (§13.4)
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def closure_hash(root, extra=()):
    """(hex, [(relative path, sha256)], foreign): every project module in sys.modules with a file under
    `root`, plus `extra` files (the job script). `foreign`: ragnarok.* / scripts.* modules from elsewhere."""
    root = pathlib.Path(root).resolve()
    files, foreign = set(), []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        if root in p.parents and p.suffix == ".py":
            files.add(p)
        elif name.split(".")[0] in ("ragnarok", "scripts"):
            foreign.append(f"{name}: {p}")
    files.update(pathlib.Path(x).resolve() for x in extra)
    pairs = sorted((p.relative_to(root).as_posix(), _sha(p)) for p in files)
    h = hashlib.sha256("\n".join(f"{a} {b}" for a, b in pairs).encode()).hexdigest()
    return h, pairs, sorted(foreign)


# ---------------------------------------------------------------- memory and JSON
def peak_mem_mb():
    """(peak working set, peak commit) of this process in MB (Windows; (0, 0) elsewhere)."""
    if os.name != "nt":
        return 0.0, 0.0
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20, c.PeakPagefileUsage / 2 ** 20


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path, obj):
    """Atomic: tmp + replace. Deterministic bytes for equal content (insertion order, indent 1)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_np)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)
