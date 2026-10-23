"""v84 G-build tests 1-6 (docs/ARC4_V84_PROTOCOL_EN.md §5): isolation, ledger, scripted trajectory,
label causality, prefix identity, determinism."""
import ast
import pathlib
import re
import subprocess
import sys

import numpy as np

from ragnarok.agent_v83.logs import labels as labels_v83
from ragnarok.agent_v84.rowlog import labels_L
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, EXCLUDED, RESERVED, SEALED, SECOND
from ragnarok.evaluation.senses_v83 import SenseEnv
from ragnarok.evaluation.senses_v84 import LOAD_FRAMES, SenseEnv84, Spent, Trace84

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_DIRS = [ROOT / "ragnarok" / "agent_v84", ROOT / "ragnarok" / "agent_v83"]
FILES = sorted(f for d in AGENT_DIRS for f in d.glob("*.py"))
BAD_ATTR = {"ram", "getRAM", "true_score", "game", "ale", "lives", "actions", "cloneState", "restoreState",
            "getScreenRGB", "getMinimalActionSet"}
BAD_NAMES = {"getattr", "setattr", "vars", "eval", "exec", "globals", "__import__", "compile", "open"}
ALLOWED = {"numpy", "scipy", "scipy.ndimage", "scipy.optimize", "scipy.stats", "zlib", "math", "collections",
           "heapq", "json", "time", "pathlib", "ragnarok.agent_v71.tracker", "scripts.ref_extract_v70"}


# ---------------------------------------------------------------- 1. isolation
def test_1_isolation_imports_attributes_names():
    names = DEV + CALIBRATION + SEALED + SECOND + RESERVED + EXCLUDED
    pat = re.compile(r"\b(" + "|".join(names) + r")\b", re.I)
    for f in FILES:
        text = f.read_text(encoding="utf-8")
        assert not pat.search(text), f.name
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mods:
                    assert m in ALLOWED or m.startswith("ragnarok.agent_v83") or m.startswith("ragnarok.agent_v84"), (f.name, m)
            if isinstance(node, ast.Attribute):
                own = isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")
                assert own or node.attr not in BAD_ATTR, (f.name, node.lineno, node.attr)
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    assert own, (f.name, node.lineno, node.attr)
            if isinstance(node, ast.Name):
                assert node.id not in BAD_NAMES, (f.name, node.lineno, node.id)


def test_1_isolation_import_loads_nothing_forbidden():
    mods = ", ".join(f"ragnarok.{f.parent.name}.{f.stem}" for f in FILES if f.stem != "__init__")
    code = (f"import sys, {mods}\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'ale_py') or m.startswith('ragnarok.evaluation')]\n"
            "print(bad); assert not bad\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- 2. ledger
def test_2_ledger_monotone_restore_billed():
    env = SenseEnv84("breakout", seed=9700)
    rng = np.random.default_rng(0)
    seen, hs = [env.billed], []
    for t in range(500):
        env.step(int(rng.integers(env.n_actions)))
        if t % 40 == 0:
            hs.append(env.clone())
        if t % 60 == 59:
            b = env.billed
            env.restore(hs[int(rng.integers(len(hs)))])
            assert env.billed == b + LOAD_FRAMES
        seen.append(env.billed)
    assert all(b >= a for a, b in zip(seen, seen[1:]))


# ---------------------------------------------------------------- 3. scripted trajectory
def test_3_scripted_trajectory_v83_v84_identical():
    """The same buttons and restores give the same ledger, true score at clones and (L_t = 100) labels."""
    rng = np.random.default_rng(3)
    plan = [(int(rng.integers(4)), bool(rng.random() < 0.05)) for _ in range(800)]
    out = []
    for cls in (SenseEnv, SenseEnv84):
        env = cls("breakout", seed=9701)
        hs, led, trues = [], [], []
        for t, (a, rest) in enumerate(plan):
            env.step(a)
            if t % 50 == 0:
                hs.append(env.clone())
                trues.append(env._store[hs[-1]][2])
            if rest and hs:
                env.restore(hs[t % len(hs)])
            led.append(env.billed)
        out.append((led, trues))
    assert out[0] == out[1]
    lag = rng.integers(0, 256, 2000)
    avail = rng.integers(0, 101, 2000)
    t = rng.integers(4, 100, 2000)
    for H, soe in ((24, False), (40, True)):
        assert np.array_equal(labels_v83(lag, avail, t, H, stops_on_event=soe),
                              labels_L(lag, avail, t, np.full(2000, 100), H, stops_on_event=soe))


def test_3b_trace84_cumulative_rises_replay():
    tr = Trace84()
    env = SenseEnv84("breakout", seed=9702, trace=tr)
    rng = np.random.default_rng(4)
    env.reset()
    hs = [env.clone()]
    for _ in range(40):
        env.restore(hs[int(rng.integers(len(hs)))])
        for _ in range(int(rng.integers(5, 80))):
            if env.step(int(rng.integers(env.n_actions)))[3]:
                break
        hs.append(env.clone())
    for b, cum, h, true, depth in tr.records:
        c2, t2, d2 = tr.replay84(h)
        assert c2 == cum and t2 == true and d2 == depth


# ---------------------------------------------------------------- 4. label causality
def test_4_label_causality():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(20, 400))
        L = np.full(n, n)
        t = np.arange(0, n, 4)
        ev = rng.random(n) < 0.03
        for H, soe in ((24, False), (40, True)):
            def mask(e):
                r = np.nonzero(e)[0]
                lag = np.array([min(255, int(r[r >= x][0] - x)) if (r >= x).any() else 255 for x in t])
                return labels_L(lag, n - t, t, L[t], H, soe) >= 0
            e2 = ev.copy()
            cut = int(rng.integers(0, n))
            e2[cut:] = rng.random(n - cut) < 0.5          # alter events after `cut`
            m1, m2 = mask(ev), mask(e2)
            before = t + H <= cut                            # rows whose window ends before the change
            assert np.array_equal(m1[before], m2[before])


# ---------------------------------------------------------------- 5, 6. prefix identity and determinism
class _StubSteer:
    """Draws exactly like V2 (its own functions, no extra RNG draws); records the hooks."""

    def __init__(self):
        self.iters = 0

    def pick_cell(self, ex, rng, exp_n):
        ex.steer, s = None, ex.steer
        try:
            return ex.pick_cell(rng, exp_n)
        finally:
            ex.steer = s

    def pick_process(self, ex, rng, i):
        return ex.bandit.pick(rng)

    def on_put(self, ex, j):
        assert ex.arch.used[j]

    def on_iter(self, ex, n):
        self.iters = n


def _v2_run(tmp, steer=None, cap=180000, game="amidar"):   # entry ~100k + ~80k of exploration
    from ragnarok.agent_v83.entry import Body
    from ragnarok.agent_v84.explorer2 import Explorer2
    tr = Trace84()
    env = SenseEnv84(game, seed=9700, cap=cap, trace=tr)
    rng = np.random.default_rng(9700)
    body = Body()
    body.enter(env, rng, 9700, warm_steps=1000, seed_steps=300, check_steps=200)
    ex = Explorer2(body, tmp, budget=cap, steer=steer)
    out = ex.run(env, rng, np.random.default_rng([9700, 1]), spent=Spent)
    rows = [dict(np.load(p)) for p in sorted(pathlib.Path(tmp).glob("rows_*.npz"))]
    return out, ex, tr, rows


def _same(a, b):
    oa, ea, ta, ra = a
    ob, eb, tb, rb = b
    assert oa == ob
    assert np.array_equal(np.array(ea.exp_rows), np.array(eb.exp_rows))
    assert ea.exp_meta == eb.exp_meta
    assert ta.records == tb.records
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for k in x:
            assert np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f"), k


def test_5_prefix_identity_stub_steer(tmp_path):
    stub = _StubSteer()
    _same(_v2_run(tmp_path / "v2"), _v2_run(tmp_path / "steer", steer=stub))
    assert stub.iters > 0


def test_6_determinism(tmp_path):
    _same(_v2_run(tmp_path / "a"), _v2_run(tmp_path / "b"))
