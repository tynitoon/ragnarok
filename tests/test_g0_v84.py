"""v84 G0 tests T1-T7 (docs/ARC4_V84_PROTOCOL_EN.md A.15) and unit tests of the metric."""
import math
import pathlib
import re

import numpy as np
import pytest

from ragnarok.agent_v84.consequence import auc, checkpoints, run_curve
from ragnarok.evaluation import g0_v84 as G
from ragnarok.evaluation.g0_pipeline_v84 import arm_sigmas, curve_for, direction_data, fit_sources

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "craft_v6_out" / "v84" / "g0" / "store"
A1 = pathlib.Path(r"F:\dev\ragnarok_v83\craft_v6_out\v83")


# ---------------------------------------------------------------- T1, T2, T6 (need the store)
needs_store = pytest.mark.skipif(not (STORE / "store.json").exists(), reason="store not built")


@needs_store
def test_T1_features_and_labels_match_v83():
    """1,000 random stored rows (seed 9700): features (float64 before the cast) and labels equal the v83
    computation (promise.feats; train_promise_v83.head_labels rule)."""
    import json
    from ragnarok.agent_v83.logs import labels
    from ragnarok.agent_v83.promise import feats
    info = json.load(open(STORE / "store.json"))["streams"]
    rng = np.random.default_rng(9700)
    picks = rng.choice(len(info), 10, replace=False)
    n = 0
    for k in picks:
        s = info[int(k)]
        z = np.load(STORE / f"{s['game']}_{s['seed']}.npz")
        a = np.load(A1 / "room" / f"room_{s['game']}_{s['seed']}.npz")
        rows = rng.choice(len(z["rows_index"]), min(100, len(z["rows_index"])), replace=False)
        for j in rows:
            i = int(z["rows_index"][j])
            f = feats(a["row_view"][i], a["row_screen"][i], False, (float(a["row_mx"][i]), float(a["row_my"][i])),
                      int(a["row_lives"][i]), int(a["row_slot"][i]), (210, 160))
            assert np.array_equal(f.astype(np.float16), z["X"][j])
            yg = labels(a["row_rise_lag"][i:i + 1], a["row_avail"][i:i + 1], a["row_t"][i:i + 1], 24)[0]
            yb = labels(a["row_drop_lag"][i:i + 1], a["row_avail"][i:i + 1], a["row_t"][i:i + 1], 40, stops_on_event=True)[0]
            assert yg == z["y_good"][j] and yb == z["y_bad"][j]
            n += 1
    assert n >= 900


@needs_store
def test_T2_clock_non_decreasing_and_cut_rows_counted():
    import json
    info = json.load(open(STORE / "store.json"))["streams"]
    assert len(info) == 82
    for s in info:
        z = np.load(STORE / f"{s['game']}_{s['seed']}.npz")
        assert np.all(np.diff(z["clock"]) >= 0)
        assert s["n_kept"] == len(z["clock"]) and "n_cut" in s
        a = np.load(A1 / "room" / f"room_{s['game']}_{s['seed']}.npz")
        assert np.all(z["it"] < len(a["iter_billed"]))


@needs_store
def test_T6_dedup_applied_and_counted():
    import json
    info = json.load(open(STORE / "store.json"))["streams"]
    g = info[0]["game"]
    zs = [np.load(STORE / f"{g}_{s}.npz") for s in (9131, 9132)]
    st = [dict(X=z["X"], y=dict(good=z["y_good"], bad=z["y_bad"]), clock=z["clock"], block=z["block"], it=z["it"],
               hash=z["hash"], slot=z["slot"]) for z in zs]
    idx, removed = G.test_set(st[0], st[1], "good", 0, 9132)
    assert not np.isin(st[1]["hash"][idx], st[0]["hash"]).any()
    assert 0.0 <= removed <= 1.0 and np.all(st[1]["block"][idx] % 2 == 1)


# ---------------------------------------------------------------- T3 prefix certificate
def test_T3_prefix_certificate_zero_mean_arms():
    rng = np.random.default_rng(1)
    n, d = 600, 12
    Z = rng.standard_normal((n, d))
    y = (rng.random(n) < 0.1).astype(np.int8)
    y[:300] = 0                                          # no positive in the first 300 rows
    Zt = rng.standard_normal((400, d))
    yt = (rng.random(400) < 0.3).astype(np.int8)
    S = G.sigma_from_dirs(rng.standard_normal((5, d)), d)
    for Sig in (np.eye(d), S):
        cks, a = run_curve(Z, y, Zt, yt, Sig, 0.25)
        for c, v in zip(cks, a):
            if c <= 300:
                assert v == 0.5


# ---------------------------------------------------------------- T4 spectra and traces
def test_T4_spectrum_and_trace_identities():
    rng = np.random.default_rng(2)
    d = 30
    S_X = G.sigma_from_dirs(rng.standard_normal((20, d)), d)
    lam = G.eig_desc(S_X)[0]
    S_P = G.spectrum_matched(G.sigma_from_dirs(rng.standard_normal((20, d)), d), lam)
    R = G.haar(d, rng)
    for S in (S_X, S_P, np.diag(lam), R @ S_X @ R.T):
        assert abs(np.trace(S) - d) < 1e-9 * d
        assert np.max(np.abs(G.eig_desc(S)[0] - lam) / lam) <= 1e-9


# ---------------------------------------------------------------- T5 sealed-path guard
def test_T5_sealed_path_guard():
    from scripts.g0_store_v84 import guard
    with pytest.raises(AssertionError):
        guard(pathlib.Path(r"F:\dev\ragnarok_v83\craft_v6_out\v83\explore\explore_promise.json"))
    pat = re.compile(r"explore", re.I)
    for f in list((ROOT / "scripts").glob("g0_*_v84.py")) + list((ROOT / "scripts").glob("g0_v84.py")) + \
            list((ROOT / "ragnarok" / "evaluation").glob("g0_*v84.py")):
        text = f.read_text(encoding="utf-8")
        hits = [ln for ln in text.splitlines() if pat.search(ln) and "SEALED" not in ln and "sealed" not in ln.lower()]
        assert not hits, (f.name, hits[:3])


# ---------------------------------------------------------------- metric unit tests
def test_checkpoints_and_crossing():
    assert checkpoints(128) == [128]
    c = checkpoints(1000)
    assert c[0] == 128 and c[-1] == 1000 and all(b > a for a, b in zip(c, c[1:]))
    clocks = [1e4, 2e4, 4e4, 8e4, 16e4]
    N, ok = G.crossing([0.5, 0.55, 0.7, 0.72, 0.75], clocks, 0.6, False, 2e5)
    assert ok and 2e4 < N < 4e4
    N, ok = G.crossing([0.5, 0.55, 0.58, 0.59, 0.7], clocks, 0.6, True, 2e5)
    assert not ok and N == 4e5                           # carried arm never crossing -> 2 c_end
    N, ok = G.crossing([0.5, 0.55, 0.58, 0.59, 0.7], clocks, 0.6, False, 2e5)
    assert not ok and N == 2e5                           # control -> c_end


def test_auc_ties_half():
    assert auc(np.zeros(10), np.array([0, 1] * 5)) == 0.5
    assert auc(np.arange(4), np.array([0, 0, 1, 1])) == 1.0


# ---------------------------------------------------------------- T7 Gaussian unit test
T7_GEN = dict(wn=0.8, bias=-1.5)     # defaults; the frozen-size scan of build note 10 found no qualifying setting


def _gauss_dataset(shared, seed, G_games=6, n=2000, d=20, wn=None, bias=None):
    """Synthetic games: logistic labels on Gaussian features; weights in a shared 3-dim subspace or
    independent. Frozen sizes (A.15 T7): 6 games, 2,000-row streams, 20 dims; wn and bias free."""
    wn = T7_GEN["wn"] if wn is None else wn
    bias = T7_GEN["bias"] if bias is None else bias
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((d, 3)))[0]
    data = {}
    for g in range(G_games):
        w = V @ rng.standard_normal(3) if shared else rng.standard_normal(d)
        w = wn * w / np.linalg.norm(w)
        streams = []
        for s in range(2):
            X = rng.standard_normal((n, d))
            y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w + bias)))).astype(np.int8)
            streams.append(dict(X=X, y=dict(good=y, bad=y.copy()), clock=(np.arange(n) + 1) * 400, block=np.arange(n) // 50,
                                it=np.arange(n) // 25, hash=rng.integers(0, 2**62, n).astype(np.uint64), slot=np.zeros(n, np.int16)))
        data[f"g{g}"] = streams
    return data


def _gauss_curves(shared, seed, arms_wanted=("F", "X"), **gen):
    data = _gauss_dataset(shared, seed, **gen)
    games = list(data)
    gi = {g: i for i, g in enumerate(games)}
    sums = {g: G.game_sums(data[g]) for g in games}
    sources = fit_sources(data, games, gi, (9131, 9132), heads=("good",), kinds=("real", "perm")) if "X" in arms_wanted else {}
    out = []
    for g in games:
        src = [h for h in games if h != g]
        geo = G.geometry(sums, src)
        dd = direction_data(g, "good", data[g], (9131, 9132), geo, gi, 0, 1)
        arms = arm_sigmas(g, "good", geo, sources, src, gi) if "X" in arms_wanted else {"F": (np.eye(geo["d"]), None)}
        out.append((dd, {a: curve_for(a, arms[a], dd, 0.25) for a in arms_wanted}))
    return out


def _gauss_ratio(shared, seed, **gen):
    lam_X, lam_F = [], []
    for dd, cur in _gauss_curves(shared, seed, **gen):
        fend = float(np.mean(cur["F"][-3:]))
        lam_F.append(G.lambdas(cur["F"], dd["clocks"], fend, False, dd["c_end"])[0])
        lam_X.append(G.lambdas(cur["X"], dd["clocks"], fend, True, dd["c_end"])[0])
    return math.exp(np.mean(np.array(lam_F) - np.array(lam_X)))


def f_only_late_games(shared, seed, **gen):
    """Number of games where F alone crosses N(0.25) after c(n_0) (build note 10 rule)."""
    n = 0
    for dd, cur in _gauss_curves(shared, seed, arms_wanted=("F",), **gen):
        fend = float(np.mean(cur["F"][-3:]))
        N, _ = G.crossing(cur["F"], dd["clocks"], 0.5 + 0.25 * (fend - 0.5), False, dd["c_end"])
        n += N > max(G.FLOOR, dd["clocks"][0])
    return n


def test_T7_gaussian_shared_vs_independent():
    """No generator at the frozen sizes passed the F-only rule of build note 10 (0 of 12 settings), so T7
    keeps the 3,000-row / 60-dim generator under the "fixed and recorded" clause (build notes 2 and 11)."""
    assert _gauss_ratio(True, 11, n=3000, d=60, wn=0.8, bias=-1.5) > 1.5
    assert _gauss_ratio(False, 12, n=3000, d=60, wn=0.8, bias=-1.5) <= 1.25


# ---------------------------------------------------------------- aggregation (A.11-A.12)
def test_ratio_sign_cstar_min_and_game_weighting():
    from ragnarok.evaluation.g0_stages_v84 import summarize
    cl = [1e4 * 2 ** k for k in range(8)]
    slow = [0.5, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.7]
    fast = [0.5, 0.6, 0.66, 0.7, 0.7, 0.7, 0.7, 0.7]
    units = []
    for g, heads in (("a", ("good", "bad")), ("b", ("good",))):
        for h in heads:
            units.append(dict(game=g, head=h, eligible=True,
                              dirs=[dict(dir=d, auc_F=slow, auc_F_end=0.7, clocks=cl, c_end=cl[-1], prefix_ok=True) for d in (0, 1)]))
    rows = [dict(game=u["game"], head=u["head"], dir=d, curves=dict(X=fast, P=slow, U=fast, **{"K*A": fast}))
            for u in units for d in (0, 1)]
    res = summarize(units, rows)
    assert res["X:F"]["R"] > 1                     # X faster than F gives R > 1
    assert abs(res["X:C*"]["R"] - 1.0) < 1e-12     # C* = per-unit min over F, P, U: U is as fast as X
    assert res["n_games"] == 2 and set(res["X:F"]["per_game"]) == {"a", "b"}


def test_late_censoring_cases():
    cl = [1e4 * 2 ** k for k in range(6)]
    never = [0.5] * 6
    lam_c, late_c, _, _ = G.lambdas(never, cl, 0.8, False, cl[-1])
    lam_x, late_x, _, _ = G.lambdas(never, cl, 0.8, True, cl[-1])
    assert lam_x > lam_c                            # carried arm never crossing: 2 c_end; control: c_end
    assert late_c == math.log(G.FLOOR)              # control without N(0.5): D = 5,000
    assert abs(late_x - math.log(2 * cl[-1])) < 1e-12
    half = [0.5, 0.66, 0.66, 0.66, 0.66, 0.66]      # reaches L(0.5) = 0.65 but never L(0.625)
    _, late_c2, N, ok = G.lambdas(half, cl, 0.8, False, cl[-1])
    assert ok[0.5] and not ok[0.625] and abs(late_c2 - math.log(cl[-1] - N[0.5])) < 1e-12


def test_signflip_exact():
    lg = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert abs(G.signflip_p(lg) - 1 / 8) < 1e-12


def test_crossing_requires_two_checkpoints():
    cl = [1e4, 2e4, 4e4, 8e4, 16e4]
    N, ok = G.crossing([0.5, 0.7, 0.55, 0.7, 0.7], cl, 0.6, False, 2e5)
    assert ok and N > 4e4                           # the dip at k+1 delays the crossing
