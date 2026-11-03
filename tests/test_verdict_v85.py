"""G-build test 7 (Part B): v85 verdict and calibration code on synthetic run JSONs only (seed 9800)."""
import math

import numpy as np
import pytest

from scripts import v85_calib as C
from scripts import v85_verdict as V

KS = (2, 3, 4)
CAP = 4_000_000


def mk(game, start, arm, seed, T, ks=KS, **kw):
    r = dict(game=game, start=start, arm=arm, seed=seed, cap=CAP, T={str(k): t for k, t in zip(ks, T)},
             replay=[dict(k=ks[0], ok=True, replayed=1.0, recorded=1.0)], interruption=False,
             provenance=dict(dirty=False, closure_hash=f"h_{V.canon(arm)}"))
    r.update(kw)
    return r


def synthetic(factors, arm="O_4", ks=KS, n_arm=3):
    """V2s: 3 runs per (unit, start); the arm's run i is V2s run i with every T divided by the unit's factor."""
    rng = np.random.default_rng([9800, 1])
    out = []
    for u, f in enumerate(factors):
        for s, s0 in (("A", 9806), ("B", 9809)):
            for i in range(3):
                T = np.cumsum(rng.uniform(1e5, 9e5, size=len(ks)))
                out.append(mk(f"g{u}", s, "V2s", s0 + i, T.tolist(), ks))
                if i < n_arm:
                    out.append(mk(f"g{u}", s, arm, V.G1_SEED0[s] + i, (T / f).tolist(), ks))
    return out


def kgs(units, ks=KS):
    return {(f"g{u}", s): list(ks) for u in range(units) for s in "AB"}


# ------------------------------------------------------------------ exact R, floor, cap, records

def test_exact_R_when_neither_floor_nor_cap_binds():
    runs = V.group(synthetic([1.7] * 6))
    st = V.contrast(runs, kgs(6), "O_4", "V2s")
    for k in ("R", "R_late", "R_8M"):
        assert st[k] == pytest.approx(1.7, rel=1e-12)
    assert all(v == pytest.approx(1.7, rel=1e-12) for v in st["r_g"].values())
    assert st["LB90"] <= st["R"] <= st["UB95"] and st["cens_arm"] == st["cens_comp"] == 0.0
    # unit heterogeneity: R = geometric mean of r_g, each clipped to [1/8, 8]; breadth = share r_g >= 1.5
    f = [1.5, 2.0, 3.0, 0.9, 1.2, 20.0]
    st = V.contrast(V.group(synthetic(f)), kgs(6), "O_4", "V2s")
    want = [1.5, 2.0, 3.0, 0.9, 1.2, 8.0]
    assert st["R"] == pytest.approx(math.exp(np.mean(np.log(want))), rel=1e-12)
    assert st["r_g"]["g5"] == pytest.approx(8.0) and st["l_g"]["g5"] == pytest.approx(math.log(20.0))
    assert st["breadth"] == pytest.approx(4 / 6)
    # the floor binds: T / f below 5000 frames gives less than f
    st = V.contrast(V.group(synthetic([400.0] * 4)), kgs(4), "O_4", "V2s", clip=False, B=0)
    assert st["R"] < 400.0


def test_times_from_records_and_total_frames():
    lad = [0, 10, 20, 40, 80]
    r = dict(cap=CAP, ladder=lad, oracle_frames=7000,
             records=[[100, 12, 0, 1, 10], [5000, 25, 0, 2, 900], [70000, 45, 0, 3, 4000]])
    assert V.run_times(r, KS) == [5000.0, 70000.0, None]
    assert V.run_times(r, KS, total=True) == [5900.0, 74000.0, None]
    m = V.run_measures(r, KS, total=True)
    assert m["lam"] == pytest.approx(np.mean(np.log([5900, 74000, CAP + 7000])))
    assert V.run_times(dict(r, records=[[CAP + 1, 30, 0, 1, 0]]), KS) == [None, None, None]


def test_bootstrap_matches_a_naive_two_level_loop():
    rng = np.random.default_rng([9800, 2])
    U, S, Na, Nc, B = 3, 2, 2, 3, 40
    la, lc = rng.normal(10, 1, (2, U, S, Na)), rng.normal(11, 1, (2, U, S, Nc))
    na = np.array([[2, 1], [2, 0], [1, 2]])
    nc = np.array([[3, 3], [3, 0], [2, 3]])
    valid = (na > 0) & (nc > 0)
    draws = V.boot_draws(np.random.default_rng([9800, 3]), B, U, S, Na, Nc)
    l, lr, lR, bR = V.ssc_core(la, lc, na, nc, valid, draws, [(0, 0), (1, 1)])
    I, Fa, Fc = draws
    for p in range(2):
        for b in range(B):
            acc = []
            for pos in range(U):
                g = I[b, pos]
                ds = []
                for s in range(S):
                    if not valid[g, s]:
                        continue
                    ma = np.mean([la[p, g, s, int(Fa[b, pos, s, i] * na[g, s])] for i in range(na[g, s])])
                    mc = np.mean([lc[p, g, s, int(Fc[b, pos, s, i] * nc[g, s])] for i in range(nc[g, s])])
                    ds.append(mc - ma)
                acc.append(np.clip(np.mean(ds), -V.LOG8, V.LOG8))
            assert bR[p, b] == pytest.approx(np.mean(acc), abs=1e-12)
    pt = [np.mean([lc[0, g, s, :nc[g, s]].mean() - la[0, g, s, :na[g, s]].mean() for s in range(S) if valid[g, s]])
          for g in range(U)]
    assert l[0] == pytest.approx(pt)


# ------------------------------------------------------------------ censoring treatments

def test_censoring_in_the_yardstick():
    lam, late = V.lam_arrays(np.array([1e5, 3e5, CAP]), np.array([False, False, True]), np.ones(3, bool))
    assert lam == pytest.approx(np.mean(np.log([1e5, 3e5, CAP])))
    assert late == pytest.approx(np.mean(np.log([2e5, CAP - 1e5])))
    lam, late = V.lam_arrays(np.array([CAP, CAP, CAP]), np.ones(3, bool), np.ones(3, bool))
    assert late == pytest.approx(math.log(CAP))                       # M_{k_first} missed: log(4M) terms
    lam8 = V.lam_arrays(np.array([1e5, 3e5, 0.0]), np.array([False, False, True]), np.ones(3, bool), V.CAP8)[0]
    assert lam8 == pytest.approx(np.mean(np.log([1e5, 3e5, V.CAP8])))
    # V2s misses at 8M raise R and the 8M bound; censoring rate = share of runs missing the top counted rung
    rs = synthetic([1.0] * 4)
    for r in rs:
        if r["arm"] == "V2s" and r["seed"] in (9806, 9809):
            r["T"]["4"] = None
    st = V.contrast(V.group(rs), kgs(4), "O_4", "V2s")
    assert st["R_8M"] > st["R"] > 1.0 and st["UB95_8M"] >= st["UB95"]
    assert st["cens_comp"] == pytest.approx(1 / 3) and st["cens_arm"] == 0.0


def test_censoring_treatments_in_the_calibration():
    T = np.array([[[1e5, 2e5, CAP]]])
    miss = np.array([[[False, False, True]]])
    one, zero = np.ones((1,)), np.zeros((1, 1))
    Tp, mp = C.plant(T, miss, "D2", "stay", 2 * one, zero)
    assert Tp.tolist() == [[[5e4, 1e5, CAP]]] and mp.tolist() == [[[False, False, True]]]
    Tp, mp = C.plant(T, miss, "D2", "divide", 2 * one, zero)
    assert Tp.tolist() == [[[5e4, 1e5, CAP / 2]]] and not mp.any()
    Tp, mp = C.plant(T, miss, "Dlate", "stay", 2 * one, zero)
    assert Tp.tolist() == [[[1e5, 1.5e5, CAP]]] and mp[0, 0, 2]
    Tp, mp = C.plant(T, miss, "Djump", "stay", one, zero)
    assert Tp.tolist() == [[[5000, 1.05e5, CAP]]]
    Tp, mp = C.plant(np.full((1, 1, 3), float(CAP)), np.ones((1, 1, 3), bool), "Djump", "divide", one, zero)
    assert Tp.tolist() == [[[5000, 5000, 5000]]] and not mp.any()
    Tp, mp = C.plant(np.array([[[3e6, 3.5e6, 3.9e6]]]), np.zeros((1, 1, 3), bool), "D1", "stay", one,
                     np.full((1, 1), 0.2))
    assert mp.tolist() == [[[False, True, True]]] and Tp.max() == CAP  # noise past the cap is a miss
    # resampling: a rung missed by every run stays censored; a rung missed by one run is censored when drawn
    cells = {("u0", "A"): np.array([[1e5, 2e5, np.nan], [2e5, 4e5, np.nan], [1.5e5, 3e5, 3e6]]),
             ("u0", "B"): np.array([[1e5, 2e5, 3e5], [2e5, 4e5, 6e5], [1.5e5, 3e5, 4.5e5]])}
    base = C.prepare(cells, ["u0"])
    Tr, mr = C.resample(base, np.array([[0, 1, 2], [0, 1, 2]]))
    assert mr[0, :, 2].tolist() == [True, True, False] and not mr[1].any()
    assert base["infl"][0] == pytest.approx(math.sqrt(1.5))
    # resampled vectors keep the cell mean when residuals are drawn evenly
    assert np.log(Tr[1]).mean(0) == pytest.approx(np.log(cells[("u0", "B")]).mean(0))


def test_calibration_recovers_planted_dilations_exactly():
    """Identical runs per cell (zero residuals, zero sigma2) and tau = 0: every planted R is exact."""
    cells = {(f"u{u}", s): np.tile(np.cumsum([2e5 + 1e4 * u, 3e5, 4e5]), (3, 1)) for u in range(10) for s in "AB"}
    base = C.prepare(cells, [f"u{u}" for u in range(10)])
    st = C.simulate(base, sims=2, n_list=(4,), taus=(0.0,), B=50)[0, 0]
    get = lambda sc, k: st[:, :, C.COMBOS.index((sc, "stay")), C.STAT.index(k)]       # noqa: E731
    for sc, D in (("D2", 2.0), ("D1.5", 1.5), ("D1.25", 1.25), ("D1", 1.0)):
        assert np.allclose(get(sc, "R"), D, rtol=1e-12) and np.allclose(get(sc, "LB90"), D, rtol=1e-12)
        assert np.allclose(get(sc, "R_late"), D, rtol=1e-12)
    assert np.allclose(get("Dlate", "R_late"), 2.0, rtol=1e-12) and np.allclose(get("Djump", "R_late"), 1.0)
    assert np.all(get("Djump", "R") > 1.0)


# ------------------------------------------------------------------ labels: intersection-union, precedence

ROOMY = dict(R=2.2, LB90=1.4, LB90_late=1.1, breadth=0.5, cens_arm=0.1, cens_comp=0.1, UB95=3.0, UB95_8M=3.2)


def test_room_clauses_intersection_union():
    assert V.row_label(ROOMY, 1.8, 1.25)["label"] == "ROOM-K"
    for k, bad in (("R", 1.79), ("LB90", 1.2), ("LB90_late", 1.0), ("breadth", 0.39), ("cens_arm", 0.4),
                   ("cens_comp", 0.45)):
        assert V.row_label(dict(ROOMY, **{k: bad}), 1.8, 1.25)["label"] != "ROOM-K", k
    no = dict(R=1.1, LB90=0.9, LB90_late=0.95, breadth=0.2, cens_arm=0.1, cens_comp=0.2, UB95=1.6, UB95_8M=1.8)
    assert V.row_label(no, 1.8, 1.25)["label"] == "NO-ROOM-K"
    for k, bad in (("UB95", 2.0), ("UB95_8M", 2.1), ("cens_arm", 0.26), ("cens_comp", 0.3)):
        assert V.row_label(dict(no, **{k: bad}), 1.8, 1.25)["label"] == "INCONCLUSIVE-K", k
    lab = V.row_label(no, 1.8, 1.25, map_weak=True)                   # MAP-WEAK turns NO-ROOM into INCONCLUSIVE
    assert lab["label"] == "INCONCLUSIVE-K" and "MAP-WEAK" in lab["captions"]


def test_label_precedence_and_gate_verdict():
    both = dict(R=1.9, LB90=1.3, LB90_late=1.05, breadth=0.5, cens_arm=0.1, cens_comp=0.1, UB95=1.95, UB95_8M=1.97)
    lab = V.row_label(both, 1.8, 1.25)
    assert lab["label"] == "ROOM-K" and lab["captions"] == ["BELOW-2X"] and lab["clauses"]["noroom_a"]
    assert "BELOW-2X" not in V.row_label(ROOMY, 1.8, 1.25)["captions"]
    g = V.g1_label
    assert g(dict(o4="NO-ROOM-K", kmap="ROOM-K", kmac="INCONCLUSIVE-K")) == "ROOM"
    assert g(dict(o4="NO-ROOM-K", kmap="NO-ROOM-K", kmac="NO-ROOM-K")) == "NO-ROOM"
    assert g(dict(o4="NO-ROOM-K", kmap="NO-ROOM-K", kmac="INCONCLUSIVE-K")) == "INCONCLUSIVE"
    assert g(dict(o4="NO-ROOM-K", kmap="NO-ROOM-K")) == "INCONCLUSIVE"            # a missing row is not NO-ROOM
    assert g(dict(kmap="NO-ROOM-K", kmac="NO-ROOM-K"), o_dropped=True) == "NO-ROOM"
    assert g(dict(o4="ROOM-K", kmap="ROOM-K", kmac="ROOM-K"), ["replay mismatch"]) == "Interruption"
    # array form used by the calibration agrees with the scalar form
    arr = {k: np.array([v, ROOMY[k]]) for k, v in both.items()}
    room, noroom, below = V.a11(V.a11_clauses(*(arr[k] for k in ("R", "LB90", "LB90_late", "breadth", "cens_arm",
                                                                    "cens_comp", "UB95", "UB95_8M")), 1.8, 1.25))
    assert room.tolist() == [True, True] and noroom.tolist() == [False, False] and below.tolist() == [True, False]


def test_g1p_g2_and_routing_each_contrast_on_its_own():
    assert V.g1p_label(dict(UB90=1.2, LB90=0.9)) == "GENERIC-K"
    assert V.g1p_label(dict(UB90=1.6, LB90=1.05)) == "SEPARATED-K"
    assert V.g1p_label(dict(UB90=1.6, LB90=0.95)) == "NOT-SEPARATED-K"
    good = dict(R=1.6, LB90=1.2, LB90_late=1.1)
    others = dict(V2s=dict(LB90=1.3), DM_on=dict(LB90=1.1), P_on=dict(LB90=1.05))
    assert V.g2b_label(good, others)["label"] == "GO"
    assert V.g2b_label(good, dict(others, P_on=dict(LB90=0.99)))["label"] == "KC-dev"
    assert V.g2b_label(dict(good, LB90_late=1.0), others)["label"] == "KC-dev"
    assert V.g2a_label(dict(R=2.1, LB90=1.6), dict(LB90=1.1)) == dict(label="OBJECT-ROOM", sealed_top_label="FASTER-2X")
    assert V.g2a_label(dict(R=1.5, LB90=1.1), dict(LB90=1.0))["label"] == "KC-object"
    assert V.vocab_caption(dict(LB90=1.0)) == "HEAD-START-VOCABULARY"
    r = V.routing(dict(o4="NO-ROOM-K", kmap="ROOM-K", kmac="INCONCLUSIVE-K"), dict(kmap="SEPARATED-K"))
    assert r["result"] == "ROOM-UNCARRIED"
    r = V.routing(dict(o4="ROOM-K", kmap="ROOM-K", kmac="ROOM-K"), dict(o4="GENERIC-K", kmac="NOT-SEPARATED-K"))
    assert r["families"] == ["M"]
    assert V.routing(dict(o4="NO-ROOM-K", kmap="INCONCLUSIVE-K", kmac="NO-ROOM-K"), {})["families"] == []


# ------------------------------------------------------------------ cross-fit, I-stage, A.12

def test_cross_fit_rule():
    lad = [0, 10, 20, 40, 80, 160, 320, 640, 1280]
    rec = lambda top: dict(cap=CAP, ladder=lad, records=[[1000 * i, lad[i] + 1, 0, i, 0] for i in range(1, top + 1)])  # noqa: E731
    K_g = [3, 4, 5, 7]
    cf = V.cross_fit(K_g, 7, [rec(8), rec(5), rec(6)])
    assert cf["k_top"] == 5 and cf["K"] == [3, 4, 5] and cf["complete"]
    assert V.cross_fit(K_g, 7, [], other_valid=False)["K"] == K_g        # invalid other start: k_ref
    assert V.cross_fit(K_g, 7, [rec(8), rec(3), rec(8)])["K"] == []      # |K_{g,s}| < 2: dropped
    assert V.cross_fit(K_g, 7, [rec(8), rec(2), rec(8)])["K"] == []      # a run reaching no counted rung
    assert not V.cross_fit(K_g, 7, [rec(8), rec(8)])["complete"]
    # I-stage: every (g, s) is cross-fitted from the other start's V2s runs
    runs = [dict(r, rungs=K_g, k_ref=7) for r in synthetic([1.0], ks=K_g)]
    runs = [r for r in runs if r["arm"] == "V2s"]
    for r in runs:
        if r["start"] == "B" and r["seed"] == 9810:
            r["T"]["7"] = None
    K_gs, det = V.istage_kgs(V.group(runs))
    assert K_gs[("g0", "A")] == [3, 4, 5] and K_gs[("g0", "B")] == K_g
    # S0 starts.json: K_g and validity; a valid other start without V2s runs gives no k_top, an invalid one k_ref
    starts = {"order": ["g0"], "units": {"g0": {"K_g": K_g, "A": {"valid": True}, "B": {"valid": True}}}}
    only_a = V.group([r for r in runs if r["start"] == "A"])
    K_gs, _ = V.istage_kgs(only_a, starts, V.start_validity(starts))
    assert ("g0", "A") not in K_gs
    starts["units"]["g0"]["B"] = None
    K_gs, _ = V.istage_kgs(only_a, starts, V.start_validity(starts))
    assert K_gs == {("g0", "A"): K_g}


def test_d_control_and_instrument_line():
    rs = synthetic([1.0, 1.0, 1.0, 1.0, 1.0], arm="D", n_arm=1)
    for r in rs:
        if r["arm"] == "D":
            r["T"] = {k: v / 6 for k, v in r["T"].items()}
    runs = V.group(rs)
    units = [f"g{u}" for u in range(5)]
    d = V.d_control(runs, kgs(5), units)
    assert d["passed"] and d["share"] == 1.0
    d = V.d_control(runs, kgs(5), units + ["g9"])                         # a unit without D counts as a failure
    assert d["share"] == pytest.approx(5 / 6) and d["passed"]
    runs[("g0", "A", "d")][0]["replay"][0]["ok"] = False
    assert not V.d_control(runs, kgs(5), units)["passed"]
    # instrument line: identical halves -> interval [1, 1]; a 3x faster half -> INSTRUMENT-FAIL
    v = [r for r in synthetic([1.0] * 5) if r["arm"] == "V2s"]
    for r in v:
        if r["seed"] in (9807, 9810):
            twin = next(x for x in v if x["game"] == r["game"] and x["start"] == r["start"] and x["seed"] == r["seed"] - 1)
            r["T"] = dict(twin["T"])
    line = V.instrument_line(V.group(v), kgs(5))
    assert line["CI99"] == pytest.approx([1.0, 1.0]) and line["instrument_fail"] is False
    for r in v:
        if r["seed"] in (9806, 9809):
            r["T"] = {k: t / 3 for k, t in r["T"].items()}
    assert V.instrument_line(V.group(v), kgs(5))["instrument_fail"] is True


def test_a12_parameters_and_downgrade_only():
    rng = np.random.default_rng([9800, 4])
    rs = []
    for u in range(8):
        for s in "AB":
            for i in range(3):
                rs.append(mk(f"g{u}", s, "V2s", 9806 + i, np.cumsum(rng.uniform(1e5, 2e5, 3) * np.exp(rng.normal(0, 0.1))).tolist()))
                rs.append(mk(f"g{u}", s, "K*map", 9812 + i, np.cumsum(rng.uniform(1e5, 2e5, 3) * np.exp(rng.normal(0, 0.6))).tolist()))
    p = V.a12_params(V.group(rs), kgs(8), "K*map")
    assert p["v_K"] > 1.5 and V.a12_triggered(p) and p["tau_K"] >= 0.0
    ok = dict(ii=True, iii=True, iv=True, v=True, vi=True)
    assert V.a12_downgrade("ROOM-K", ok) == "ROOM-K"
    assert V.a12_downgrade("ROOM-K", dict(ok, iv=False)) == "INCONCLUSIVE-K"
    assert V.a12_downgrade("ROOM-K", dict(ok, vi=False)) == "ROOM-K"
    assert V.a12_downgrade("NO-ROOM-K", dict(ok, vi=False)) == "INCONCLUSIVE-K"
    assert V.a12_downgrade("INCONCLUSIVE-K", dict(ok, ii=False, vi=False)) == "INCONCLUSIVE-K"


# ------------------------------------------------------------------ code identity: certificate, Interruption

def test_comparator_certificate_and_interruptions():
    ist = [dict(r, records=[[10, 5.0, 0, 1, 0]], exp_meta_sha256="e1") for r in synthetic([1.0]) if r["arm"] == "V2s"]
    rerun = [dict(r, provenance=dict(dirty=False, closure_hash="h_new")) for r in ist if r["seed"] in (9806, 9809)]
    assert V.certificate(ist, rerun, "g0")["ok"]
    assert V.certificate(ist, rerun, "g0", closure="h_new")["ok"]        # reruns under the committed V2s closure
    assert not V.certificate(ist, rerun, "g0", closure="h_v2s")["ok"]    # reruns under another closure: no certificate
    bad = [dict(r, records=[[11, 5.0, 0, 1, 0]]) if r["start"] == "B" else r for r in rerun]
    assert not V.certificate(ist, bad, "g0")["ok"]
    assert not V.certificate(ist, [rerun[0]], "g0")["ok"]                 # 9809 not re-executed
    assert not V.certificate(ist, [dict(r, exp_meta_sha256="e2") for r in rerun], "g0")["ok"]
    assert not V.certificate([dict(r, exp_meta_sha256=None) for r in ist], rerun, "g0")["ok"]
    job = lambda rs, h: [dict({k: v for k, v in r.items() if k != "exp_meta_sha256"}, exp=dict(n=3, meta_sha256=h))  # noqa: E731
                         for r in rs]
    assert V.certificate(job(ist, "m1"), job(rerun, "m1"), "g0")["ok"]                # v85_job's exp.meta_sha256
    assert not V.certificate(job(ist, "m1"), job(rerun, "m2"), "g0")["ok"]
    g1 = [r for r in synthetic([2.0]) if r["arm"] == "O_4"]
    com = dict(V2s="h_v2s", O_4="h_o4")
    assert V.interruptions(ist + g1, com) == []
    why = V.interruptions(ist + g1, dict(com, V2s="h_new"))
    assert why == ["missing comparator certificate: V2s closure changed"]
    assert V.interruptions(ist + g1, dict(com, V2s="h_new"), V.certificate(ist, rerun, "g0")) == []
    assert "comparator certificate failed" in V.interruptions(ist + g1, dict(com, V2s="h_new"),
                                                            V.certificate(ist, bad, "g0"))[0]
    assert V.interruptions(ist + g1, dict(com, O_4="h_other")) == ["changed closure hash: O_4"]
    mixed = g1[:-1] + [dict(g1[-1], provenance=dict(dirty=False, closure_hash="h_x"))]
    assert V.interruptions(ist + mixed) == ["mixed closure hashes: O_4"]
    dirty = g1[:-1] + [dict(g1[-1], provenance=dict(dirty=True, closure_hash="h_o4"))]
    assert V.interruptions(ist + dirty)[0].startswith("dirty tree")
    rep = g1[:-1] + [dict(g1[-1], replay=[dict(k=2, ok=False, replayed=1, recorded=2)])]
    assert V.interruptions(ist + rep)[0].startswith("replay mismatch")
    seg = g1[:-1] + [dict(g1[-1], provenance=dict(dirty=False, closure_hash="h_o4", closure_consistent=False))]
    assert V.interruptions(ist + seg)[0].startswith("mixed closure hashes within a resumed run")


def test_g1_report_end_to_end_on_synthetic_runs():
    rs = synthetic([2.6] * 6, arm="O_4", n_arm=2) + [r for r in synthetic([1.0] * 6, arm="K*map", n_arm=2) if r["arm"] != "V2s"]
    rs += [r for r in synthetic([1.0] * 6, arm="K*mac", n_arm=2) if r["arm"] != "V2s"]
    for r in rs:
        if r["arm"] == "K*map":
            r["instrument"] = dict(coins=100, found=10)
        if r["arm"] == "O_4":
            r["records"], r["ladder"], r["oracle_frames"] = None, None, 1000
            r["T_total"] = {k: 4 * t for k, t in r["T"].items()}
    dice = [dict(r, arm="O_4-dice", interruption=True) for r in rs if r["arm"] == "O_4" and r["seed"] == 9812]
    rep = V.g1_report(V.group(rs + dice), kgs(6), dict(n=4, tau_pt=1.8, tau_lb=1.25), a12_rerun=lambda *a: None)
    assert rep["rows"]["o4"]["label"] == "ROOM-K" and rep["G1"] == "ROOM"
    so = rep["rows"]["o4"]["stats"]
    assert so["R_total_frames"] == pytest.approx(so["R"] / 4, rel=1e-12) and so["oracle_frames"] == 1000 * 24
    assert rep["rows"]["kmap"]["label"] == "INCONCLUSIVE-K" and "MAP-WEAK" in rep["rows"]["kmap"]["captions"]
    assert rep["rows"]["kmap"]["stats"]["coverage"] == pytest.approx(0.1)
    assert rep["rows"]["kmac"]["label"] == "NO-ROOM-K"
    assert rep["sentences"][0].startswith("From a savestate shared by all arms at M1 on 6 development games")
    assert "O_4-dice" in rep["caption_rows"] and "label" not in rep["caption_rows"]["O_4-dice"]
    # O-TIES pooled over expansions, weighted by the key OChooser.caption() writes ("expansions")
    from ragnarok.evaluation.oracle_v85 import OChooser
    o = OChooser(4)
    o.log = [(e, 0, [(0.0, 0, 0)] * 4, True) for e in range(9)] + [(9, 1, [(0.0, 0, 0), (1.0, 0, 0)] * 2, False)]
    c1 = dict(o.caption(), oracle_frames=1)                               # 10 expansions, 0.9 tied
    o.log = o.log[-1:] * 990 + o.log[:10]                                  # 1000 expansions, 9 tied
    c2 = dict(o.caption(), oracle_frames=1)
    assert (c1["expansions"], c1["o_ties"], c2["expansions"], c2["o_ties"]) == (10, 0.9, 1000, 0.009)
    assert V.o_ties([dict(instrument=c1), dict(instrument=c2)]) == pytest.approx((9 + 9) / 1010)   # not 0.4545
    assert rep["interruptions"] == [] and rep["claim_header"] == V.CLAIM   # caption rows never interrupt G1


def fake_starts(units, K=(2, 3, 4), dropped=(), invalid_b=()):
    """An s0/starts.json: units in A.2 order, both starts valid (E 9819 / 9820), dropped units with K_g [] (A.3.5)."""
    u = {}
    for g in list(units) + list(dropped):
        drop = g in dropped
        u[g] = dict(K_g=[] if drop else list(K), k_first=None if drop else min(K), k_ref=None if drop else max(K),
                    kept=not drop, drop="A.3.5: no valid start" if drop else None,
                    A=dict(valid=not drop, E=9819, s1_sha256=f"s1{g}A", entry=dict(sha256=f"e{g}A")),
                    B=dict(valid=not drop and g not in invalid_b, E=9820, s1_sha256=f"s1{g}B", entry=dict(sha256=f"e{g}B")))
    order = list(dropped) + list(units)
    return dict(order=order, kept=[g for g in order if g not in dropped], units=u)


def istage_runs(factors, K=(2, 3, 4), d_speed=6.0):
    """I-stage runs: 3 V2s per start (9806.. / 9809..), 1 D per start on E_s, D d_speed x faster."""
    rs = [dict(r, rungs=list(K), k_ref=max(K)) for r in synthetic(factors, arm="D", ks=K, n_arm=1)]
    for r in rs:
        if r["arm"] == "D":
            r["T"] = {k: v / d_speed for k, v in r["T"].items()}
            r["seed"] = 9819 if r["start"] == "A" else 9820
    return rs


def write_runs(d, rs):
    import json
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rs):
        (d / f"run{i}.json").write_text(json.dumps(r), encoding="utf-8")


def test_cli_istage_calib_f1_g1_g1p(tmp_path):
    import json
    K = [2, 3, 4]
    units = [f"g{u}" for u in range(4)]
    (tmp_path / "s0").mkdir()
    (tmp_path / "s0" / "starts.json").write_text(json.dumps(fake_starts(units, K, dropped=["gx"])), encoding="utf-8")
    write_runs(tmp_path / "istage", istage_runs([1.0] * 4, K))
    V.main(["istage", "--out", str(tmp_path)])
    iv = json.loads((tmp_path / "istage" / "istage_verdict.json").read_text(encoding="utf-8"))
    assert iv["stage"] == "PASS" and iv["missing"] == [] and iv["interruptions"] == []
    assert iv["d_control"]["passed"] and iv["K_gs"]["g0"] == {"A": K, "B": K} and iv["provenance"]["closure_hash"]
    assert "gx" not in iv["K_gs"] and iv["units"] == units                # S0 drop: no crash, no cell
    C.main(["run", "--out", str(tmp_path), "--sims", "2"])
    cal = json.loads((tmp_path / "calib.json").read_text(encoding="utf-8"))
    assert cal["sims"] == 2 and cal["U"] == 4 and cal["units"] == units
    cal.update(outcome="SELECTED", selection=dict(n=4, tau_pt=1.8, tau_lb=1.25, oc={}, criteria={}))
    (tmp_path / "calib.json").write_text(json.dumps(cal), encoding="utf-8")
    hs, sg, clo = tmp_path / "hashes.json", tmp_path / "grep.json", tmp_path / "closures.json"
    hs.write_text(json.dumps({k: {"x": "h"} for k in C.HASH_KEYS}), encoding="utf-8")
    sg.write_text(json.dumps(dict(PASS=True, foreign_files=0)), encoding="utf-8")
    with pytest.raises(SystemExit):                                       # A.10.6: hashes and seed grep required
        C.main(["f1", "--out", str(tmp_path), "--closures", str(clo)])
    clo.write_text(json.dumps(dict(closures={"v2s": "h_v2s", "o4": "h_o4"})), encoding="utf-8")
    with pytest.raises(SystemExit):                                       # closures per arm type: all gated arms
        C.main(["f1", "--out", str(tmp_path), "--closures", str(clo), "--hashes", str(hs), "--seed-grep", str(sg)])
    clo.write_text(json.dumps(dict(closures={a: f"h_{V.canon(a)}" for a in C.GATE_ARMS}, unit="g0")), encoding="utf-8")
    C.main(["f1", "--out", str(tmp_path), "--closures", str(clo), "--hashes", str(hs), "--seed-grep", str(sg)])
    f1 = json.loads((tmp_path / "f1_thresholds.json").read_text(encoding="utf-8"))
    assert f1["budget"]["mean_v2s_billed"] == 4e6 and f1["n_O4"] == 4 and f1["istage_stage"] == "PASS"
    assert f1["hashes"]["starts"]["g0/A"] == dict(s1="s1g0A", entry="eg0A") and f1["units"] == units
    rs = [r for r in synthetic([2.6] * 4, arm="O_4", n_arm=2) if r["arm"] != "V2s"]
    rs += [r for r in synthetic([1.0] * 4, arm="K*map", n_arm=2) if r["arm"] != "V2s"]
    rs += [r for r in synthetic([1.0] * 4, arm="K*mac", n_arm=2) if r["arm"] != "V2s"]
    for r in rs:
        if r["arm"] == "K*map":
            r["instrument"] = dict(coins=100, found=10)
    gf4 = [dict(r, arm="GF4", provenance=dict(dirty=False, closure_hash="h_gf4"), T={k: t * 2 for k, t in r["T"].items()})
           for r in rs if r["arm"] == "O_4"]
    write_runs(tmp_path / "g1", rs[1:])                                   # one O_4 run missing
    V.main(["g1", "--out", str(tmp_path)])
    g1 = json.loads((tmp_path / "g1" / "g1_verdict.json").read_text(encoding="utf-8"))
    assert g1["G1"] == "INCOMPLETE" and g1["rows"] == {} and g1["missing"] == ["g0/A/O_4/9812"]
    write_runs(tmp_path / "g1", rs)
    V.main(["g1", "--out", str(tmp_path)])
    g1 = json.loads((tmp_path / "g1" / "g1_verdict.json").read_text(encoding="utf-8"))
    assert g1["G1"] == "ROOM" and g1["rows"]["o4"]["label"] == "ROOM-K" and g1["interruptions"] == []
    assert g1["units"] == units and g1["certificate_unit"] == "g0"
    write_runs(tmp_path / "g1p", gf4[:-1])
    V.main(["g1p", "--out", str(tmp_path)])
    g1p = json.loads((tmp_path / "g1p" / "g1p_verdict.json").read_text(encoding="utf-8"))
    assert g1p["status"] == "INCOMPLETE" and g1p["routing"] is None   # a placebo run missing: no routing
    write_runs(tmp_path / "g1p", gf4)
    V.main(["g1p", "--out", str(tmp_path)])
    g1p = json.loads((tmp_path / "g1p" / "g1p_verdict.json").read_text(encoding="utf-8"))
    assert g1p["status"] == "COMPLETE" and g1p["rows"]["o4"]["label"] == "SEPARATED-K"
    assert g1p["routing"]["families"] == ["W"]
    # G2a for the licensed family W (V2s reused from the I-stage), then G2b
    write_runs(tmp_path / "g2a", fam_runs({"K_S": [3.0] * 4, "F_S": [1.0] * 4}, {"K_S": 4, "F_S": 6}, V.G2_SEEDS, ks=K))
    V.main(["g2a", "--out", str(tmp_path)])
    ga = json.loads((tmp_path / "g2a" / "g2a_verdict.json").read_text(encoding="utf-8"))
    assert ga["families"]["W"]["status"] == "Interruption"                # family closures not committed at F1
    f1["closure_hashes"].update({a: f"h_{V.canon(a)}" for a in ("K_S", "F_S", *V.FAM["W"]["g2b_runs"])})
    (tmp_path / "f1_thresholds.json").write_text(json.dumps(f1), encoding="utf-8")
    V.main(["g2a", "--out", str(tmp_path)])
    ga = json.loads((tmp_path / "g2a" / "g2a_verdict.json").read_text(encoding="utf-8"))
    assert ga["families"]["W"]["label"] == "OBJECT-ROOM" and ga["g2b_family"] == "W"
    fx = {a: [1.0] * 4 for a in V.FAM["W"]["g2b_runs"]}
    fx.update(X_S=[3.0] * 4)
    write_runs(tmp_path / "g2b", fam_runs(fx, V.FAM["W"]["g2b_runs"], V.G2_SEEDS, ks=K))
    V.main(["g2b", "--out", str(tmp_path)])
    gb = json.loads((tmp_path / "g2b" / "g2b_verdict.json").read_text(encoding="utf-8"))
    assert gb["status"] == "COMPLETE" and gb["label"] == "GO" and gb["interruptions"] == []
    assert gb["stats"]["X:F"]["R"] == pytest.approx(3.0) and gb["phi"] == pytest.approx(1.0)
    # an I-stage that is not PASS stops F1 (§10: INSTRUMENT-FAIL / D stop before G1)
    iv["stage"] = "INSTRUMENT-FAIL"
    (tmp_path / "istage" / "istage_verdict.json").write_text(json.dumps(iv), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.main(["f1", "--out", str(tmp_path), "--closures", str(clo), "--hashes", str(hs), "--seed-grep", str(sg)])


# ------------------------------------------------------------------ review fixes (G1-P routing, S0 drops, order,
# completeness, D control, budget, closures, captions, G2 / B1)

def test_routing_needs_g1p_labels_and_g1p_needs_room():
    room = dict(o4="ROOM-K", kmap="NO-ROOM-K", kmac="NO-ROOM-K")
    with pytest.raises(ValueError):                                       # no G1-P label: no licence
        V.routing(room, dict(o4=None))
    with pytest.raises(ValueError):
        V.routing(room, {})
    assert V.routing(room, dict(o4="NOT-SEPARATED-K"))["families"] == ["W"]
    assert V.routing(room, dict(o4="GENERIC-K"))["result"] == "ROOM-UNCARRIED"
    # g1p_report on g1_report's own format: rows {"o4": {"label": ...}}
    rs = synthetic([2.6] * 4, arm="O_4", n_arm=2)
    gf4 = [dict(r, arm="GF4", provenance=dict(dirty=False, closure_hash="h_gf4")) for r in rs if r["arm"] == "O_4"]
    v = dict(G1="ROOM", rows=dict(o4=dict(label="ROOM-K"), kmap=dict(label="NO-ROOM-K"), kmac=dict(label="NO-ROOM-K")))
    rep = V.g1p_report(v, V.group(rs + gf4), kgs(4), B=200)
    assert rep["status"] == "COMPLETE" and rep["rows"]["o4"]["label"] == "GENERIC-K" and rep["routing"]["families"] == []
    assert V.g1p_report(v, V.group(rs), kgs(4), B=200)["routing"] is None             # no placebo runs
    assert V.g1p_report(v, V.group(rs + gf4), kgs(4), missing=["g0/A/GF4/9812"], B=200)["status"] == "INCOMPLETE"
    for g1 in ("NO-ROOM", "INCONCLUSIVE", "Interruption", "INCOMPLETE"):
        rep = V.g1p_report(dict(v, G1=g1), V.group(rs + gf4), kgs(4), B=200)
        assert rep["status"] == "NOT-RUN" and rep["routing"] is None
    # interruptions only on the ROOM rows' K runs and their placebos: a caption row's closure never counts
    dice = [dict(r, arm="O_4-dice", provenance=dict(dirty=False, closure_hash=f"h_dice{i}"))
            for i, r in enumerate(x for x in rs if x["arm"] == "O_4")]
    com = dict(o4="h_o4", gf4="h_gf4")
    rep = V.g1p_report(v, V.group(rs + gf4 + dice), kgs(4), com, B=200, require_committed=True)
    assert rep["interruptions"] == [] and rep["status"] == "COMPLETE"
    rep = V.g1p_report(v, V.group(rs + gf4), kgs(4), None, B=200, require_committed=True)
    assert rep["status"] == "Interruption" and rep["routing"] is None


def test_s0_dropped_units_and_istage_cells():
    starts = fake_starts(["g0", "g1"], dropped=["gx"])
    starts["units"]["g9"] = dict(starts["units"]["g0"])                  # kept, but outside the I-stage
    starts["units"]["g9"]["B"] = dict(valid=False, E=9820)
    starts["order"].append("g9")
    starts["kept"].append("g9")
    ur = V.unit_rungs({}, starts)                                         # K_g [] and k_ref None: no TypeError
    assert set(ur) == {"g0", "g1", "g9"} and ur["g0"] == ([2, 3, 4], 4)
    assert V.unit_rungs({}, {"units": {"g": {"K_g": [3], "k_ref": 3}}}) == {}          # |K_g| < 2: no unit
    runs = V.group(istage_runs([1.0, 1.0]))
    K_gs, det = V.istage_kgs(runs, starts, V.start_validity(starts))
    assert set(K_gs) == {("g0", "A"), ("g0", "B"), ("g1", "A"), ("g1", "B")}       # no g9 cell, no gx
    assert V.a2_order(starts, {"g1", "g9", "g0"}) == ["g0", "g1", "g9"]


def test_a2_order_is_not_a_repermutation_and_certificate_unit():
    from ragnarok.evaluation.start_v85 import unit_order
    games = [f"u{i:02d}" for i in range(16)]
    order = unit_order(games)
    starts = dict(order=order, kept=order[1:], units={g: dict(kept=g != order[0]) for g in order})
    sub = set(order[1:11])
    assert V.a2_order(starts, sub) == order[1:11]
    assert unit_order(sub) != order[1:11]                                 # what the old code did
    ist = [dict(game=g, start="A", arm="V2s", seed=9806) for g in (order[3], order[2])]
    assert V.certificate_unit(starts, ist) == order[2]
    with pytest.raises(SystemExit):
        V.a2_order(None, sub)


def test_completeness_istage_and_g1():
    starts = fake_starts(["g0", "g1", "g2"], invalid_b=["g2"])
    rs = istage_runs([1.0, 1.0])                                         # g2 not run yet
    units, stray = V.istage_units(starts, rs, U=3)
    assert units == ["g0", "g1", "g2"] and stray == []
    miss = V.missing_istage(V.group(rs), starts, units)
    assert miss == ["g2/A/V2s/9806", "g2/A/V2s/9807", "g2/A/V2s/9808", "g2/A/D/9819"]   # B invalid: nothing there
    assert V.istage_units(starts, rs, U=None)[0] == ["g0", "g1"]        # U unknown: units with runs (in order)
    rep = V.istage_report(V.group(rs), ["g0", "g1", "g2"], starts, valid=V.start_validity(starts), missing=miss, B=50)
    assert rep["stage"] == "INCOMPLETE" and rep["K_gs"] is None
    # cross-fit needs all 3 V2s runs at the other start: 2 runs give no K (not a drop on fewer runs)
    lad = [0, 10, 20, 40, 80, 160]
    rec = dict(cap=CAP, ladder=lad, records=[[1000 * i, lad[i] + 1, 0, i, 0] for i in range(1, 6)])
    assert V.cross_fit([2, 3, 4], 4, [rec, rec])["K"] == [] and V.cross_fit([2, 3, 4], 4, [rec] * 3)["K"] == [2, 3, 4]
    # G1: n/2 per row per kept (g, s), O_4 at n_O4, K*mac only where the other start is valid
    K_gs = {("g0", "A"): [2, 3], ("g0", "B"): [2, 3], ("g2", "A"): [2, 3]}
    exp = V.g1_expected(K_gs, starts, n=6, n_O4=4)
    assert exp[("g0", "A", "o4")] == [9812, 9813] and exp[("g0", "B", "kmap")] == [9815, 9816, 9817]
    assert ("g2", "A", "kmac") not in exp and exp[("g2", "A", "o4")] == [9812, 9813]
    rep = V.g1_report({}, K_gs, dict(n=4, tau_pt=1.8, tau_lb=1.25), missing=["g0/A/O_4/9812"])
    assert rep["G1"] == "INCOMPLETE" and rep["rows"] == {}
    with pytest.raises(ValueError):                                       # cells of exactly 3 V2s runs
        C.prepare({("g0", "A"): np.array([[1e5, 2e5], [1.5e5, 3e5]])}, ["g0"], require_m=3)
    with pytest.raises(ValueError):                                       # one (game, start, arm, seed) twice
        V.group(rs + rs[:1])


def test_d_control_scores_cross_fit_dropped_units_on_K_g():
    rs = istage_runs([1.0] * 5, d_speed=8.0)
    for r in rs:                                                          # one V2s run per start misses rung 4
        if r["arm"] == "V2s" and r["game"] in ("g0", "g1") and r["seed"] in (9807, 9810):
            r["T"]["4"] = None
    runs = V.group(rs)
    units = [f"g{u}" for u in range(5)]
    K_gs, _ = V.istage_kgs(runs)
    K_gs = {k: v for k, v in K_gs.items() if k[0] not in ("g0", "g1")}  # as if A.10.4 dropped g0, g1
    d = V.d_control(runs, K_gs, units, {g: [2, 3, 4] for g in units})
    assert d["passed"] and d["share"] == 1.0 and d["uncross_fitted_units"] == ["g0", "g1"]
    assert not V.d_control(runs, K_gs, units)["passed"]                   # the old rule: 3/5 < 0.8


def test_istage_stages_d_fix_interruptions():
    starts = fake_starts([f"g{u}" for u in range(5)])
    rs = istage_runs([1.0] * 5, d_speed=2.0)                               # D only 2x: the D control fails
    runs = V.group(rs)
    units = [f"g{u}" for u in range(5)]
    rep = V.istage_report(runs, units, starts, valid=V.start_validity(starts), B=50)
    assert rep["stage"] == "D-CONTROL-FAIL"
    rep = V.istage_report(runs, units, starts, valid=V.start_validity(starts), B=50, d_fix="2026-11-04")
    assert rep["stage"] == "Interruption (D)" and rep["d_fix"] == "2026-11-04"
    for r in rs:                                                          # D reruns launched after the fix
        if r["arm"] == "D":
            r["provenance"] = dict(r["provenance"], launched="2026-11-04 09:00:00")
    assert V.stale_d(runs, "2026-11-04") == [] and V.stale_d(runs, None) == []
    runs[("g0", "A", "d")][0]["provenance"]["launched"] = "2026-11-03 22:00:00"   # a first-pass D left in istage/
    assert len(V.stale_d(runs, "2026-11-04")) == 1
    dirty = [dict(r, provenance=dict(dirty=True, closure_hash="h_v2s")) if r["arm"] == "V2s" and r["seed"] == 9806
             else r for r in istage_runs([1.0] * 5)]
    rep = V.istage_report(V.group(dirty), units, starts, valid=V.start_validity(starts), B=50)
    assert rep["stage"] == "Interruption" and rep["K_gs"] is None
    dmis = [dict(r, replay=[dict(k=2, ok=False)], interruption=True) if r["arm"] == "D" and r["game"] == "g0" else r
            for r in istage_runs([1.0] * 5)]
    rep = V.istage_report(V.group(dmis), units, starts, valid=V.start_validity(starts), B=50)
    assert rep["interruptions"] == [] and rep["stage"] == "D-CONTROL-FAIL"   # D's confirmation: the D control's
    assert V.interruptions(dmis, None, require_committed=True)[0].startswith("no committed closure hashes")
    rep = V.istage_report(V.group(istage_runs([1.0] * 5)), units, starts, valid=V.start_validity(starts), B=50,
                          committed=dict(v2s="h_new", d="h_d"))           # V2s changed (a D fix touched it)
    assert rep["stage"] == "Interruption" and rep["interruptions"] == ["missing comparator certificate: V2s closure changed"]
    assert V.interruptions([r for r in dmis if r["arm"] == "V2s"], dict(v2s="h_v2s", d="h_d")) == []
    assert V.interruptions(dmis, dict(v2s="h_v2s"), require_committed=True) == [
        "replay mismatch: g0/A/D/9819", "replay mismatch: g0/B/D/9820", "no committed closure hash: D"]


def test_budget_projection_and_n_O4():
    starts = fake_starts([f"g{u}" for u in range(12)], invalid_b=["g11"])
    K_gs = {(f"g{u}", s): [2, 3] for u in range(12) for s in "AB" if not (u == 11 and s == "B")}
    v2s = [dict(game="g0", start="A", arm="v2s", seed=9806, billed=1_000_000, stopped=True),
           dict(game="g0", start="A", arm="v2s", seed=9807, billed=3_000_000, stopped=True),
           dict(game="g0", start="A", arm="v2s", seed=9808, billed=4_000_100, stopped=False)]
    b = C.budget_projection(v2s, K_gs, starts, n=6)
    mb = (1e6 + 3e6 + 4e6) / 3                                            # the unstopped run at the 4M cap
    cells, kmac = 23, 22                                                  # g11/A has no K*mac (B invalid)
    assert b["mean_v2s_billed"] == pytest.approx(mb) and (b["cells"], b["kmac_cells"]) == (cells, kmac)
    assert b["projection_frames"] == pytest.approx(mb * (3 * 5 * cells + 3 * (cells + kmac)))
    assert b["projection_frames"] > 0.6e9 and b["n_O4"] == 4
    assert C.budget_projection(v2s[:1], K_gs, starts, n=6)["n_O4"] == 6   # 1M per run: below 0.6G
    f1 = C.assemble_f1(dict(U=12, units=[], outcome="SELECTED", selection=dict(n=6)), K_gs, b)
    assert f1["n_O4"] == 4
    assert V.row_thr(dict(n=6, n_O4=4, tau_pt=1.8, tau_lb=1.25), "o4")["n"] == 4
    assert V.row_thr(dict(n=6, n_O4=4, tau_pt=1.8, tau_lb=1.25), "kmap")["n"] == 6


def test_arm_closures_producer(tmp_path):
    import json
    (tmp_path / "s0").mkdir()
    st = fake_starts(["g0", "g1"], invalid_b=["g0"])
    (tmp_path / "s0" / "starts.json").write_text(json.dumps(st), encoding="utf-8")
    calls = []

    def runner(v85, unit, arm, seed):
        calls.append((unit, arm, seed))
        return dict(closure_hash=f"h_{arm}", closure_files=[], closure_foreign=[])
    c = C.arm_closures(tmp_path, runner=runner)
    assert c["unit"] == "g1" and set(c["closures"]) == set(C.arm_modules()) >= set(C.GATE_ARMS)   # every arm module
    assert ("g1", "d", 9819) in calls and ("g1", "o4_dice", 9818) in calls and len(calls) == len(C.arm_modules())
    with pytest.raises(RuntimeError):
        C.arm_closures(tmp_path, runner=lambda *a: dict(closure_hash="h", closure_foreign=["ragnarok.x: elsewhere"]))


def test_dice_caption_and_instrument_summary():
    rs = synthetic([2.0] * 4, arm="O_4", n_arm=2)
    dice = []
    for r in rs:
        if r["arm"] == "O_4" and r["seed"] in (9812, 9815):
            d = dict(r, arm="O_4-dice", seed=9818, aborted=None)
            if r["game"] == "g0":                                          # aborted early: censored if kept
                d.update(aborted="DiceMismatch: expansion 7", T={k: None for k in r["T"]})
            dice.append(d)
    g4 = dict(G4=1.9, G4_dice_starts=1.8, per_unit={f"g{u}": {"A": 1.5 + u / 10, "B": 2.5} for u in range(4)})
    cap = V.dice_caption(V.group(rs + dice), kgs(4), B=100, g4=g4)
    assert cap["aborted"] == 2 and cap["runs"] == 8 and len(cap["aborted_runs"]) == 2
    clean = V.contrast(V.group(rs + [d for d in dice if not d["aborted"]]), kgs(4), "O_4-dice", "v2s", gate="G1C", B=100)
    assert cap["R"] == clean["R"] and cap["units"] == ["g1", "g2", "g3"]    # aborted runs left out of R
    assert cap["G4_pred_cells"] == pytest.approx(np.mean([1.6, 1.7, 1.8, 2.5, 2.5, 2.5])) and cap["G4_S0"] == 1.9
    ins = [dict(game="g0", start="A", instrument=dict(fidelity=0.5, size=10, hold=4, bandit_k=5, q5_expansions=10)),
           dict(game="g0", start="A", instrument=dict(fidelity=0.7, size=10, hold=4, bandit_k=5, q5_expansions=30)),
           dict(game="g1", start="B", instrument=dict(fidelity=0.9, size=3, hold=4, bandit_k=4, q5_expansions=0))]
    s = V.instrument_summary(ins)
    assert s["fidelity"] == pytest.approx(0.7) and s["q5_expansions"] == pytest.approx(40 / 3)
    assert "size" not in s and "hold" not in s and "bandit_k" not in s
    assert s["per_start"] == {"g0/A": dict(size=10, hold=4, bandit_k=5), "g1/B": dict(size=3, hold=4, bandit_k=4)}


def fam_runs(factors, plan, seeds, ks=KS):
    """Runs of every arm in plan {arm: runs per unit} at the §12 seed sets; arm a's T = the (unit, start)'s base T
    / factors[a][u] for every run (no run noise: every R and the interaction are exact)."""
    rng = np.random.default_rng([9800, 7])
    out = []
    for u in range(len(next(iter(factors.values())))):
        for s in "AB":
            base = np.cumsum(rng.uniform(1e5, 6e5, size=len(ks)))
            for arm, n in plan.items():
                for sd in seeds[n][s]:
                    f = factors.get(arm, [1.0] * 99)[u]
                    out.append(mk(f"g{u}", s, arm, sd, (base / f).tolist(), ks, rungs=list(ks), k_ref=max(ks)))
    return out


def as_v2s(rs, arm):
    """V2s runs (9806.. / 9809..) equal to `arm`'s 6-run G2 runs (9826.. / 9829..)."""
    m = dict(zip(V.G2_SEEDS[6]["A"] + V.G2_SEEDS[6]["B"], V.V2S_SEEDS["A"] + V.V2S_SEEDS["B"]))
    return [dict(r, arm="V2s", seed=m[r["seed"]]) for r in rs if r["arm"] == arm]


def test_g2a_g2b_and_censoring_sensitivity():
    U = 6
    fac = {"K_mine": [2.2] * U, "F_M": [1.0] * U}
    runs = fam_runs(fac, {"K_mine": 4, "F_M": 6}, V.G2_SEEDS)
    runs += as_v2s(runs, "F_M")
    rep = V.g2a_report(V.group(runs), kgs(U), "M", B=300)
    assert rep["label"] == "OBJECT-ROOM" and rep["stats"]["K:F"]["R"] == pytest.approx(2.2, rel=1e-9)
    assert rep["sealed_top_label"] == "FASTER-2X" and rep["captions"] == []
    assert V.g2a_report(V.group(runs), kgs(U), "M", missing=["x"])["status"] == "INCOMPLETE"
    assert V.g2b_family({"M": rep, "W": dict(rep, label="KC-object")}) == "M"
    assert V.g2b_family({"M": rep, "W": rep}) == "M"                      # a tie goes to M
    # CENSORING-SENSITIVE: a claim that changes with misses at 8M drops one step
    assert V.sensitive("OBJECT-ROOM", "KC-object") == dict(label="KC-object", primary="OBJECT-ROOM",
                                                          label_8M="KC-object", captions=["CENSORING-SENSITIVE"])
    assert V.sensitive("FASTER-2X", "FASTER-PARTIAL")["label"] == "FASTER-PARTIAL"
    assert V.sensitive("FASTER-PARTIAL", "NULL")["label"] == "NOT-SIGNIFICANT"
    assert V.sensitive("NULL", "FASTER-PARTIAL")["label"] == "NULL"
    # G2b family M: X_on 2x faster than F_M and every comparator; interaction exact on noiseless runs
    A = V.FAM["M"]
    f = {a: [1.0] * U for a in A["g2b_runs"]}
    f.update(X_on=[3.0] * U, X_frozen=[1.5] * U, X4_on=[1.2] * U, F_M=[1.0] * U)
    plan = dict(A["g2b_runs"], F_M=6, K_mine=4)
    rs = fam_runs(f, plan, V.G2_SEEDS)
    v2s = as_v2s(rs, "F_M")                                               # lam(V2s) - lam(F_M) = 0
    rep = V.g2b_report(V.group(rs + v2s), kgs(U), "M", B=300, R_KF=2.0)
    assert rep["status"] == "COMPLETE" and rep["label"] == "GO" and rep["clauses"]["vs_P_on"]
    assert rep["phi"] == pytest.approx(math.log(3.0) / math.log(2.0), rel=0.05)
    I = rep["interaction"]                                                # [lam(X_frozen) - lam(X_on)] - 0 = log 2
    assert I["I"] == pytest.approx(math.log(2.0), abs=1e-9) and I["LB90"] <= I["I"] <= I["UB90"]
    assert rep["vocab"] == "ADAPTS" and rep["accelerates_in_run_learning"]
    assert rep["dose"]["X4:F"]["R"] == pytest.approx(1.2, rel=1e-6)


def test_b1_ladder_p_value_and_page():
    assert V.sign_flip_p([1.0] * 6) == pytest.approx(1 / 64)
    assert V.sign_flip_p([1.0, -1.0]) == pytest.approx(0.75)
    pl = V.page_L(np.array([[4.0, 3.0, 2.0, 1.0]] * 6))                   # lam falls with dose: perfect order
    assert pl["L"] == 6 * (1 + 4 + 9 + 16) and pl["p"] < 0.001
    st = lambda R, LB, UB=3.0, late=1.2: dict(R=R, LB90=LB, UB95=UB, LB90_late=late)   # noqa: E731
    good = {"V2s": st(1.8, 1.2), "DM_on": st(1.7, 1.1), "S": st(1.6, 1.1), "P_on": st(1.5, 1.05)}
    tv = {"DM_on": st(1.0, 0.9), "S": st(1.0, 0.9)}
    assert V.b1_label(st(2.1, 1.4), 0.01, 1.2, True, good, tv) == "FASTER-2X"
    assert V.b1_label(st(2.1, 1.4), 0.01, 1.2, False, good, tv) == "FASTER-PARTIAL"   # ceiling rule
    assert V.b1_label(st(2.1, 1.4, late=1.0), 0.01, 1.2, True, good, tv) == "EARLY-RUNGS-ONLY"
    assert V.b1_label(st(0.7, 0.5, UB=0.9), 0.9, 0.6, True, good, tv) == "NEGATIVE-TRANSFER"
    bad_t = dict(good, S=st(1.1, 0.95))
    assert V.b1_label(st(1.5, 1.2), 0.01, 1.2, True, bad_t, dict(tv, S=st(1.4, 1.1))) == "TUNED-DEFAULTS"
    assert V.b1_label(st(1.5, 1.2), 0.01, 1.2, True, dict(good, P_on=st(1.1, 0.9)), tv) == "GENERIC"
    assert V.b1_label(st(1.3, 0.95), 0.2, 1.0, True, dict(good, P_on=st(1.1, 0.9)), tv) == "NOT-SIGNIFICANT"
    assert V.b1_label(st(1.1, 0.95), 0.2, 1.0, True, dict(good, P_on=st(1.1, 0.9)), tv) == "NULL"
    # end to end on sealed-like runs: X_on 2.5x faster than every other arm, family M
    U = 8
    A = V.FAM["M"]
    f = {a: [1.0] * U for a in A["b1_runs"]}
    f.update(X_on=[2.5] * U, X4_on=[1.3] * U, X12_on=[1.8] * U)
    rs = fam_runs(f, A["b1_runs"], V.B1_SEEDS)
    K_gs, _ = V.istage_kgs(V.group([r for r in rs if r["arm"] == "V2s"]))
    rep = V.b1_report(V.group(rs), K_gs, "M", allow_2x=False, B=300)
    assert rep["status"] == "COMPLETE" and rep["label"] == "FASTER-PARTIAL" and rep["p"] == pytest.approx(1 / 256)
    assert V.b1_report(V.group(rs), K_gs, "M", allow_2x=True, B=300)["label"] == "FASTER-2X"
    assert rep["accumulates"]["ACCUMULATES"] and rep["logo_min"] == pytest.approx(2.5, rel=1e-6)
    assert V.b1_report(V.group(rs), K_gs, "M", True, B=300, line=dict(instrument_fail=True))["label"] == "INCONCLUSIVE"


# ------------------------------------------------------------------ calibration: selection, smoke, runtime

def test_selection_rule_order():
    ok = dict(i=True, ii=True, iii=True, iv=True, v=True, vi=True, vii=True, viii=True)
    tab = [dict(n=n, tau_pt=tp, tau_lb=tl, criteria=dict(ok)) for n in (4, 6) for tp in C.TAU_PT for tl in C.TAU_LB]
    for t in tab:
        if t["n"] == 4 and t["tau_pt"] == 2.0:
            t["criteria"]["i"] = False
        if t["n"] == 4 and t["tau_pt"] == 1.8 and t["tau_lb"] == 1.35:
            t["criteria"]["ii"] = False
    s = C.select(tab)
    assert (s["n"], s["tau_pt"], s["tau_lb"]) == (4, 1.8, 1.25)
    for t in tab:
        if t["n"] == 4:
            t["criteria"]["vii"] = False
    s = C.select(tab)
    assert (s["n"], s["tau_pt"], s["tau_lb"]) == (6, 2.0, 1.35)
    for t in tab:
        t["criteria"]["viii"] = False
    assert C.select(tab) is None


def test_calibration_smoke_and_f1_assembly():
    cells, units = C.synthetic_cells(10, 4)
    base = C.prepare(cells, units)
    cal = C.calibrate(base, sims=12, B=200)
    assert len(cal["table"]) == 2 * 3 * 4 * 3 and cal["outcome"] in ("SELECTED", "TOP-UP")
    assert C.calibrate(base, sims=2, B=50, topped_up=True)["outcome"] in ("SELECTED", "UNDERPOWERED")
    t = next(x for x in cal["table"] if (x["n"], x["tau"], x["tau_pt"], x["tau_lb"]) == (4, 0.2, 1.6, 1.15))
    P = t["oc"]["P_room"]
    assert P["D2/stay"] >= P["D1.5/divide"] >= P["D1/divide"]
    st1 = C.simulate(base, sims=2, n_list=(4,), taus=(0.2,), B=50)
    st2 = C.simulate(base, sims=2, n_list=(4,), taus=(0.2,), B=50)
    assert np.array_equal(st1, st2)                                        # [9802, b] streams: reproducible
    p = C.p_room_x2(cells, units, sims=6, B=100)
    assert 0.0 <= p <= 1.0
    a12 = C.rerun_a12(cells, units, dict(n=4, tau_pt=1.8, tau_lb=1.25), v=2.0, tau=0.5, sims=4, B=50)
    assert set(a12["passed"]) == {"ii", "iii", "iv", "v", "vi"}
    f1 = C.assemble_f1(dict(cal, selection=dict(n=6, tau_pt=1.8, tau_lb=1.25, oc={}, criteria={}), outcome="SELECTED"),
                       {("u0", "A"): [2, 3]}, budget=dict(projection_frames=0.65e9))
    assert f1["n"] == 6 and f1["n_O4"] == 4 and f1["K_gs"] == {"u0": {"A": [2, 3]}} and f1["hashes"] is None
    assert V.kgs_from_json(f1["K_gs"]) == {("u0", "A"): [2, 3]}


def test_calibration_runtime_under_30_minutes():
    t = C.time_run(sims=4, U=16, K=5)                                     # worst case: U = 16, 5 counted rungs
    assert t["extrapolated_s"] < C.RUNTIME_LIMIT_S, t


def test_cli_b1_sealed(tmp_path):
    import json
    U = 6
    root = tmp_path / "sealed"
    (root / "s0").mkdir(parents=True)
    (root / "s0" / "starts.json").write_text(json.dumps(fake_starts([f"g{u}" for u in range(U)])), encoding="utf-8")
    A = V.FAM["M"]
    f = {a: [1.0] * U for a in A["b1_runs"]}
    f.update(X_on=[2.5] * U, X4_on=[1.3] * U, X12_on=[1.8] * U)
    write_runs(root / "b1", fam_runs(f, A["b1_runs"], V.B1_SEEDS))
    V.main(["b1", "--out", str(tmp_path), "--family", "M"])
    b = json.loads((root / "b1" / "b1_verdict.json").read_text(encoding="utf-8"))
    assert b["status"] == "Interruption" and b["label"] is None           # no F2 closures committed
    (root / "closures.json").write_text(json.dumps(dict(closures={a: f"h_{V.canon(a)}" for a in A["b1_runs"]})),
                                        encoding="utf-8")
    V.main(["b1", "--out", str(tmp_path), "--family", "M"])
    b = json.loads((root / "b1" / "b1_verdict.json").read_text(encoding="utf-8"))
    assert b["status"] == "COMPLETE" and b["label"] == "FASTER-PARTIAL" and not b["allow_2x"]   # no G2a ceiling
    assert b["instrument_line"]["instrument_fail"] is False and b["accumulates"]["ACCUMULATES"]
    assert set(b["K_gs"]) == {f"g{u}" for u in range(U)}
