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
                    out.append(mk(f"g{u}", s, arm, 9812 + i, (T / f).tolist(), ks))
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
    assert V.o_ties([dict(instrument=dict(o_ties=0.9, expansions_scored=300)),
                     dict(instrument=dict(o_ties=0.5, expansions_scored=100))]) == pytest.approx(0.8)
    assert rep["interruptions"] == [] and rep["claim_header"] == V.CLAIM   # caption rows never interrupt G1


def test_cli_istage_calib_f1_g1_g1p(tmp_path):
    import json
    K = [2, 3, 4]
    ist = [dict(r, rungs=K, k_ref=4) for r in synthetic([1.0] * 4, arm="D", n_arm=1)]
    for r in ist:
        if r["arm"] == "D":
            r["T"] = {k: v / 6 for k, v in r["T"].items()}
    for d, rs in (("istage", ist),):
        (tmp_path / d).mkdir()
        for i, r in enumerate(rs):
            (tmp_path / d / f"run{i}.json").write_text(json.dumps(r), encoding="utf-8")
    V.main(["istage", "--out", str(tmp_path)])
    iv = json.loads((tmp_path / "istage" / "istage_verdict.json").read_text(encoding="utf-8"))
    assert iv["d_control"]["passed"] and iv["K_gs"]["g0"] == {"A": K, "B": K} and iv["provenance"]["closure_hash"]
    C.main(["run", "--out", str(tmp_path), "--sims", "2"])
    cal = json.loads((tmp_path / "calib.json").read_text(encoding="utf-8"))
    assert cal["sims"] == 2 and cal["U"] == 4
    cal.update(outcome="SELECTED", selection=dict(n=4, tau_pt=1.8, tau_lb=1.25, oc={}, criteria={}))
    (tmp_path / "calib.json").write_text(json.dumps(cal), encoding="utf-8")
    clo = tmp_path / "closures.json"
    clo.write_text(json.dumps({"V2s": "h_v2s", "O_4": "h_o4", "K*map": "h_kmap", "K*mac": "h_kmac"}), encoding="utf-8")
    C.main(["f1", "--out", str(tmp_path), "--closures", str(clo)])
    rs = [r for r in synthetic([2.6] * 4, arm="O_4", n_arm=2) if r["arm"] != "V2s"]
    rs += [r for r in synthetic([1.0] * 4, arm="K*mac", n_arm=2) if r["arm"] != "V2s"]
    rs += [dict(r, arm="GF4", provenance=dict(dirty=False, closure_hash="h_gf4"), T={k: t * 2 for k, t in r["T"].items()})
           for r in rs if r["arm"] == "O_4"]
    (tmp_path / "g1").mkdir()
    (tmp_path / "g1p").mkdir()
    for i, r in enumerate(rs):
        (tmp_path / ("g1p" if r["arm"] == "GF4" else "g1") / f"run{i}.json").write_text(json.dumps(r), encoding="utf-8")
    V.main(["g1", "--out", str(tmp_path)])
    g1 = json.loads((tmp_path / "g1" / "g1_verdict.json").read_text(encoding="utf-8"))
    assert g1["G1"] == "ROOM" and g1["rows"]["o4"]["label"] == "ROOM-K" and g1["interruptions"] == []
    V.main(["g1p", "--out", str(tmp_path)])
    g1p = json.loads((tmp_path / "g1p" / "g1p_verdict.json").read_text(encoding="utf-8"))
    assert g1p["rows"]["o4"]["label"] == "SEPARATED-K" and g1p["routing"]["families"] == ["W"]


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
