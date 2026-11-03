"""Family W ("which proposal to bill", docs/ARC4_V85_PROTOCOL_EN.md §4.4) and the shared arm S: critic, steering,
objects, kappa rule and arms (Part B items 12-13; G-build test 6 "every W arm, until its first two-class refit").

Seeds: 9800 only (§12; smoke runs and tests, never a measurement). Object draws inside the builders use their own
§4.4/§12 seeds ([9803, 2], [9804, 1], ...), here on small inputs: real G1b streams truncated to their first
expansions (builders on a few small games) or synthetic streams (kappa rule). No full development object is built.

1. Prior application, exact on a synthetic case: the whitened objective equals v84 A.6's objective, the refit is
   its minimiser (Newton solve), kappa -> 0 gives the intercept-only fit; the both-classes rule and the refit
   schedule; inactive steering draws only V2s's P_{e,0} (no C_e draw).
2. Robust z, pick weights, kappa rule mechanics (selection, ties, edge rule once), kappa rule end to end on
   synthetic streams, dose seeds per unit pool and the closure-only X12_S on a development unit (review fix W-3,
   also through the job), the critic's object/kappa identity on resume (W-2), independent sealed dose draws per
   unit (W-1/M-5).
3. Builders on small real games: LOGO, integrity (trace d; P and U carry X's spectrum), the [9803, 2]
   permutation, T_clock heads only on the clock columns, U = diag(lambda_X), K_S from the unit's own streams,
   X4 draws from [9804, 1, i_g], S's q*, content hashes (tamper detection, rebuild determinism), the G1b npz
   check, development and sealed layouts (synthetic).
4. Through scripts/v85_job.py on the breakout smoke start: every W critic arm equals V2s (JSON apart from arm,
   instrument and volatile fields; exp_meta and exp_feat bytes) at a cap below the first refit; X_S equals V2s
   byte for byte on expansions 0..n*-1 (n* = its first two-class refit), then steers (C_e picks), and differs;
   the vectorised pair features equal Explorer2s._features (expected over the first-slot law); the steered
   pick follows P3 exactly; nothing bills between expansions (the restore charge assumption); a kill after the
   activation resumes byte-identically; S plays only q*.
"""
import hashlib
import json
import math
import pathlib
import shutil
import types

import numpy as np
import pytest
from scipy.special import expit

from ragnarok.agent_v84.consequence import sqrt_psd
from ragnarok.agent_v85 import critic_s as C
from ragnarok.agent_v85.critic_s import CriticS, SingleProcess, pair_features, pick_weights, robust_z
from ragnarok.evaluation import ckpt_v85 as ck
from ragnarok.evaluation import g0_v84 as G
from ragnarok.evaluation import objects_w_v85 as W
from ragnarok.evaluation import start_v85 as ST
from scripts import v85_job as J
from tests.test_job_v85 import Kill, stable

SEED, GAME = 9800, "breakout"
CAP_SHORT, CAP_LONG = 10_000, 90_000
SMALL = ("qbert", "beam_rider", "alien", "amidar", "pong", GAME)        # small real inputs (truncated streams)
N_MAX = 1500
HAVE_V84 = all((ST.V84 / "g1b" / f"{g}_{r}.npz").exists() for g in SMALL for r in (9712, 9713))
needs_v84 = pytest.mark.skipif(not HAVE_V84, reason="v84 G1b logs absent")
CRITICS = ("f_s", "x_s", "k_s", "p_s", "u_s", "t", "x4_s")               # development critic arms (x12_s: sealed)
KAPPA_TEST = dict(f_s=0.25, x_s=0.25, k_s=0.25, p_s=0.25, u_s=0.25, t=0.25)


# ---------------------------------------------------------------- helpers
class FakeEx:
    """The explorer attributes the critic reads in refits (exp_rows, exp_meta)."""

    def __init__(self, X, y):
        self.exp_rows = [np.asarray(x, np.float32) for x in X]
        self.exp_meta = [(e, 0, 0, int(v), 0, 0.0, 0, 100) for e, v in enumerate(y)]


def synthetic_critic(d=6, seed=1, kappa=0.3):
    rng = np.random.default_rng([SEED, seed])
    m, sd = rng.normal(size=44), rng.uniform(0.5, 2.0, 44)
    Q = np.linalg.qr(rng.normal(size=(44, d)))[0]
    A = rng.normal(size=(d, d))
    S = A @ A.T + 0.1 * np.eye(d)
    S *= d / np.trace(S)
    return CriticS(m, sd, Q, S, kappa, restore_frames=60), rng


def newton_map(Xh, y, Sinv, kappa, iters=60):
    """Minimiser of sum softplus(z) - y z + (1/(2 kappa)) v^T Sigma^-1 v, z = b + x_hat^T v (v84 A.6)."""
    n, d = Xh.shape
    Z = np.hstack([Xh, np.ones((n, 1))])
    P = np.zeros((d + 1, d + 1))
    P[:d, :d] = Sinv / kappa
    th = np.zeros(d + 1)
    for _ in range(iters):
        p = expit(Z @ th)
        g = Z.T @ (p - y) + P @ th
        H = (Z * (p * (1 - p))[:, None]).T @ Z + P
        th = th - np.linalg.solve(H, g)
    return th


def objective(Xh, y, Sinv, kappa, v, b):
    z = Xh @ v + b
    return float(np.sum(np.logaddexp(0, z) - y * z) + v @ Sinv @ v / (2 * kappa))


def synthetic_stream(game, seed, n, w, base=-0.8, rng=None):
    rng = rng or np.random.default_rng([SEED, 3, W.GI[game], seed])
    X = (rng.normal(size=(n, 44)) * np.linspace(0.3, 2.0, 44)).astype(np.float32)
    y = (rng.random(n) < expit(base + (X.astype(np.float64) / np.linspace(0.3, 2.0, 44)) @ w)).astype(np.int8)
    hs = np.array([np.frombuffer(hashlib.blake2b(x.tobytes(), digest_size=8).digest(), np.uint64)[0] for x in X],
                  np.uint64)
    return dict(X=X, y={W.HEAD: y}, block=np.arange(n) // 50, it=np.arange(n), q=rng.integers(0, 4, n),
                frames=rng.integers(400, 2000, n).astype(np.float64), hash=hs, game=game, seed=seed,
                sha256=f"synthetic-{game}-{seed}", n=n, truncated=True)


def synthetic_data(games, n=700, shared=True):
    rng = np.random.default_rng([SEED, 4])
    w0 = rng.normal(size=44) * 0.5
    out = {}
    for g in games:
        w = w0 if shared else rng.normal(size=44) * 0.5
        out[g] = [synthetic_stream(g, r, n, w) for r in W.SEEDS]
    return out


def expected_features(ex, cells, billed_now):
    """Explorer2s._features of each (cell, process) after the executed cell's chosen increment, with
    env.billed = billed_now + 60, averaged over the process's first-slot law."""
    A = ex.arch

    class Env:
        billed = billed_now + 60
    out = np.zeros((len(cells), 4, 44))
    for a, i in enumerate(cells):
        i = int(i)
        A.chosen[i] += 1
        try:
            for q in range(4):
                law = {s: 1.0 / len(ex.slots) for s in ex.slots}
                last = int(A.last_slot[i])
                if q == 3 and last >= 0:
                    law = {s: 0.5 * p for s, p in law.items()}
                    law[last] = law.get(last, 0.0) + 0.5
                out[a, q] = sum(p * ex._features(i, q, s, Env).astype(np.float64) for s, p in law.items())
        finally:
            A.chosen[i] -= 1
    return out


# ---------------------------------------------------------------- 1. critic, prior application
def test_prior_application_exact_on_a_synthetic_case():
    crit, rng = synthetic_critic()
    n = 100
    X = rng.normal(size=(n, 44)) * 1.5 + crit.m
    y = (rng.random(n) < 0.35).astype(np.int64)
    y[:2] = (0, 1)
    ex = FakeEx(X, y)
    Xf = np.asarray(ex.exp_rows, np.float64)
    Xh = ((Xf - crit.m) / crit.sd) @ crit.Q
    Sinv = np.linalg.inv(crit.Sigma)
    # the whitened objective is the A.6 objective: equal values at random points
    for _ in range(3):
        vp, b = rng.normal(size=crit.Q.shape[1]), float(rng.normal())
        Xw = Xh @ sqrt_psd(crit.Sigma)
        z = Xw @ vp + b
        whit = float(np.sum(np.logaddexp(0, z) - y * z) + vp @ vp / (2 * crit.kappa))
        assert whit == pytest.approx(objective(Xh, y, Sinv, crit.kappa, sqrt_psd(crit.Sigma) @ vp, b), rel=1e-12)
    for k in range(1, 100):
        crit.on_iter(ex, k)
    assert crit.theta is None and not crit.active
    crit.on_iter(ex, 100)
    assert crit.active and crit.active_from == 100 and crit.refits == 1
    v, b = crit.slopes()
    th = newton_map(Xh, y.astype(np.float64), Sinv, crit.kappa)
    np.testing.assert_allclose(np.r_[v, b], th, rtol=1e-3, atol=1e-4)                  # L-BFGS-B tolerance
    J_opt = objective(Xh, y, Sinv, crit.kappa, th[:-1], th[-1])
    assert J_opt - 1e-9 <= objective(Xh, y, Sinv, crit.kappa, v, b) <= J_opt + 1e-6 * abs(J_opt)
    np.testing.assert_allclose(crit.logit(Xf), Xh @ v + b, rtol=1e-10, atol=1e-10)
    # the same over the kappa range the rule can choose (grid and its one-step edge extensions); a small kappa
    # shrinks the slopes towards 0 and leaves the intercept near logit(mean y)
    norms = []
    for kap in (4.0 ** -6, 4.0 ** -3, 1.0, 4.0):
        ck_, _ = synthetic_critic(kappa=kap)
        ck_.on_iter(ex, 100)
        vk, bk = ck_.slopes()
        thk = newton_map(Xh, y.astype(np.float64), Sinv, kap)
        np.testing.assert_allclose(np.r_[vk, bk], thk, rtol=1e-3, atol=1e-4)
        norms.append(float(np.linalg.norm(vk)))
        if kap == 4.0 ** -6:
            assert bk == pytest.approx(math.log(y.mean() / (1 - y.mean())), abs=0.05)
    assert norms == sorted(norms) and norms[0] < 0.05 * norms[-1]
    # Sigma enters exactly: a prior concentrated on one x_hat axis keeps the slopes on that axis
    d = crit.Q.shape[1]
    S1 = np.full(d, 1e-9)
    S1[0] = d
    one = CriticS(crit.m, crit.sd, crit.Q, np.diag(S1), 1.0, restore_frames=60)
    one.on_iter(ex, 100)
    v1, _ = one.slopes()
    assert abs(v1[0]) > 1e-3 and np.max(np.abs(v1[1:])) < 1e-6


def test_both_classes_rule_refit_schedule_and_state():
    crit, rng = synthetic_critic(seed=2)
    X = rng.normal(size=(300, 44)) + crit.m
    y = np.zeros(300, np.int64)
    y[150] = 1                                       # one class among the first 100 rows
    ex = FakeEx(X, y)
    seen = []

    class Rec:
        def __init__(self, tag):
            self.tag = tag

        def random(self, *a, **k):
            seen.append(self.tag)
            return 0.0

    class Ex:
        def draw_cell(self, rng, e):
            assert rng.tag == "P"
            return 7

        class bandit:
            @staticmethod
            def pick(rng):
                assert rng.tag == "P"
                return 2
    for n in range(1, 101):
        crit.on_iter(ex, n)
    assert not crit.active and crit.skipped == 1 and crit.theta is None
    assert crit.pick_cell(Ex(), Rec("P"), Rec("C"), 100) == 7 and crit.pick_process(Ex(), Rec("P"), Rec("C"), 7) == 2
    assert seen == []                                # inactive: no C_e draw, V2s's own law on P_{e,0}
    for n in range(101, 200):
        crit.on_iter(ex, n)
    assert crit.refits == 0 and not crit.active      # no refit off the 100-expansion grid
    crit.on_iter(ex, 200)
    assert crit.active and crit.active_from == 200 and crit.refits == 1
    st = crit.to_state()
    twin, _ = synthetic_critic(seed=2)
    twin.from_state(st)
    assert twin.to_state().keys() == st.keys() and np.array_equal(twin.theta, crit.theta)
    crit.on_iter(ex, 300)
    twin.on_iter(ex, 300)
    assert np.array_equal(twin.theta, crit.theta) and crit.refits == 2


def test_robust_z_and_pick_weights():
    u = np.array([0.1, 0.2, 0.3, 0.4, 10.0])
    med, mad = 0.3, 0.1
    np.testing.assert_allclose(robust_z(u), (u - med) / (1.4826 * mad))
    p = pick_weights(u)
    w = np.exp(1.5 * np.clip((u - med) / (1.4826 * mad), -2, 2))
    np.testing.assert_allclose(p, w / w.sum())
    flat = np.array([0.5, 0.5, 0.5, 0.7, 0.2])       # MAD = 0: the clipped limit
    np.testing.assert_allclose(robust_z(flat), [0, 0, 0, 2, -2])
    assert np.allclose(pick_weights(np.full(8, 0.3)), 1 / 8)


def test_single_process_steer():
    sp = SingleProcess(3)

    class Ex:
        def draw_cell(self, rng, e):
            return 11
    assert sp.pick_cell(Ex(), None, None, 0) == 11 and sp.pick_process(Ex(), None, None, 11) == 3
    sp.from_state(sp.to_state())
    with pytest.raises(AssertionError):
        SingleProcess(4)


# ---------------------------------------------------------------- 2. kappa rule
def test_kappa_selection_ties_and_edge_rule(monkeypatch):
    grid = list(W.KAPPA_GRID)
    rows = [dict(game="g", dir=0, arm=a, kappa=k, mean_auc=0.6) for a in W.TUNED for k in grid]
    sel = W.select_kappa(rows, grid)
    assert all(sel[a] == grid[0] for a in W.TUNED)               # exact ties: the smaller kappa
    assert W.edge_extension(sel, grid) == [grid[0] / 4]
    mid = {a: 0.6 for a in W.TUNED}
    rows = [dict(r, mean_auc=mid[r["arm"]] + (0.01 if r["kappa"] == grid[2] else 0)) for r in rows]
    sel = W.select_kappa(rows, grid)
    assert all(sel[a] == grid[2] for a in W.TUNED) and W.edge_extension(sel, grid) == []
    rows = [dict(r, mean_auc=0.6 + (0.02 if (r["arm"] == "t" and r["kappa"] == grid[-1]) else 0)) for r in rows]
    sel = W.select_kappa(rows, grid)
    assert sel["t"] == grid[-1] and W.edge_extension(sel, grid) == [grid[0] / 4, grid[-1] * 4]

    # end to end with a curve that depends on kappa only: peak at 4^-6, beyond the frozen grid
    data = synthetic_data(("alien", "amidar", "pong", GAME), n=400)
    heads = W.source_heads(data, list(data))
    sums = W.game_sums(data, list(data))
    calls = []

    def curve(dd, S, k):
        calls.append(k)
        return [1.0 - abs(math.log(k, 4) + 6.0)]
    res = W.kappa_rule(data, heads, sums, games=("pong", GAME), pool=list(data), curve=curve)
    assert res["extended"] == [4.0 ** -6] and set(res["first_pass"].values()) == {4.0 ** -5}
    assert all(res["kappa"][a] == 4.0 ** -6 for a in W.TUNED if res["n_curves"][a])
    assert sorted(set(calls)) == sorted(set(res["kappas"]))      # extended once, never twice


def test_kappa_rule_end_to_end_on_synthetic_streams():
    data = synthetic_data(("alien", "amidar", "assault", "pong", GAME), n=500)
    heads = W.source_heads(data, list(data))
    sums = W.game_sums(data, list(data))
    res = W.kappa_rule(data, heads, sums, games=("pong", GAME), pool=list(data))
    assert set(res["kappa"]) == set(W.TUNED) and res["n_curves"] == {a: 4 for a in W.TUNED}
    assert all(res["kappa"][a] in res["kappas"] for a in W.TUNED) and not res["skipped"]
    assert res["derived"] == {"x4_s": "x_s", "x12_s": "x_s"}
    n_k = len(res["kappas"])
    assert len(res["rows"]) == 4 * len(W.TUNED) * n_k
    for a in W.TUNED:                                            # the selection is the argmax of the criterion
        means = {k: np.mean([r["mean_auc"] for r in res["rows"] if r["arm"] == a and r["kappa"] == k])
                 for k in res["kappas"]}
        best = max(means.values())
        assert res["kappa"][a] == min(k for k, v in means.items() if v >= best - 1e-15)
    again = W.kappa_rule(data, heads, sums, games=("pong", GAME), pool=list(data))
    assert json.dumps(again, sort_keys=True, default=float) == json.dumps(res, sort_keys=True, default=float)


def test_dose_draw_seeds():
    """§12 dose seeds per unit pool (review fix W-3): a development unit takes X4_S on the G2 dose seeds and X12_S
    only at 9826 (the F1 closure build, closure-only hooks); a sealed unit takes both only at the B1 2-run seeds
    9853/9856; a smoke run (9800) takes draw 1; anything else is refused."""
    dev, sealed = "alien", "battle_zone"
    assert [W.dose_draw("x4_s", s, unit=dev) for s in (9826, 9829, 9827, 9830)] == [1, 1, 2, 2]
    assert [W.dose_draw(a, s, unit=sealed) for a in ("x4_s", "x12_s") for s in (9853, 9856)] == [1, 1, 1, 1]
    assert W.dose_draw("x12_s", 9826, unit=dev) == 1                  # the F1 closure build (v85_calib.arm_closures)
    assert W.dose_draw("x4_s", SEED, True, unit=GAME) == W.dose_draw("x12_s", SEED, True, unit=GAME) == 1
    for arm, s, u in (("x4_s", 9828, dev), ("x12_s", 9827, dev), ("x12_s", 9829, dev), ("x4_s", SEED, dev),
                      ("x4_s", 9853, dev), ("x12_s", 9856, dev), ("x4_s", 9826, sealed), ("x12_s", 9826, sealed),
                      ("x4_s", 9854, sealed), ("x12_s", 9873, sealed), ("x12_s", SEED, sealed)):
        with pytest.raises(ValueError):
            W.dose_draw(arm, s, unit=u)
    with pytest.raises(TypeError):
        W.dose_draw("x4_s", 9826)                                     # the unit's pool decides: no default


def write_synthetic_objects(root, unit, arms_draws, kappa=KAPPA_TEST):
    """Synthetic family W objects of `unit` under the objects root `root` (hooks tests only), and kappa.json."""
    crit, _ = synthetic_critic()
    arrays = dict(m=crit.m, sd=crit.sd, Q=crit.Q, Sigma=crit.Sigma)
    for arm, draw in arms_draws:
        W.write_object(root, arm, unit, arrays, dict(sources=[], note="synthetic test object"), draw=draw)
    return W.write_kappa(root, dict(kappa=dict(kappa), note="test values, not the kappa rule"))


def wctx(out, game, seed, smoke=False):
    return types.SimpleNamespace(game=game, seed=seed, smoke=smoke, paths=dict(v85=str(out)))


def test_x12_s_on_a_development_unit_is_built_for_its_closure_only(tmp_path):
    """Review fix W-3: X12_S at 9826 on a development unit builds its hooks and its explorer exactly (the F1
    closure hash), but the explorer refuses to run; B1 and smoke runs, and X4_S on a G2 seed, are ordinary."""
    from ragnarok.agent_v85.explorer2s import Explorer2s
    R = tmp_path / "objects"
    write_synthetic_objects(R, "alien", [("x12_s", 1), ("x4_s", 1)])
    write_synthetic_objects(R, "battle_zone", [("x12_s", 1)])
    h = W.hooks_w(wctx(tmp_path, "alien", 9826), "x12_s")
    assert h["w_object"]["closure_only"] and h["w_object"]["draw"] == 1
    ex = h["explorer"](None, budget=1000, seed=9826, steer=h["steer"], q5=None, chooser=None, boundary=None)
    assert isinstance(ex, Explorer2s) and ex.steer is h["steer"]
    with pytest.raises(RuntimeError, match="closure hash only"):
        ex.run(None)
    for c, arm in ((wctx(tmp_path, "battle_zone", 9853), "x12_s"), (wctx(tmp_path, "alien", SEED, True), "x12_s"),
                   (wctx(tmp_path, "alien", 9826), "x4_s")):
        h = W.hooks_w(c, arm)
        assert not h["w_object"]["closure_only"] and "explorer" not in h, (c, arm)
    for c in (wctx(tmp_path, "battle_zone", 9826), wctx(tmp_path, "alien", 9853)):
        with pytest.raises(ValueError):
            W.hooks_w(c, "x12_s")


@needs_v84
def test_x12_s_closure_build_through_the_job_and_its_run_refused(wsmoke, tmp_path):
    """v85_calib's closure build (v85_job --closure-only, production mode, seed 9826) works on a development unit;
    the same run without --closure-only refuses before its first expansion and writes no run file."""
    d = tmp_path / "v85"
    shutil.copytree(wsmoke, d)
    write_synthetic_objects(d / "objects", GAME, [("x12_s", 1)])
    c = J.run(GAME, "A", "x12_s", 9826, v85=d, s0=d, closure_only=True)
    assert c["arm"] == "x12_s" and c["closure_hash"] and not c["closure_foreign"]
    out = d / "runs" / f"{GAME}_A_x12_s_9826.json"
    with pytest.raises(RuntimeError, match="closure hash only"):
        J.run(GAME, "A", "x12_s", 9826, v85=d, s0=d, out=out)
    assert not out.exists() and not out.with_suffix(".npz").exists()


def test_critic_identity_is_checked_on_resume(tmp_path):
    """Review fix W-2: the critic's state carries the identity of its object and kappa; a resume onto another
    object or kappa is refused (hooks_w: object content SHA-256, kappa.json SHA-256, kappa)."""
    crit, _ = synthetic_critic(seed=2)

    def twin(ident):
        return CriticS(crit.m, crit.sd, crit.Q, crit.Sigma, 0.3, 60, ident=ident)
    st = twin(("obj", "kap", 0.3)).to_state()
    assert st["ident"] == ("obj", "kap", 0.3)
    twin(["obj", "kap", 0.3]).from_state(st)                         # same identity: resumes
    for other in (("obj2", "kap", 0.3), ("obj", "kap2", 0.3), ("obj", "kap", 0.25), None):
        with pytest.raises(AssertionError, match="changed since the checkpoint"):
            twin(other).from_state(st)
    R = tmp_path / "objects"
    write_synthetic_objects(R, "alien", [("x_s", None)])
    h = W.hooks_w(wctx(tmp_path, "alien", 9812), "x_s")
    side = W.load_object(R, "x_s", "alien")[1]
    assert h["steer"].ident == (side["content_sha256"], W.load_kappa(R)[1], KAPPA_TEST["x_s"])
    st = h["steer"].to_state()
    W.hooks_w(wctx(tmp_path, "alien", 9812), "x_s")["steer"].from_state(st)          # unchanged files: resumes
    W.write_kappa(R, dict(kappa=dict(KAPPA_TEST, x_s=0.0625), note="rewritten after the checkpoint"))
    with pytest.raises(AssertionError, match="changed since the checkpoint"):
        W.hooks_w(wctx(tmp_path, "alien", 9812), "x_s")["steer"].from_state(st)


def test_sealed_dose_draws_are_independent_per_unit(tmp_path):
    """Review fix W-1/M-5: each sealed unit has its own X4_S and X12_S draw, default_rng([9804, 2 | 3, i_unit])
    (i_unit = family M's CANON_UNITS row), so two units with the same 41 sources do not share one game set."""
    from ragnarok.evaluation import objects_m_v85 as OM
    assert all(W.UI[g] == OM.CANON_UNITS.index(g) for g in OM.SEALED_UNITS) and all(W.UI[g] == W.GI[g] for g in W.GAMES)
    assert len(set(W.UI.values())) == len(W.UI) == 74
    many = W.DEV[:16]
    data = synthetic_data(many, n=300)
    units = ("battle_zone", "defender", "frogger")
    W.build_sealed(tmp_path, units, data=data, games=many, exclude={"frogger": ("amidar",)})
    drawn = {}
    for u in units:
        real = W.load_object(tmp_path, "x_s", u)[1]["heads"]
        for arm, p, size in (("x4_s", 2, 4), ("x12_s", 3, 12)):
            side = W.load_object(tmp_path, arm, u, 1)[1]
            assert side["seeds"] == [9804, p, W.UI[u]] and W.dose_sealed(u)[arm] == ((9804, p, W.UI[u]), size, 1)
            pick = W.dose_picks(len(real), size, np.random.default_rng([9804, p, W.UI[u]]), 1)[0]
            assert side["draw_sources"] == [real[i] for i in pick], (u, arm)
            drawn[u, arm] = side["draw_sources"]
    assert W.load_object(tmp_path, "x_s", "battle_zone")[1]["heads"] == W.load_object(tmp_path, "x_s", "defender")[1]["heads"]
    assert drawn["battle_zone", "x4_s"] != drawn["defender", "x4_s"]           # same sources, independent draws
    assert drawn["battle_zone", "x12_s"] != drawn["defender", "x12_s"]
    assert W.dose_dev("alien") == {"x4_s": ((9804, 1, W.GI["alien"]), 4, 2), "x12_s": ((9804, 3, W.GI["alien"]), 12, 1)}


# ---------------------------------------------------------------- 3. builders
@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Real G1b streams of SMALL truncated to N_MAX expansions, their heads, and breakout's objects
    (LOGO over the other 5)."""
    if not HAVE_V84:
        pytest.skip("v84 G1b logs absent")
    data = W.load_data(SMALL, n_max=N_MAX)
    heads = W.source_heads(data, list(SMALL))
    sums = W.game_sums(data, list(SMALL))
    root = tmp_path_factory.mktemp("wobj")
    src = [g for g in SMALL if g != GAME]
    built = W.build_unit(root, GAME, data, heads, sums, src, W.DEV_ARMS, {"x4_s": ((*W.SEED_X4_DEV, W.GI[GAME]), 4, 2)})
    W.write_kappa(root, dict(kappa=dict(KAPPA_TEST), note="test values, not the kappa rule"))
    return dict(data=data, heads=heads, sums=sums, root=root, src=src, built=built, hashes=W.write_manifest(root))


@needs_v84
def test_builders_on_small_games(small, tmp_path):
    data, heads, sums, root, src = small["data"], small["heads"], small["sums"], small["root"], small["src"]
    with pytest.raises(AssertionError):
        W.build_unit(tmp_path, GAME, data, heads, sums, list(SMALL), ("f_s",))     # LOGO
    geo = G.geometry(sums, src)
    d = geo["d"]
    obj = {a: W.load_object(root, a, GAME)[0] for a in ("f_s", "x_s", "k_s", "p_s", "u_s", "t")}
    for a, o in obj.items():
        assert np.array_equal(o["m"], geo["m"]) and np.array_equal(o["Q"], geo["Q"]) and o["Sigma"].shape == (d, d)
        assert abs(np.trace(o["Sigma"]) - d) < 1e-6 * d, a                         # every Sigma has trace d
    assert np.array_equal(obj["f_s"]["Sigma"], np.eye(d))
    real = [h for h in src if heads[(h, "real")] is not None]
    V = np.array([G.map_head(heads[(h, "real")], geo) for h in real])
    np.testing.assert_allclose(obj["x_s"]["Sigma"], G.sigma_from_dirs(V, d), rtol=0, atol=1e-12)
    lam = G.eig_desc(obj["x_s"]["Sigma"])[0]
    np.testing.assert_allclose(obj["u_s"]["Sigma"], np.diag(lam), atol=1e-12)
    np.testing.assert_allclose(G.eig_desc(obj["p_s"]["Sigma"])[0], lam, rtol=1e-9)
    side = W.load_object(root, "x_s", GAME)[1]
    assert side["integrity"]["trace_ok"] and side["integrity"]["spectrum_ok"] and side["sources"] == src
    assert set(side["inputs"]) == {f"{h}_{r}.npz" for h in src for r in W.SEEDS} and side["truncated"]
    # P_S: labels permuted within each source run with [9803, 2, i_h, r]
    h = src[0]
    lab = [G.permuted_labels(s["y"][W.HEAD], np.random.default_rng([9803, 2, W.GI[h], s["seed"]])) for s in data[h]]
    ref = G.source_head(data[h], W.HEAD, lab)
    assert np.array_equal(ref["w"], heads[(h, "perm")]["w"])
    assert all(np.array_equal(np.sort(a), np.sort(s["y"][W.HEAD])) and not np.array_equal(a, s["y"][W.HEAD])
               for a, s in zip(lab, data[h]))
    # T: clock-only heads
    for hh in src:
        hd = heads[(hh, "clock")]
        if hd is not None:
            off = np.setdiff1d(np.arange(44), W.CLOCK_COLS)
            assert np.all(hd["w"][off] == 0) and np.any(hd["w"][list(W.CLOCK_COLS)] != 0)
    # K_S: the unit's own streams, A.8 K*
    S_K = W.k_sigma(data[GAME], geo)
    assert np.array_equal(obj["k_s"]["Sigma"], S_K)
    lam_k = G.eig_desc(S_K)[0]
    np.testing.assert_allclose(lam_k[1:], lam_k[-1], rtol=1e-9)                       # norm[u u^T + tau^2 I]
    assert lam_k[0] > 2 * lam_k[-1]
    # X4_S: 2 draws of 4 from [9804, 1, i_g]
    picks = W.dose_picks(len(real), 4, np.random.default_rng([9804, 1, W.GI[GAME]]), 2)
    for k, pick in enumerate(picks, 1):
        o, s = W.load_object(root, "x4_s", GAME, k)
        assert s["draw_sources"] == [real[i] for i in pick] and s["seeds"] == [9804, 1, W.GI[GAME]]
        np.testing.assert_allclose(o["Sigma"], G.sigma_from_dirs(V[pick], d), atol=1e-12)
    # S: q* = argmax of the equal-game mean ADV per billed frame
    rates = []
    for hh in src:
        a, f = np.zeros(4), np.zeros(4)
        for st in data[hh]:
            for q in range(4):
                a[q] += st["y"][W.HEAD][st["q"] == q].sum()
                f[q] += st["frames"][st["q"] == q].sum()
        rates.append(a / f)
    o, s = W.load_object(root, "s", GAME)
    assert int(o["q_star"]) == int(np.argmax(np.mean(rates, 0))) and np.allclose(o["rate"], np.mean(rates, 0))
    # determinism: a rebuild gives the same content hashes and F1 hash
    again = W.build_unit(tmp_path, GAME, data, heads, sums, src, W.DEV_ARMS,
                         {"x4_s": ((*W.SEED_X4_DEV, W.GI[GAME]), 4, 2)})
    assert {k: v["content_sha256"] for k, v in again.items()} == {k: v["content_sha256"] for k, v in small["built"].items()}
    shutil.copy(root / "w" / "kappa.json", tmp_path / "w" / "kappa.json")
    assert W.write_manifest(tmp_path) == small["hashes"] == W.hashes(root)
    # tamper detection
    bad = tmp_path / "w" / "x_s" / f"{GAME}.npz"
    arr = dict(np.load(bad))
    arr["Sigma"] = arr["Sigma"] * 1.0000001
    np.savez(bad, **arr)
    with pytest.raises(RuntimeError):
        W.load_object(tmp_path, "x_s", GAME)
    with pytest.raises(FileNotFoundError):
        W.load_object(tmp_path, "x12_s", GAME, 1)


@needs_v84
def test_g1b_inputs_are_hash_checked(tmp_path):
    for ext in (".json", ".npz"):
        shutil.copy(ST.V84 / "g1b" / f"pong_9712{ext}", tmp_path / f"pong_9712{ext}")
    st = W.load_run("pong", 9712, tmp_path, n_max=50)
    assert st["n"] == 50 and st["frames"][0] == 460 and np.all(st["frames"] > 0)
    js = json.loads((tmp_path / "pong_9712.json").read_text(encoding="utf-8"))
    js["npz_sha256"] = "0" * 64
    (tmp_path / "pong_9712.json").write_text(json.dumps(js), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Interruption"):
        W.load_run("pong", 9712, tmp_path)


def test_development_and_sealed_layouts_on_synthetic_streams(tmp_path):
    games = ("alien", "amidar", "assault", "asterix", "atlantis", "pong", GAME)
    data = synthetic_data(games, n=400)
    out = W.build_development(tmp_path / "dev", ["alien"], data=data, games=games, kappa_games=("pong", GAME))
    names = sorted(p.relative_to(tmp_path / "dev" / "w").as_posix() for p in (tmp_path / "dev" / "w").glob("*/*.npz"))
    assert names == sorted([f"{a}/alien.npz" for a in ("f_s", "x_s", "k_s", "p_s", "u_s", "t", "s")]
                           + ["x4_s/alien_d1.npz", "x4_s/alien_d2.npz"])
    assert out["kappa"]["kappa"] == W.load_kappa(tmp_path / "dev")[0]["kappa"]
    man = json.loads((tmp_path / "dev" / "w" / "manifest.json").read_text(encoding="utf-8"))
    assert man["development_objects"] == out["hashes"]["development_objects"] and man["kappa_sha256"] == out["hashes"]["kappa"]
    many = W.DEV[:14]                                            # X12 needs >= 12 source heads
    data = synthetic_data(many, n=300)
    sealed = W.build_sealed(tmp_path / "sealed", ["frogger"], data=data, games=many, exclude={"frogger": ("amidar",)})
    names = sorted(p.relative_to(tmp_path / "sealed" / "w").as_posix() for p in (tmp_path / "sealed" / "w").glob("*/*.npz"))
    assert names == sorted([f"{a}/frogger.npz" for a in ("f_s", "x_s", "p_s", "u_s", "t", "s")]
                           + ["x4_s/frogger_d1.npz", "x12_s/frogger_d1.npz"])
    s4 = W.load_object(tmp_path / "sealed", "x4_s", "frogger", 1)[1]
    assert "amidar" not in s4["sources"] and s4["seeds"] == [9804, 2, W.UI["frogger"]] and len(s4["draw_sources"]) == 4
    s12 = W.load_object(tmp_path / "sealed", "x12_s", "frogger", 1)[1]
    assert s12["seeds"] == [9804, 3, W.UI["frogger"]] and len(set(s12["draw_sources"])) == 12 \
        and "amidar" not in s12["draw_sources"]
    assert sealed["built"]["frogger"]["x_s"]["sources"] == [g for g in many if g != "amidar"]


# ---------------------------------------------------------------- 4. through the job
@pytest.fixture(scope="module")
def wsmoke(small, tmp_path_factory):
    """The breakout smoke start (v85_job.build_smoke) with the small objects under <smoke>/objects."""
    d = J.build_smoke(tmp_path_factory.mktemp("wsmoke"), GAME)
    shutil.copytree(small["root"], d / "objects")
    return d


@pytest.fixture(scope="module")
def runs(wsmoke, tmp_path_factory):
    """Uninterrupted runs, once per (arm, cap): (JSON, path, observed run objects, pre-restore billed list)."""
    d, cache = tmp_path_factory.mktemp("wruns"), {}

    def get(arm, cap):
        if (arm, cap) not in cache:
            out, seen = d / f"{arm}_{cap}.json", {}

            def observe(o):
                seen.update(o)
                env, real, pre = o["env"], o["env"].restore, seen.setdefault("pre", [])

                def restore(h):
                    pre.append(int(env.billed))
                    return real(h)
                env.restore = restore
            J.run(GAME, "A", arm, SEED, out=out, v85=wsmoke, s0=wsmoke, cap=cap, smoke=True, observe=observe)
            cache[arm, cap] = (json.loads(out.read_text(encoding="utf-8")), out, seen)
        return cache[arm, cap]
    return get


def npz_arrays(path):
    z = np.load(pathlib.Path(path).with_suffix(".npz"))
    return z["exp_meta"], z["exp_feat"]


@needs_v84
@pytest.mark.parametrize("arm", CRITICS)
def test_every_w_arm_equals_v2s_before_its_first_refit(runs, small, arm):
    v, vp, _ = runs("v2s", CAP_SHORT)
    o, op, seen = runs(arm, CAP_SHORT)
    skip = set(J.VOLATILE) | {"arm", "instrument"}
    assert {k: x for k, x in o.items() if k not in skip} == {k: x for k, x in v.items() if k not in skip}
    assert all(np.array_equal(a, b) for a, b in zip(npz_arrays(op), npz_arrays(vp))) and 10 < v["summary"]["expansions"] < 100
    ins = o["instrument"]
    draw = 1 if arm == "x4_s" else None
    assert ins["content_sha256"] == W.load_object(small["root"], arm, GAME, draw)[1]["content_sha256"]
    assert ins["kappa_sha256"] == W.load_kappa(small["root"])[1] and ins["kappa"] == KAPPA_TEST[W.KAPPA_OF[arm]]
    assert not ins["active"] and ins["refits"] == 0 and ins["steered_picks"] == 0 and ins["truncated"]


@needs_v84
def test_x_s_identical_to_v2s_until_its_first_two_class_refit_then_steers(runs):
    v, vp, vs = runs("v2s", CAP_LONG)
    o, op, os_ = runs("x_s", CAP_LONG)
    vm, vf = npz_arrays(vp)
    om, of = npz_arrays(op)
    n = next(k for k in range(100, len(vm) + 1, 100) if 0 < vm[:k, 3].sum() < k)          # first two-class refit
    ins = o["instrument"]
    assert ins["active_from"] == n and ins["first_steered"] >= n and ins["steered_picks"] > 0
    assert ins["refits"] == len(range(n, len(om) + 1, 100)) and ins["refits_one_class"] == n // 100 - 1
    assert len(om) > n + 10, "the long run must go past the first refit"
    assert np.array_equal(om[:n], vm[:n]) and np.array_equal(of[:n], vf[:n])
    cut = int(vm[n - 1, 4]) - int(vs["env"].clock_start)                             # end of expansion n - 1
    assert [r for r in o["records"] if r[0] <= cut] == [r for r in v["records"] if r[0] <= cut]
    assert not (len(om) == len(vm) and np.array_equal(om, vm)), "steering never changed the run"
    assert 0.3 < ins["steered_share"] < 0.7                                              # the 0.5 C_e coin
    m = min(len(om), len(vm))
    diff = np.flatnonzero(np.any(om[:m] != vm[:m], axis=1))
    first_diff = int(diff[0]) if len(diff) else m
    assert first_diff >= n
    print(json.dumps(dict(x_s_activation=dict(n_star=n, expansions_x_s=len(om), expansions_v2s=len(vm),
                                              first_differing_expansion=first_diff, refits=ins["refits"],
                                              steered_picks=ins["steered_picks"], steered_share=ins["steered_share"],
                                              d=ins["d"]))))
    # nothing bills between two expansions: the critic's billed count is exact
    for seen in (vs, os_):
        meta = (vm if seen is vs else om)
        b0 = int(seen["ex"].b0)
        assert seen["pre"][:len(meta)] == [b0] + [int(b) for b in meta[:-1, 4]]


@needs_v84
def test_pair_features_reproduce_explorer_features(runs):
    v, vp, seen = runs("v2s", CAP_LONG)
    ex = seen["ex"]
    used = np.flatnonzero(ex.arch.used)
    cells = np.random.default_rng([SEED, 81]).choice(used, size=min(24, len(used)), replace=False)
    par = ex.arch.parent[used]
    cells = np.r_[cells, used[par == used][:3]]                                          # parent == cell cases
    billed = int(ex.exp_meta[-1][4])
    got = pair_features(ex, cells, 60)
    want = expected_features(ex, cells, billed)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    sc = list(C.SC_COLS)
    others = [c for c in range(44) if c not in sc]
    np.testing.assert_array_equal(got[..., others].astype(np.float32), want[..., others].astype(np.float32))


@needs_v84
def test_steered_pick_follows_p3_exactly(runs):
    o, op, seen = runs("x_s", CAP_LONG)
    ex, crit = seen["ex"], seen["hooks"]["steer"]
    assert crit.active and crit.theta is not None
    A = ex.arch
    v, b = crit.slopes()
    checked = 0
    for e in range(ex.exp_n, ex.exp_n + 40):
        if not np.random.default_rng([SEED, 85, e, 2000]).random() < 0.5:
            continue
        rc = np.random.default_rng([SEED, 85, e, 2000])
        rc.random()
        idx, w = A.weights(e)
        cand = idx[rc.choice(len(idx), size=32, p=w / w.sum())]
        F = expected_features(ex, cand, int(ex.exp_meta[-1][4])).reshape(-1, 44)
        u = expit(((F - crit.m) / crit.sd) @ crit.Q @ v + b)
        med = np.median(u)
        s = 1.4826 * np.median(np.abs(u - med))
        z = (u - med) / s if s > 0 else 2 * np.sign(u - med)
        p = np.exp(1.5 * np.clip(z, -2, 2))
        k = int(rc.choice(len(u), p=p / p.sum()))
        rp = np.random.default_rng([SEED, 85, e, 0, 0])
        cell = crit.pick_cell(ex, rp, np.random.default_rng([SEED, 85, e, 2000]), e)
        q = crit.pick_process(ex, rp, None, cell)
        assert (cell, q) == (int(cand[k // 4]), k % 4)
        assert rp.random() == np.random.default_rng([SEED, 85, e, 0, 0]).random()      # P_{e,0} untouched
        checked += 1
    assert checked >= 5


@needs_v84
def test_w_steering_cost_report(runs):
    """Report only (G-build test 9's 'W steering cost', never a failure): wall time of one steered pick
    (32 x 4 pairs, vectorised) and of one refit, on the long smoke run's final state. A 4M run's archive
    (20k cells) and exp rows (~10-30k) are larger."""
    import time
    o, op, seen = runs("x_s", CAP_LONG)
    ex, crit = seen["ex"], seen["hooks"]["steer"]
    t0, k = time.perf_counter(), 0
    for e in range(ex.exp_n, ex.exp_n + 200):
        rc = np.random.default_rng([SEED, 85, e, 2000])
        rp = np.random.default_rng([SEED, 85, e, 0, 0])
        crit.pick_process(ex, rp, rc, crit.pick_cell(ex, rp, rc, e))
        k += 1
    pick_ms = 1e3 * (time.perf_counter() - t0) / k
    twin = CriticS(crit.m, crit.sd, crit.Q, crit.Sigma, crit.kappa, 60)
    t0 = time.perf_counter()
    twin.refit(ex, ex.exp_n)
    refit_ms = 1e3 * (time.perf_counter() - t0)
    print(json.dumps(dict(w_steering_cost=dict(ms_per_pick=round(pick_ms, 3), ms_per_cold_refit=round(refit_ms, 1),
                                               cells=int(len(ex.arch)), rows=int(ex.exp_n)))))


@needs_v84
def test_x_s_resumes_byte_identically_after_a_kill_past_its_activation(runs, wsmoke, tmp_path):
    o, op, _ = runs("x_s", CAP_LONG)
    om, _ = npz_arrays(op)
    n0 = o["instrument"]["active_from"]
    rng = np.random.default_rng([SEED, 80])
    e_kill = int(rng.integers(n0 + 1, len(om) - 1))
    j_kill = int(rng.integers(1, int(om[e_kill, 7]) + 1))
    out, seen = tmp_path / "x_s_kill.json", {}

    def observe(d):
        env, ex, real, cnt = d["env"], d["ex"], d["env"].step, [0]

        def step(a):
            if ex.exp_n == e_kill:
                cnt[0] += 1
                if cnt[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(a)
        env.step = step
    kw = dict(out=out, v85=wsmoke, s0=wsmoke, cap=CAP_LONG, smoke=True)
    with pytest.raises(Kill):
        J.run(GAME, "A", "x_s", SEED, ckpt_s=0, observe=observe, **kw)
    assert ck.load(out.with_suffix(".ckpt"))["exp"] == seen["e"] == e_kill and not out.exists()
    o2 = J.run(GAME, "A", "x_s", SEED, **kw)
    assert stable(out) == stable(op) and o2["provenance"]["segments"][1]["resumed_at_exp"] == e_kill
    assert all(np.array_equal(a, b) for a, b in zip(npz_arrays(out), npz_arrays(op)))


@needs_v84
def test_s_arm_plays_only_its_single_best_process(runs, small):
    o, op, seen = runs("s", CAP_SHORT)
    om, _ = npz_arrays(op)
    q = int(W.load_object(small["root"], "s", GAME)[0]["q_star"])
    assert o["instrument"]["q_star"] == q and len(om) > 10 and np.all(om[:, 2] == q)
    v, vp, _ = runs("v2s", CAP_SHORT)
    vm, _ = npz_arrays(vp)
    assert len(set(vm[:, 2].astype(int))) > 1                                             # V2s's bandit varies
