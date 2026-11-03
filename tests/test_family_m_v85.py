"""Family M, "how to move" (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8, §9, §12; Part B items 12-13; G-build test 6 for
F_M). Seed 9800 only for runs (§12); small caps and 2-3 small games, so nothing here is a measurement.

Synthetic: the frozen miner's rules (lift, c_adv >= 3 in 2 sources, RLE edit distance >= 3, at most 4 macros
dominated by one source, top 32, the one-source rule), against a naive re-implementation of the §4.4 text; G1b
window extraction (candidates, non-ADV windows, one baseline window per non-ADV rollout); DM exactly as listed;
online re-mining every 250 expansions with the n0 = 10 merge; the X_frozen switch never mines; Q5 state round trip.
Real data (v84 G1b logs, data only): the builders on 3 small games (breakout, pong, qbert), the dose draws, object
files and SHA-256. Jobs (breakout smoke start A): every family-M arm writes a valid run JSON; X_frozen equals X_on
before 250 expansions; F_M is byte-identical to V2s for its first 250 expansions (G-build test 6); a kill
mid-rollout after online re-mines (period shortened by monkeypatch) resumes to the same JSON.
"""
import dataclasses
import itertools
import json
import pickle
import types

import numpy as np
import pytest

from ragnarok.agent_v83.adapter import SLOTS
from ragnarok.agent_v85 import macros as M
from ragnarok.agent_v85.q5_v85 import Q5
from ragnarok.evaluation import arms_v85
from ragnarok.evaluation import ckpt_v85 as ck
from ragnarok.evaluation import objects_m_v85 as OM
from ragnarok.evaluation.senses_v85 import Spent
from scripts import v85_job as J
from tests.test_job_v85 import check_schema, stable

SEED, GAME = 9800, "breakout"
SMALL = ("breakout", "pong", "qbert")
CAP = 20_000
ARMS = ("x_on", "x_frozen", "f_m", "k_mine", "dm_on", "p_on", "x4_on", "x12_on")
HAVE = all(OM.run_files(g, s)[0].exists() for g in OM.BURNED for s in OM.REF)
needs_g1b = pytest.mark.skipif(not HAVE, reason="v84 G1b logs absent")


class Kill(Exception):
    pass


def S(name):
    return SLOTS.index(name)


def key(rows):
    return int(M.encode([rows])[0])


def src(adv, base, n_adv=None, n_base=None):
    ak, bk = sorted(adv), sorted(base)
    return dict(adv_k=np.array(ak, np.int64), adv_c=np.array([adv[k] for k in ak], np.int64),
                n_adv=sum(adv.values()) if n_adv is None else n_adv, base_k=np.array(bk, np.int64),
                base_c=np.array([base[k] for k in bk], np.int64), n_base=sum(base.values()) if n_base is None else n_base)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def naive_mine(sources):
    """The §4.4 keep rule written directly from the text, on dicts (independent of macros.mine)."""
    G = len(sources)
    cnt = [dict(zip(s["adv_k"].tolist(), s["adv_c"].tolist())) for s in sources]
    bas = [dict(zip(s["base_k"].tolist(), s["base_c"].tolist())) for s in sources]
    keys = set().union(*cnt)
    elig = [k for k in keys if sum(c.get(k, 0) >= 3 for c in cnt) >= min(2, G)]

    def lift(k):
        return sum(((c.get(k, 0) + 1) / (s["n_adv"] + 1)) / ((b.get(k, 0) + 1) / (s["n_base"] + 1))
                   for c, b, s in zip(cnt, bas, sources)) / G
    tot = {k: sum(c.get(k, 0) for c in cnt) for k in elig}
    ranked = sorted(elig, key=lambda k: (-lift(k), -tot[k], k))
    kept, per = [], {}
    for k in ranked:
        rows = M.decode([k])[0].tolist()
        if any(lev(rows, M.decode([o])[0].tolist()) < 3 for o in kept):
            continue
        dom = [g for g, c in enumerate(cnt) if c.get(k, 0) > tot[k] / 2]
        if G >= 2 and dom and per.get(dom[0], 0) >= 4:
            continue
        if dom:
            per[dom[0]] = per.get(dom[0], 0) + 1
        kept.append(k)
        if len(kept) == 32:
            break
    return kept, {k: lift(k) for k in kept}


def far_rows(n, seed):
    """n random 8-row sequences pairwise at row edit distance >= 3."""
    rng, out = np.random.default_rng([SEED, seed]), []
    while len(out) < n:
        r = rng.integers(0, M.BASE, M.ROWS).tolist()
        if all(lev(r, o) >= 3 for o in out):
            out.append(r)
    return out


# ---------------------------------------------------------------- the miner (synthetic)
def test_lift_is_the_equal_game_mean_of_the_per_game_ratio():
    m1, m3 = key([6] * 8), key([12] * 8)
    s0 = src({m1: 5, m3: 3}, {m1: 1}, n_adv=20, n_base=10)
    s1 = src({m1: 3, m3: 4}, {}, n_adv=10, n_base=5)
    kept, rows, summ = M.mine([s0, s1])
    want = {m1: ((6 / 21) / (2 / 11) + (4 / 11) / (1 / 6)) / 2, m3: ((4 / 21) / (1 / 11) + (5 / 11) / (1 / 6)) / 2}
    assert sorted(kept, key=lambda k: -want[k]) == kept and set(kept) == {m1, m3}
    for r in rows:
        assert r["lift"] == pytest.approx(want[r["key"]], rel=1e-12)
    assert summ["sources"] == 2 and summ["need_sources"] == 2 and summ["cap_applied"]
    # a source where the macro never occurs still enters the mean
    s2 = src({key([2] * 8): 9}, {}, n_adv=9, n_base=40)
    _, rows3, _ = M.mine([s0, s1, s2])
    r1 = next(r for r in rows3 if r["key"] == m1)
    assert r1["lift"] == pytest.approx((2 * want[m1] + (1 / 10) / (1 / 41)) / 3, rel=1e-12)


def test_keep_needs_c_adv_3_in_2_sources():
    a, b, c, d = (key(r) for r in far_rows(4, 1))
    s0 = src({a: 50, b: 3, c: 3, d: 2}, {}, n_adv=100, n_base=100)
    s1 = src({a: 2, b: 3, c: 9, d: 3}, {}, n_adv=100, n_base=100)
    kept, _, summ = M.mine([s0, s1])
    assert set(kept) == {b, c} and summ["eligible"] == 2          # a: 50 + 2 (one source); d: 2 + 3


def test_rle_edit_distance_and_greedy_order():
    hold, var1, var3 = key([6] * 8), key([6] * 7 + [0]), key([6] * 5 + [0] * 3)
    assert M.edit_distance(M.macro_of(hold), M.macro_of(var1)) == 1
    assert M.edit_distance(M.macro_of(hold), M.macro_of(var3)) == 3
    assert M.edit_distance(((6, 32),), ((12, 32),)) == 8                    # rows, not (slot, steps) tokens
    assert M.edit_distance(((6, 8), (0, 24)), ((0, 24), (6, 8))) == 4
    s0 = src({hold: 9, var1: 8, var3: 7}, {}, n_adv=50, n_base=50)
    s1 = src({hold: 9, var1: 8, var3: 7}, {}, n_adv=50, n_base=50)
    kept, _, _ = M.mine([s0, s1])
    assert kept == [hold, var3]                                              # var1 is within distance 1 of hold
    s0 = src({hold: 5, var1: 8, var3: 7}, {}, n_adv=50, n_base=50)
    kept, _, _ = M.mine([s0, s1])
    assert kept[0] == var1 and hold not in kept                             # the higher lift wins the pair


def test_at_most_4_macros_dominated_by_one_source_and_top_32():
    rows = far_rows(48, 2)
    dom = [key(r) for r in rows[:6]]
    even = [key(r) for r in rows[6:8]]
    s0 = src({**{k: 20 - i for i, k in enumerate(dom)}, **{k: 3 for k in even}}, {}, n_adv=200, n_base=200)
    s1 = src({**{k: 3 for k in dom}, **{k: 3 for k in even}}, {}, n_adv=200, n_base=200)
    s2 = src({k: 3 for k in even}, {}, n_adv=200, n_base=200)
    kept, kr, summ = M.mine([s0, s1, s2])
    assert kept[:4] == dom[:4] and set(kept) == set(dom[:4]) | set(even)
    assert summ["dominated_per_source"] == {"0": 4} and [r["dominant"] for r in kr[:4]] == [0] * 4
    many = [key(r) for r in rows[8:48]]                                        # 40 eligible, none dominated
    t0 = src({k: 3 + i for i, k in enumerate(many)}, {}, n_adv=500, n_base=500)
    t1 = src({k: 3 + i for i, k in enumerate(many)}, {}, n_adv=500, n_base=500)
    kept, _, _ = M.mine([t0, t1])
    assert kept == many[::-1][:32]
    for sources in ([s0, s1, s2], [t0, t1], [s0], [t1]):
        k1, k2 = M.mine(sources)[0], naive_mine(sources)[0]
        assert k1 == k2


def test_one_source_rule():
    rows = far_rows(7, 3)
    ks = [key(r) for r in rows]
    s0 = src({**{k: 3 + i for i, k in enumerate(ks[:6])}, ks[6]: 2}, {}, n_adv=60, n_base=60)
    kept, rr, summ = M.mine([s0])
    assert set(kept) == set(ks[:6]) and summ["need_sources"] == 1 and not summ["cap_applied"]
    assert all(r["dominant"] == 0 for r in rr)                               # dominated, but no cap with G = 1


# ---------------------------------------------------------------- G1b windows (synthetic rows)
def test_run_windows_candidates_and_baseline():
    rng = np.random.default_rng([SEED, 4])
    layout = [(0, 12), (1, 10), (2, 5), (3, 9), (4, 8)]                       # (it, rows); it 4 = cut rollout
    adv = np.array([1, 0, 0, 1])
    it = np.concatenate([[i] * n for i, n in layout])
    t = np.concatenate([4 * np.arange(1, n + 1) for _, n in layout])
    slot = rng.integers(0, len(SLOTS), len(it))                              # priv slots included
    rl = np.full(len(it), 255)
    first = {i: int(np.flatnonzero(it == i)[0]) for i, _ in layout}
    for i, k, v in [(0, 3, 2), (0, 9, 0), (0, 10, 4), (0, 11, 3), (3, 8, 1), (4, 7, 0), (1, 9, 0)]:
        rl[first[i] + k] = v
    u = np.array([0.7, 0.2])                                                   # rollouts 1 and 2 (adv == 0)
    cand, non, base = OM.run_windows(it, t, slot, rl, adv, u)
    cs = M.canon(slot)

    def k(i, a):
        return key(cs[first[i] + a:first[i] + a + 8])
    assert sorted(cand.tolist()) == sorted([k(0, 2), k(0, 4), k(3, 1)])
    assert sorted(non.tolist()) == sorted([k(1, 0), k(1, 1), k(1, 2)])
    assert base.tolist() == [k(1, 2)]                                          # floor(0.7 * 3) = 2
    assert all(int(s) < M.BASE for s in M.decode(cand).ravel())               # priv -> NOOP


@needs_g1b
def test_baseline_uniforms_are_one_9803_4_stream_in_canonical_run_order():
    U = OM.baseline_uniforms()
    runs = [(g, s) for g in OM.BURNED for s in OM.REF]
    assert list(U) == runs and len(runs) == 82 and len(OM.CANON_UNITS) == 74
    full = np.random.default_rng([9803, 4]).random(sum(len(v) for v in U.values()))
    assert np.concatenate([U[r] for r in runs]).tolist() == full.tolist()
    for g, s in (("pong", 9712), ("qbert", 9713)):
        assert len(U[(g, s)]) == int(np.count_nonzero(OM.run_adv(g, s) == 0))


# ---------------------------------------------------------------- DM
def test_dm_is_exactly_the_listed_33_macros():
    dirs = ["0,1", "0,-1", "1,0", "1,1", "1,-1", "-1,0", "-1,1", "-1,-1"]
    want = [((S(d + ",0"), 8),) for d in dirs] + [((S(d + ",0"), 32),) for d in dirs]
    want += [((S(d + ",1"), 16),) for d in dirs]
    want += [((S("0,0,1"), 4), (S("0,0,0"), 4)) * 4]
    want += [((S("-1,0,0"), 8), (S("1,0,0"), 8)), ((S("0,-1,0"), 8), (S("0,1,0"), 8))]
    want += [((S("1,-1,0"), 8), (S("1,1,0"), 8)), ((S("-1,-1,0"), 8), (S("-1,1,0"), 8))]
    want += [((S("0,0,0"), 16),), ((S("0,0,1"), 32),)]
    want += [((S("1,0,0"), 8), (S("1,0,1"), 8)), ((S("-1,0,0"), 8), (S("-1,0,1"), 8))]
    assert M.DM == want and len(M.DM) == 33 and len(set(M.DM)) == 33
    for m in M.DM:
        n = M.steps(m)
        assert 8 <= n <= 64 and n % 4 == 0 and all(not SLOTS[s].startswith("priv") for s, _ in m)
    o = OM.build_dm()
    assert o["kind"] == "DM" and [tuple(map(tuple, m)) for m in o["macros"]] == want


def test_dm_report_records_the_frozen_count_conflict():
    """Review finding M-3: the frozen §4.4 DM list counts 33 against its stated 32; build note 6 keeps all 33."""
    rep = OM.build_dm()["report"]
    assert rep["size"] == 33 and rep["note"] == OM.DM_NOTE
    assert all(s in rep["note"] for s in ("33", "32", "build note 6"))


# ---------------------------------------------------------------- online mining (synthetic explorer)
class FakeEnv:
    def __init__(self):
        self.rises = b""

    def take(self):
        r, self.rises = self.rises, b""
        return r


def fake_ex(seed=SEED):
    ad = types.SimpleNamespace(to_effect=lambda a: int(a))                    # ALE action = slot (test only)
    return types.SimpleNamespace(body=types.SimpleNamespace(adapter=ad), seed=seed)


def rollout(e):
    """40 steps (rows at t = 4..36). ADV on even e: hold right, or hold left when e % 6 == 0, rise at step 36 or 37;
    non-ADV on odd e: hold down, no rise."""
    if e % 2:
        return [S("0,1,0")] * 40, bytes(40), False
    s = S("-1,0,0") if e % 6 == 0 else S("1,0,0")
    r = bytearray(40)
    r[36 + (e % 4 == 0)] = 1
    return [s] * 40, bytes(r), True


def drive(q5, es, ex=None, env=None):
    ex, env = ex or fake_ex(), env or FakeEnv()
    for e in es:
        slots, rises, adv = rollout(e)
        env.rises = rises
        q5.after(ex, env, e, dict(steps=len(slots), acts=bytes(slots), adv=adv))
        assert env.rises == b""
    return q5


def test_online_remining_every_250_with_n0_10_merge():
    R, L, D = ((S("1,0,0"), 32),), ((S("-1,0,0"), 32),), ((S("0,0,0"), 16),)
    q = drive(M.MinedQ5([R, D]), range(249))
    assert q.mines == 0 and q.lib == [(R, 10.0), (D, 10.0)] and q.n_adv == 125 and q.n_base == 124
    drive(q, [249])
    n_left = len(range(0, 250, 6))
    assert q.mines == 1 and q.c_adv == {key([S("1,0,0")] * 8): 125 - n_left, key([S("-1,0,0")] * 8): n_left}
    assert q.lib == [(R, 10.0 + 125 - n_left), (D, 10.0), (L, float(n_left))]
    assert q.c_base == {key([S("0,1,0")] * 8): 125}
    f = drive(M.MinedQ5([]), range(249))
    assert len(f) == 0 and f.mines == 0
    drive(f, [249])
    assert f.lib == [(R, float(125 - n_left)), (L, float(n_left))] and f.mines == 1
    drive(f, range(250, 500))
    assert f.mines == 2 and f.lib == [(R, float(250 - len(range(0, 500, 6)))), (L, float(len(range(0, 500, 6))))]


def test_online_baseline_window_uses_its_own_stream():
    q, env = M.MinedQ5([]), FakeEnv()
    slots = [int(v) for v in np.random.default_rng([SEED, 5]).integers(0, M.BASE, 60)]
    env.rises = bytes(60)
    q.after(fake_ex(), env, 17, dict(steps=60, acts=bytes(slots), adv=False))
    rows = [M.canon(slots[t]) for t in range(4, 60, 4)]
    s0 = int(np.random.default_rng([SEED, 85, 17, 3000]).integers(len(rows) - 7))
    assert q.c_base == {key(rows[s0:s0 + 8]): 1} and q.n_base == 1


def test_x_frozen_never_mines():
    R = ((S("1,0,0"), 32),)
    q = drive(M.MinedQ5([R], mining=False), range(1000))
    assert q.mines == 0 and q.seen == 0 and q.c_adv == {} and q.c_base == {} and q.lib == [(R, 10.0)]
    assert q.caption()["mining"] is False and q.caption()["library_size"] == 1


def test_q5_state_round_trip_mid_run():
    R = ((S("1,0,0"), 32),)
    a = drive(M.MinedQ5([R], every=50), range(300))
    b = drive(M.MinedQ5([R], every=50), range(150))
    c = M.MinedQ5([R], every=50)
    c.from_state(pickle.loads(pickle.dumps(b.to_state())))
    drive(c, range(150, 300))
    assert c.to_state() == a.to_state() and a.mines == 6 and isinstance(c, Q5)


# ---------------------------------------------------------------- builders on small games (real G1b data)
@pytest.fixture(scope="module")
def small():
    if not HAVE:
        pytest.skip("v84 G1b logs absent")
    x = OM.build_x(GAME, sources=["pong", "qbert"])
    return dict(x=x, k=OM.build_k_mine(GAME), p=OM.build_p(GAME, x),
                x4=OM.build_dose(GAME, "X4", 1, sources=["pong", "qbert"]),
                x12=OM.build_dose(GAME, "X12s", 1, sources=["pong", "qbert"]))


@needs_g1b
def test_x_and_k_mine_follow_the_miner(small):
    x, k = small["x"], small["k"]
    st = [OM.game_stats(g) for g in ("pong", "qbert")]
    assert x["kind"] == "X" and x["sources"] == ["pong", "qbert"] and 0 < len(x["macros"]) <= 32
    want, lifts = naive_mine(st)
    assert [list(map(list, M.macro_of(q))) for q in want] == x["macros"]
    rows = x["report"]["macros"]
    assert [r["lift"] for r in rows] == pytest.approx([lifts[q] for q in want], rel=1e-12)
    for r in rows:
        assert min(r["c_adv_by_source"]) >= 3 and M.steps(r["macro"]) == 32
    for a, b in itertools.combinations(x["macros"], 2):
        assert M.edit_distance(a, b) >= 3
    kst = [OM.game_stats(GAME)]
    assert k["kind"] == "K_mine" and k["sources"] == [GAME] and not k["report"]["summary"]["cap_applied"]
    assert [list(map(list, M.macro_of(q))) for q in naive_mine(kst)[0]] == k["macros"] and len(k["macros"]) > 0
    for r in k["report"]["macros"]:
        assert r["c_adv"] >= 3
    s = OM.game_stats("pong")
    assert s["n_base"] <= int(np.count_nonzero(OM.run_adv("pong", 9712) == 0) + np.count_nonzero(
        OM.run_adv("pong", 9713) == 0)) and s["n_adv"] == int(s["adv_c"].sum())


@needs_g1b
def test_p_is_matched_on_length_and_slot_frequency(small):
    x, p = small["x"], small["p"]
    assert p["kind"] == "P" and p["sources"] == x["sources"] and len(p["macros"]) == len(x["macros"])
    st = [OM.game_stats(g) for g in x["sources"]]
    pool = set(np.concatenate([s["non_k"] for s in st]).tolist())
    keys = [r["key"] for r in p["report"]["macros"]]
    assert len(set(keys)) == len(keys) and set(keys) <= pool and p["report"]["matched_to"] == x["library_sha256"]
    for r, xm, pm in zip(p["report"]["macros"], x["macros"], p["macros"]):
        hx = np.bincount(M.unroll(xm), minlength=M.BASE)
        hp = np.bincount(M.unroll(pm), minlength=M.BASE)
        assert M.steps(pm) == 32 and int(np.abs(hx - hp).sum()) == r["distance"]
    assert p["report"]["exact"] >= len(p["macros"]) // 2
    assert OM.build_p(GAME, x)["library_sha256"] == p["library_sha256"]         # deterministic
    u = OM.p_uniforms(GAME)
    assert len(u) == 32 and u.tolist() == np.random.default_rng([9803, 1]).random((74, 32))[OM.BURNED.index(GAME)].tolist()


def test_dose_draws():
    for unit in ("alien", "zaxxon"):
        d1, d2 = OM.dose_sources(unit, "X4", 1), OM.dose_sources(unit, "X4", 2)
        for d in (d1, d2):
            assert len(d) == len(set(d)) == 4 and unit not in d and set(d) <= set(OM.BURNED)
        assert d1 != d2 and d1 == OM.dose_sources(unit, "X4", 1)
    x12 = OM.dose_sources("pacman", "X12s", 1)
    assert len(set(x12)) == 12 and "ms_pacman" not in x12 and "pacman" not in OM.sources_of("pacman")
    assert len(OM.sources_of("alien")) == 40 and len(OM.sources_of("pacman")) == 40
    assert len(OM.sources_of("frogger")) == 40 and "freeway" not in OM.sources_of("frogger")
    assert len(OM.sources_of("defender")) == 41
    with pytest.raises(AssertionError):
        OM.dose_sources("alien", "X4s", 2)                                    # one sealed draw


def test_dose_kind_refuses_seeds_without_a_dose_draw():
    """Review fix M-7 (§12): development X4_on on the G2 dose seeds, X12_on only at 9826 (the F1 closure build);
    sealed X4_on / X12_on only at the B1 2-run seeds 9853/9856; smoke draw 1 at 9800 only; the rest refused."""
    def c(game, seed, smoke=False):
        return types.SimpleNamespace(game=game, seed=seed, smoke=smoke)
    assert [OM.dose_kind(c("alien", s), "X4s") for s in (9826, 9829, 9827, 9830)] == \
        [("X4", 1), ("X4", 1), ("X4", 2), ("X4", 2)]
    assert OM.dose_kind(c("alien", 9826), "X12s") == ("X12s", 1)                # the F1 closure build only
    assert OM.dose_kind(c(GAME, SEED, True), "X4s") == ("X4", 1) and OM.dose_kind(c(GAME, SEED, True), "X12s") == ("X12s", 1)
    for sk in ("X4s", "X12s"):
        assert OM.dose_kind(c("battle_zone", 9853), sk) == OM.dose_kind(c("pacman", 9856), sk) == (sk, 1)
    for game, seed, sk in (("alien", 9812, "X12s"), ("alien", 9827, "X12s"), ("alien", 9853, "X12s"),
                           ("alien", 9853, "X4s"), ("alien", SEED, "X4s"), ("alien", 9828, "X4s"),
                           ("battle_zone", 9826, "X4s"), ("battle_zone", 9826, "X12s"), ("battle_zone", 9854, "X4s"),
                           ("battle_zone", 9873, "X12s"), ("battle_zone", SEED, "X12s")):
        with pytest.raises(ValueError):
            OM.dose_kind(c(game, seed), sk)


def test_x12_on_on_a_development_unit_is_built_for_its_closure_only(tmp_path):
    """Review fix M-7: X12_on at 9826 on a development unit builds its hooks and explorer exactly (the F1 closure
    hash) but the explorer refuses to run; on a sealed unit at a B1 seed it is an ordinary Explorer2m run."""
    od = tmp_path / "objects" / "m"
    for unit in ("alien", "battle_zone"):
        OM.write_object(od, OM._object("X12s", unit, 1, [9804, 3], ["pong"], [], {}, {}))
    ctx = types.SimpleNamespace(game="alien", seed=9826, smoke=False, paths=dict(v85=str(tmp_path)))
    _, h = arms_v85.build("x12_on", ctx)
    assert h["object"]["closure_only"] and h["object"]["kind"] == "X12s" and h["explorer"] is not M.Explorer2m
    ex = h["explorer"](None, budget=1000, seed=9826, steer=None, q5=h["q5"], chooser=None, boundary=None, **h["kwargs"])
    assert isinstance(ex, M.Explorer2m) and ex.q5 is h["q5"]
    with pytest.raises(RuntimeError, match="closure hash only"):
        ex.run(None)
    _, h = arms_v85.build("x12_on", types.SimpleNamespace(game="battle_zone", seed=9853, smoke=False,
                                                          paths=dict(v85=str(tmp_path))))
    assert not h["object"]["closure_only"] and h["explorer"] is M.Explorer2m


def test_build_sealed_writes_x41_and_its_near_sibling_variants(tmp_path, monkeypatch):
    """Review fix M-4 (§9 "X_41 and its near-sibling variants"): build_sealed writes the minus-sibling X (primary)
    and, for a unit with a near-sibling, X_41 on all 41 burned games under variants/all41/ with its manifest
    (the layout of scripts/v85_objects.py --pool sealed). The miner and P are stubbed: only the sources matter."""
    monkeypatch.setattr(OM, "_mined", lambda kind, unit, draw, seed, sources, root, order:
                        OM._object(kind, unit, draw, seed, sources, [], {}, {}))
    monkeypatch.setattr(OM, "build_p", lambda unit, x, *a, **k:
                        OM._object("P", unit, None, list(OM.SEED_P), x["sources"], [], {}, {}))
    man = OM.build_sealed(["pacman", "defender"], tmp_path)
    var = tmp_path / "variants" / "all41"
    main = {f"{k}_{u}.json" for k in ("x", "p", "x4s", "x12s") for u in ("pacman", "defender")} | {"dm.json"}
    assert set(man) == main | {"variants/all41/x_pacman.json"}
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {k: man[k] for k in main}
    assert json.loads((var / "manifest.json").read_text(encoding="utf-8")) == {"x_pacman.json": man["variants/all41/x_pacman.json"]}
    assert OM.load_object(tmp_path / "x_pacman.json")["sources"] == [g for g in OM.BURNED if g != "ms_pacman"]
    assert OM.load_object(var / "x_pacman.json")["sources"] == list(OM.BURNED)    # manifest hash checked
    assert OM.load_object(tmp_path / "x_defender.json")["sources"] == list(OM.BURNED) and not (var / "x_defender.json").exists()


@needs_g1b
def test_object_files_and_sha256(small, tmp_path):
    for name in ("x", "k", "p", "x4", "x12"):
        o = small[name]
        path, h = OM.write_object(tmp_path, o)
        assert OM.load_object(path, h)["library_sha256"] == o["library_sha256"]
        assert path.name == OM.object_name(o["kind"], o["unit"], o["draw"])
    assert small["x4"]["sources"] == ["pong", "qbert"] and small["x4"]["seed"] == [9804, 1]
    path, h = OM.write_object(tmp_path, small["x"])
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["macros"][0][0][0] = (bad["macros"][0][0][0] + 1) % M.BASE
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        OM.load_object(path)
    OM.write_object(tmp_path, small["x"])
    ck.write_json(tmp_path / "manifest.json", {path.name: "0" * 64})
    with pytest.raises(ValueError):
        OM.load_object(path)


@needs_g1b
def test_build_development_writes_every_object_and_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(OM, "sources_of", lambda unit: [g for g in SMALL if g != unit])
    man = OM.build_development([GAME], tmp_path)
    want = {"dm.json", f"x_{GAME}.json", f"kmine_{GAME}.json", f"p_{GAME}.json", f"x4_{GAME}_d1.json",
            f"x4_{GAME}_d2.json", f"x12s_{GAME}.json"}
    assert set(man) == want and json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == man
    for name, h in man.items():
        o = OM.load_object(tmp_path / name)                                      # file hash from the manifest
        assert OM._sha_file(tmp_path / name) == h and o["format"] == OM.FORMAT
    x = OM.load_object(tmp_path / f"x_{GAME}.json")
    assert x["sources"] == ["pong", "qbert"] and set(x["inputs"]) == {"pong_9712", "pong_9713", "qbert_9712",
                                                                      "qbert_9713"}


# ---------------------------------------------------------------- arms through the job (breakout smoke start A)
@pytest.fixture(scope="module")
def smoke(tmp_path_factory, small):
    d = J.build_smoke(tmp_path_factory.mktemp("v85fm"), GAME)
    od = d / "objects" / "m"
    for name in ("x", "k", "p", "x4", "x12"):
        OM.write_object(od, small[name])
    return d


def run(d, arm, out, cap=CAP, **kw):
    return J.run(GAME, "A", arm, SEED, out=out, v85=d, s0=d, cap=cap, smoke=True, **kw)


@pytest.fixture(scope="module")
def ref(smoke, tmp_path_factory):
    d, cache = tmp_path_factory.mktemp("v85fmref"), {}

    def get(arm):
        if arm not in cache:
            out = d / f"{GAME}_A_{arm}_{SEED}.json"
            run(smoke, arm, out)
            o = json.loads(out.read_text(encoding="utf-8"))
            check_schema(o)
            cache[arm] = (o, out)
        return cache[arm]
    return get


def ctx_of(sd, **kw):
    St = J.Starts(sd)
    u = St.unit(GAME)
    c = arms_v85.Ctx(game=GAME, start="A", seed=SEED, E=SEED, s1=St.load_s1(GAME, "A"), other_s1=St.load_s1(GAME, "B"),
                     body=St.load_body(GAME, "A"), ladder=u["ladder"], rungs=u["K_g"], k_first=min(u["K_g"]),
                     k_ref=max(u["K_g"]), cap=CAP, units=[GAME], paths=dict(v85=str(sd)), starts=St.data,
                     load_s1=St.load_s1, load_body=St.load_body, smoke=True)
    return dataclasses.replace(c, **kw)


@needs_g1b
def test_arm_hooks_objects_and_refusals(smoke, small):
    size = dict(x_on=len(small["x"]["macros"]), x_frozen=len(small["x"]["macros"]), f_m=0,
                k_mine=len(small["k"]["macros"]), dm_on=33, p_on=len(small["p"]["macros"]),
                x4_on=len(small["x4"]["macros"]), x12_on=len(small["x12"]["macros"]))
    for arm in ARMS:
        mod, hooks = arms_v85.build(arm, ctx_of(smoke))
        q5 = hooks["q5"]
        assert isinstance(q5, M.MinedQ5) and hooks["explorer"] is M.Explorer2m and len(q5) == size[arm], arm
        assert q5.mining == (arm != "x_frozen") and hooks["kwargs"]["after"] == q5.after
        assert arms_v85.stateful(hooks) == {} and all(w == 10.0 for _, w in q5.lib), arm
    assert arms_v85.build("x4_on", ctx_of(smoke))[1]["object"]["draw"] == 1
    with pytest.raises(ValueError):
        arms_v85.build("x4_on", ctx_of(smoke, seed=9812))                      # X4_on runs on the dose seeds
    with pytest.raises(FileNotFoundError):
        arms_v85.build("x_on", ctx_of(smoke, smoke=False, game="alien"))       # production: F1 objects only
    import scripts.v85_verdict as V
    assert {V.canon(a) for a in ARMS} <= {V.canon(n) for n in V.FAM["M"].values() if isinstance(n, str)} | \
        {V.canon(n) for n in V.FAM["M"]["tuned"] + V.FAM["M"]["placebos"]}


@needs_g1b
@pytest.mark.parametrize("arm", ARMS)
def test_family_arm_smoke_runs(ref, arm):
    o, _ = ref(arm)
    ins = o["instrument"]
    assert o["arm"] == arm and not o["interruption"] and o["aborted"] is None and o["oracle_frames"] == 0
    assert ins["mining"] == (arm != "x_frozen") and ins["n0"] == 10 and ins["every"] == 250
    assert ins["bandit_k"] == (4 if arm == "f_m" else 5) and (ins["q5_expansions"] > 0) == (arm != "f_m")
    assert ins["mines"] == 0 and o["summary"]["expansions"] < 250 and ins["object"]["size"] == ins["library_size"]
    assert ins["rollouts_logged"] == (0 if arm == "x_frozen" else o["summary"]["expansions"])
    files = [f for f, _ in o["provenance"]["closure_files"]]
    assert "ragnarok/agent_v85/macros.py" in files and f"ragnarok/evaluation/arms_v85/{arm}.py" in files


@needs_g1b
def test_x_frozen_equals_x_on_before_250_expansions(ref):
    a, b = ref("x_on")[0], ref("x_frozen")[0]
    keys = [k for k in a if k not in set(J.VOLATILE) | {"arm", "instrument"}]
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}


def stop_at(n, seen):
    """An observe hook that ends the run (Spent, as the cap does) at the boundary of expansion n."""
    def observe(d):
        seen.update(d)
        ex, cp = d["ex"], d["ex"].boundary

        def boundary(x, e):
            if e == n:
                raise Spent(f"test stop at expansion {n}")
            cp(x, e)
        ex.boundary = boundary
    return observe


@needs_g1b
def test_f_m_identical_to_v2s_for_250_expansions(smoke, tmp_path):
    """G-build test 6: F_M is byte-identical to V2s for 250 expansions, through the job (run JSON apart from arm and
    captions, exp_meta and exp_feat bytes); its first re-mine happens after the 250th expansion."""
    outs, seen = {}, {}
    for arm in ("v2s", "f_m"):
        outs[arm] = tmp_path / f"{GAME}_A_{arm}_{SEED}.json"
        seen[arm] = {}
        run(smoke, arm, outs[arm], cap=2_000_000, observe=stop_at(250, seen[arm]))
    v, f = (json.loads(outs[a].read_text(encoding="utf-8")) for a in ("v2s", "f_m"))
    assert v["summary"]["expansions"] == f["summary"]["expansions"] == 250 and not v["stopped"]
    keys = [k for k in v if k not in set(J.VOLATILE) | {"arm", "instrument"}]
    assert {k: v[k] for k in keys} == {k: f[k] for k in keys}
    za, zb = (np.load(outs[a].with_suffix(".npz")) for a in ("v2s", "f_m"))
    assert za["exp_meta"].tobytes() == zb["exp_meta"].tobytes() and za["exp_feat"].tobytes() == zb["exp_feat"].tobytes()
    ins = f["instrument"]
    assert ins["mines"] == 1 and ins["rollouts_logged"] == 250 and ins["bandit_k"] == 4
    q5 = seen["f_m"]["hooks"]["q5"]
    assert q5.n_adv == sum(q5.c_adv.values()) and len(q5) == len(q5.kept)


@needs_g1b
def test_x_on_resumes_after_a_kill_mid_rollout_with_online_remines(smoke, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "EVERY", 5)
    a = tmp_path / "ref" / f"{GAME}_A_x_on_{SEED}.json"
    a.parent.mkdir()
    oa = run(smoke, "x_on", a)
    assert oa["instrument"]["mines"] >= 3 and oa["instrument"]["every"] == 5
    meta = np.load(a.with_suffix(".npz"))["exp_meta"]
    cands = [int(r[0]) for r in meta if r[0] >= 6 and r[7] >= 2]
    rng = np.random.default_rng([SEED, 80])
    e_kill = cands[int(rng.integers(len(cands)))]
    j_kill = int(rng.integers(1, int(meta[e_kill, 7]) + 1))
    out, seen = tmp_path / f"{GAME}_A_x_on_{SEED}.json", {}

    def observe(d):
        env, ex, real, n = d["env"], d["ex"], d["env"].step, [0]

        def step(x):
            if ex.exp_n == e_kill:
                n[0] += 1
                if n[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(x)
        env.step = step
    with pytest.raises(Kill):
        run(smoke, "x_on", out, ckpt_s=0, observe=observe)
    assert ck.load(out.with_suffix(".ckpt"))["exp"] == seen["e"] == e_kill and not out.exists()
    o2 = run(smoke, "x_on", out)
    assert stable(out) == stable(a) and o2["provenance"]["segments"][1]["resumed_at_exp"] == e_kill
    za, zb = (np.load(p.with_suffix(".npz")) for p in (a, out))
    assert za["exp_meta"].tobytes() == zb["exp_meta"].tobytes()
