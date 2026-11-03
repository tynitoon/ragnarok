"""oracle_v85 (A.7): prefix identities of test 6 (O_1 through the sandbox, K*map with an empty map and Q5 with an
empty library are byte-identical to V2s apart from the oracle-frame column), O_4 and K*map on a real map,
O_4-dice re-execution equal to its look-ahead on the breakout smoke start (test 8), D to the end of a synthetic
and of the real lineage, K*mac and K*mac-exact libraries, Q5, and the placebos GF4, GF and K~mac."""
import pickle
import types

import numpy as np
import pytest

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v83.adapter import SLOTS
from ragnarok.agent_v83.entry import Body
from ragnarok.agent_v85.explorer2s import Explorer2s
from ragnarok.agent_v85.q5_v85 import NOOP, Q5, resolve
from ragnarok.evaluation import start_v85 as ST
from ragnarok.evaluation.oracle_v85 import (GF, GF4, DiceMismatch, KMap, OChooser, ODice, d_script, kmac_library,
                                            lineage_map, tilde_kmac, tilde_library, tilde_partners)
from ragnarok.evaluation.senses_v85 import SenseEnv85, Spent, Trace85, make_blob

SEED, GAME, CAP, SMALL = 9800, "breakout", 40_000, 20_000
R, L, F = SLOTS.index("1,0,0"), SLOTS.index("-1,0,0"), SLOTS.index("0,0,1")


@pytest.fixture(scope="module")
def setup():
    e = AtariEnv(GAME, seed=SEED)
    e.step(0)
    rng = np.random.default_rng([SEED, 1])
    S = 0.0
    for _ in range(60):
        before = e._true_score
        e.step(int(rng.integers(e.n_actions)))
        S += max(0.0, e._true_score - before)
    assert e.ale.lives() > 0 and not e.ale.game_over()
    s1 = dict(blob=make_blob(e), S=S, actions=rng.integers(e.n_actions, size=600).astype(np.uint8))
    body = Body()
    body.enter(SenseEnv85(GAME, seed=SEED), np.random.default_rng([SEED, 0]), SEED)
    return s1, pickle.dumps(body)


@pytest.fixture(scope="module")
def real():
    """The breakout smoke start (§12): s1 at A and B from the v84 G1b top lineages (9712, 9713)."""
    if not (ST.V84 / "g1b" / f"{GAME}_9712.npz").exists():
        pytest.skip("v84 G1b logs absent")
    lad = ST.ladder(GAME)
    return lad, {s: ST.top_lineage(GAME, ST.REF_SEEDS[s], lad, 8) for s in "AB"}


def run(setup, s1=None, cap=CAP, stop_at=None, **hooks):
    s1 = setup[0] if s1 is None else s1
    tr = Trace85(stop_at)
    env = SenseEnv85(GAME, seed=SEED, trace=tr)
    obs, lives = env.start_from(s1["blob"], s1["S"], cap=cap)
    ex = Explorer2s(pickle.loads(setup[1]), budget=cap, seed=SEED, **hooks)
    summ = ex.run(env, obs, lives, s1["S"], spent=Spent)
    key = pickle.dumps(([r[:5] for r in tr.records], ex.exp_meta, np.array(ex.exp_rows).tobytes(), summ))
    return key, env, ex


def test_prefix_identities(setup):
    base, env0, ex0 = run(setup)
    o1 = OChooser(1)
    k1, env1, _ = run(setup, chooser=o1)
    assert len(ex0.arch) > 20 and k1 == base and env1.oracle_frames > 0 and env0.oracle_frames == 0
    k2, _, _ = run(setup, steer=KMap([]))
    assert k2 == base
    k3, _, ex3 = run(setup, q5=Q5([]))
    assert k3 == base and ex3.bandit.k == 4


def test_o4_and_kmap_run(setup):
    o4 = OChooser(4)
    _, env, ex = run(setup, chooser=o4, after=o4.after)
    assert len(o4.log) >= ex.exp_n and env.oracle_frames > 0
    assert not all(t for *_, t in o4.log)
    c = o4.caption()
    assert c["n_pairs"] == ex.exp_n and 0.0 <= c["o_ties"] < 1.0
    assert c["rank_corr"] is None or -1.0 <= c["rank_corr"] <= 1.0
    km = KMap(lineage_map(GAME, SEED, setup[0]))
    assert len(km.sb) > 0
    _, _, ex = run(setup, steer=km)
    assert km.coins > 0 and km.coverage is not None
    assert km.caption()["MAP_WEAK"] == (km.coverage < 0.30)


def test_dice_reexecution_matches_lookahead(setup, real):
    """G-build test 8 on the breakout smoke start A: actions, frames and max S of every executed winner
    equal its look-ahead; a tampered outcome raises."""
    _, s1s = real
    dice = ODice(4)
    _, env, ex = run(setup, s1=s1s["A"], cap=SMALL, chooser=dice, after=dice.after)
    assert dice.checked == ex.exp_n > 5 and env.oracle_frames > 0
    assert dice.caption()["dice_checked"] == dice.checked
    d = ODice(4)
    d._b0, d._want = 0.0, (3, 1, b"\x01\x02", 68, 1.0)
    with pytest.raises(DiceMismatch):
        d.after(None, None, 3, dict(acts=b"\x01\x03", frames=68, max_S=1.0, adv=False, new_cells=0))
    d.after(None, None, 3, dict(acts=b"\x01\x02", frames=68, max_S=1.0, adv=False, new_cells=0))
    assert d.checked == 1


def _check_d(setup, s1, lad, k_ref):
    """D's run ends on the stop; returns the number of life losses along its script."""
    script = d_script(s1, lad, k_ref)
    _, env, ex = run(setup, s1=s1, stop_at=lad[k_ref], script=script)
    tr = env._trace
    rec = next(r for r in tr.records if r[1] >= lad[k_ref])
    n = len(script)
    # the script ends on the stopping step, so the first rollout completes and the next restore is Spent
    assert tr.stopped and ex.exp_n == 1 and env.billed_since_start == 60 + 4 * n
    assert rec[0] == 60 + 4 * n and rec[4] == n and rec[1] == float(s1["cum"][n - 1])
    assert tr.replay85(rec[2]) == (rec[1], rec[3], rec[4])
    env = SenseEnv85(GAME, seed=SEED)
    lives = env.start_from(s1["blob"], s1["S"])[1]
    losses = 0
    for a in script:
        nl = env.step(int(a))[2]
        losses += int(nl < lives)
        lives = nl
    return losses


def test_d_reaches_end_of_lineage(setup, real):
    """D on a synthetic lineage (random play from the synthetic s1 up to its last rise) and on the real
    breakout lineage at A up to M_5 (two life losses on the way: no stop at the second loss): the first
    rollout plays the lineage open-loop with ordinary puts and the evaluator stop ends the run."""
    s1 = setup[0]
    env = SenseEnv85(GAME, seed=SEED)
    env.start_from(s1["blob"], s1["S"])
    rng = np.random.default_rng([SEED, 3])
    acts, cums = [], []
    for _ in range(3000):
        a = int(rng.integers(env.n_actions))
        end = env.step(a)[3]
        acts.append(a)
        cums.append(env._cum)
        if end:
            break
    assert cums[-1] > s1["S"]
    syn = dict(blob=s1["blob"], S=s1["S"], actions=np.array(acts, np.uint8), cum=np.array(cums))
    _check_d(setup, syn, [cums[-1]], 0)
    lad, s1s = real
    assert _check_d(setup, s1s["A"], lad, 5) >= 2


class _Adapter:
    """Buttons 0 NOOP, 1 '0,0,1', 2 and 4 '1,0,0' (4 is an alias), 3 '-1,0,0', 5 priv0; 6 is unmapped."""

    def __init__(self):
        self.slot_of = {0: NOOP, 1: F, 2: R, 3: L, 4: R, 5: SLOTS.index("priv0")}
        self.buttons = {}
        for b, s in sorted(self.slot_of.items()):
            self.buttons.setdefault(s, []).append(b)

    def to_button(self, s, rng=None):
        return int((self.buttons.get(int(s)) or self.buttons[NOOP])[0])


def test_kmac_library_synthetic():
    acts = np.zeros(100, np.uint8)
    acts[[9, 13, 17, 21, 25, 29, 33, 37]] = [2, 2, 4, 5, 6, 3, 1, 0]
    cum = 5.0 + np.cumsum(np.isin(np.arange(100), [10, 40, 80, 81]))
    lin = dict(actions=acts, cum=cum, S=5.0)
    lib, rep = kmac_library(lin, _Adapter())
    assert lib == [(((R, 12), (NOOP, 8), (L, 4), (F, 4), (NOOP, 4)), 1), (((NOOP, 32),), 2)]
    assert (rep["rises"], rep["windows"], rep["rises_without_window"], rep["size"], rep["sampled_steps"]) == (4, 3, 1, 2, 24)
    assert rep["fidelity"] == 21 / 24 and rep["fidelity_slot"] == 22 / 24 and rep["fidelity_held"] == 78 / 96
    assert rep["share_priv_or_noop"] == 19 / 24 and rep["share_priv"] == 1 / 24 and rep["share_unmapped"] == 1 / 24
    lx, rx = kmac_library(lin, _Adapter(), hold=1, keep_priv=True)
    assert rx["size"] == 2 and rx["fidelity"] == rx["fidelity_held"] and rx["sampled_steps"] == 96
    assert all(sum(n for _, n in m) == 32 for m, _ in lx)
    assert any(s == SLOTS.index("priv0") for s, _ in lx[0][0])


def test_q5_resolution_chaining_and_law():
    avail = {NOOP, R, F}
    assert [resolve(s, avail) for s in (R, SLOTS.index("1,0,1"), SLOTS.index("0,1,0"), SLOTS.index("priv3"), F)] \
        == [R, R, NOOP, NOOP, F]
    q5 = Q5([(((R, 8), (F, 4)), 3), (((L, 4),), 1)])
    ex = types.SimpleNamespace(slots=sorted(avail))
    s0 = q5.first_slot(ex, 0, np.random.default_rng([SEED, 7, 0]))
    assert s0 == q5.first_slot(ex, 0, np.random.default_rng([SEED, 7, 0])) and q5._seq is None
    rx = np.random.default_rng([SEED, 7, 1])
    seq = [s0] + [None] * 60
    for t in range(1, 61):
        seq[t] = q5.next_slot(ex, t, seq[t - 1], rx)
    k = 0
    while k < len(seq):                                 # a concatenation of the resolved macros
        m = [R] * 8 + [F] * 4 if seq[k] == R else [NOOP] * 4
        assert seq[k:k + len(m)] == m[:len(seq) - k]
        k += len(m)
    firsts = [q5.first_slot(ex, 0, np.random.default_rng([SEED, 7, 2, j])) for j in range(2000)]
    assert abs(np.mean([s == R for s in firsts]) - (0.9 * 3 / 4 + 0.1 / 2)) < 0.03
    q6 = Q5()
    q6.from_state(pickle.loads(pickle.dumps(q5.to_state())))
    assert q6.lib == q5.lib and len(q6) == 2 and len(Q5()) == 0


def test_kmac_runs_on_real_start(setup, real):
    """K*mac at start A: library from start B's lineage through A's adapter, played by Q5 (k = 5)."""
    lad, s1s = real
    body = pickle.loads(setup[1])
    lib, rep = kmac_library(s1s["B"], body.adapter)
    assert rep["size"] >= 1 and rep["windows"] >= 1 and 0.0 <= rep["fidelity"] <= 1.0
    assert sum(w for _, w in lib) == rep["windows"]
    _, env, ex = run(setup, s1=s1s["A"], cap=SMALL, q5=Q5(lib))
    assert ex.bandit.k == 5 and any(m[2] == 4 for m in ex.exp_meta)
    lx, rx = kmac_library(s1s["A"], body.adapter, hold=1, keep_priv=True)
    assert rx["size"] >= 1 and all(sum(n for _, n in m) == 32 for m, _ in lx)


def test_placebos(setup):
    fx = types.SimpleNamespace(arch=types.SimpleNamespace(S=np.array([1.0, 3.0, 3.0, 3.0]), lvs=np.array([1, 2, 2, 2]),
                                                          depth=np.array([0, 5, 3, 3])),
                               propose=lambda e, j: (j, j % 4, j))
    assert GF4()(fx, None, 0) == (2, 2, 2)
    g4 = GF4()
    _, _, ex = run(setup, cap=SMALL, chooser=g4)
    assert len(g4.log) >= ex.exp_n > 5 and g4.caption()["expansions"] == len(g4.log)

    class Coin:
        def __init__(self, u):
            self.u = u

        def random(self):
            return self.u

        def integers(self, n):
            return n - 1
    gx = types.SimpleNamespace(arch=types.SimpleNamespace(used=np.array([True, True, True, True, False]),
                                                          S=np.array([1.0, 3.0, 3.0, 2.0, 9.0]),
                                                          depth=np.array([0, 4, 7, 1, 9])),
                               draw_cell=lambda rng, e: -7)
    assert GF().pick_cell(gx, None, Coin(0.1), 0) == 2 and GF().pick_cell(gx, None, Coin(0.9), 0) == -7
    gf = GF()
    _, _, ex = run(setup, cap=SMALL, steer=gf)
    assert 0 < gf.coins < ex.exp_n

    units = ["u0", "u1", "u2", "u3", "u4"]
    p = tilde_partners(units)
    assert p == tilde_partners(units) and sorted(p.values()) == units and all(p[u] != u for u in units)
    libs = {(p["u0"], "A"): None, (p["u0"], "B"): [(((R, 4),), 1)]}
    assert tilde_library("u0", "A", units, libs) == (p["u0"], "B", [(((R, 4),), 1)])
    assert tilde_library("u0", "B", units, {}) == (p["u0"], None, [])
    acts = np.zeros(40, np.uint8)
    acts[8::4] = 2
    lin = dict(actions=acts, cum=5.0 + (np.arange(40) >= 39), S=5.0)
    s1s = {(p["u0"], "A"): None, (p["u0"], "B"): lin}             # the partner's start A is invalid
    h, s, lib, rep = tilde_kmac("u0", "A", units, lambda u, st: s1s.get((u, st)),
                                lambda u, st: types.SimpleNamespace(adapter=_Adapter()))
    assert (h, s, lib, rep) == (p["u0"], None, [], None)
    s1s[(p["u0"], "A")] = lin
    h, s, lib, rep = tilde_kmac("u0", "B", units, lambda u, st: s1s.get((u, st)),
                                lambda u, st: types.SimpleNamespace(adapter=_Adapter()))
    assert (h, s, lib, rep["windows"]) == (p["u0"], "B", [(((R, 32),), 1)], 1)
    foreign = [(((SLOTS.index("0,1,0"), 8), (SLOTS.index("1,1,1"), 4), (SLOTS.index("priv5"), 4)), 2),
               (((R, 16),), 1)]
    q5 = Q5(foreign)
    _, _, ex = run(setup, cap=SMALL, q5=q5)
    assert ex.bandit.k == 5 and any(m[2] == 4 for m in ex.exp_meta)
    assert all(s in ex.slots for seq in q5._tables(ex)[0] for s in seq)
