"""v83 explorer: archive memory at the cap, eviction rule, replay of the trace's records, view
orientation, row labels."""
import tracemalloc

import numpy as np

from ragnarok.agent_v83.entry import Body
from ragnarok.agent_v83.explorer import Archive, Explorer, cell_key, s_bin
from ragnarok.agent_v83.logs import labels
from ragnarok.agent_v83.views import WIN, features, log_view, oriented_window, N_FEAT
from ragnarok.evaluation.senses_v83 import SenseEnv, Spent, Trace


def test_archive_20k_cells_under_600mb():
    env = SenseEnv("seaquest", seed=11)
    rng = np.random.default_rng(0)
    arch = Archive(cap=20000)
    tracemalloc.start()
    for i in range(20500):
        _, _, lives, end = env.step(int(rng.integers(env.n_actions)))
        if end:
            env.reset()
        arch.put(("k", i), env.clone(), float(i % 7), i, (1.0, 2.0), lives, env)
    rss = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert len(arch) == 20000 and arch.evicted == 500
    assert len(env._store) == 20000             # evicted handles are freed (test reads the evaluator side)
    assert rss < 600 * 2**20, rss / 2**20


def test_eviction_rule():
    env = SenseEnv("pong", seed=1)
    arch = Archive(cap=4)
    for i, S in enumerate([0.0, 1.0, 1.0, 3.0]):
        arch.put(i, env.clone(), S, 10, None, 3, env)
    arch.chosen[arch.slot[1]] = 5
    arch.chosen[arch.slot[2]] = 5
    arch.chosen[arch.slot[3]] = 9                # at the best S: protected
    arch.put(4, env.clone(), 0.0, 1, None, 3, env)
    assert 1 not in arch.slot and 2 in arch.slot and 3 in arch.slot    # most chosen below best, oldest


def test_cell_key():
    assert s_bin(0) == 0 and s_bin(1) == 4 and s_bin(3) == 8
    assert cell_key(None, 3, 0.0, (210, 160)) == ((-1, -1), 3, 0)
    assert cell_key((159.9, 209.9), 2, 1.0, (210, 160)) == ((15, 15), 2, 4)


def test_explorer_records_replay():
    tr = Trace()
    env = SenseEnv("breakout", seed=9130, cap=150000, trace=tr)
    rng = np.random.default_rng(9130)
    body = Body()
    rep = body.enter(env, rng, 9130, warm_steps=2000)
    assert rep["entry_frames"] > 0
    ex = Explorer(body)
    out = ex.run(env, rng, spent=Spent)
    assert out["billed"] >= 150000 and out["iters"] > 10
    assert len(tr.records) >= 2 and tr.records[0][1] == 0.0
    for b, t, h in tr.records:
        assert tr.replay(h) == t
    rows = ex.rows.arrays()
    assert len(rows["it"]) > 50
    g = labels(rows["rise_lag"], rows["avail"], 24)
    assert set(np.unique(g)) <= {-1, 0, 1}
    assert (rows["avail"] >= 1).all() and (rows["avail"] <= 96).all()


def _rot_move(m, k):
    dx, dy = m
    for _ in range(k):
        dx, dy = dy, -dx                          # counter-clockwise on screen (y down)
    return dx, dy


def test_view_orientation_equivariant():
    rng = np.random.default_rng(3)
    W = rng.random((4, WIN, WIN)) < 0.05
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    for m in moves:
        ref = oriented_window(np.packbits(W), m)[0]
        for k in range(4):
            got = oriented_window(np.packbits(np.rot90(W, k, axes=(1, 2)).copy()), _rot_move(m, k))[0]
            assert np.array_equal(got, ref), (m, k)
    one = np.zeros((4, WIN, WIN), bool)
    one[0, 32, 50] = True                         # something to my right
    o = oriented_window(np.packbits(one), (1, 0))[0]
    ys, xs = np.nonzero(o[0])
    assert ys[0] < 32 and abs(xs[0] - 32) <= 1    # moving right: it is ahead (up)


def test_features_shape_and_lost():
    obs = np.zeros((210, 160, 3), np.uint8)
    bg = obs.copy()
    obs[100:104, 80:84] = 200
    obs[60:62, 80:82] = 100
    packed, screen, lost, _ = log_view(obs, None, bg, (81.5, 101.5), 16.0)
    assert not lost
    f = features(packed, screen, lost, (81.5, 101.5), 3, (0, -1), False, False)
    assert f.shape == (N_FEAT,) and 400 <= N_FEAT <= 460
    _, _, lost, _ = log_view(obs, None, bg, None, 16.0)
    assert lost
