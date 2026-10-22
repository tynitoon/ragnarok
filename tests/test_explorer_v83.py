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
    g = labels(rows["rise_lag"], rows["avail"], rows["t"], 24)
    assert set(np.unique(g)) <= {-1, 0, 1}
    assert (g[rows["t"].astype(int) + 24 > 100] == -1).all()
    assert (rows["t"] % 4 == 0).all() and (rows["t"] >= 4).all() and (rows["avail"] <= 96).all()
    last = rows["it"] == rows["it"].max()
    assert (rows["avail"][~last] >= 1).all()          # avail 0 only on the Spent-cut rollout
    z = rows["avail"] == 0
    assert (rows["rise_lag"][z] == 255).all() and (rows["drop_lag"][z] == 255).all()


def test_labels_censoring_not_outcome_dependent():
    t, avail = np.array([60, 80, 80, 20, 20]), np.array([40, 20, 20, 10, 10])
    lag = np.array([255, 3, 255, 5, 255])
    assert labels(lag, avail, t, 24).tolist() == [0, -1, -1, -1, -1]
    assert labels(lag, avail, t, 40, stops_on_event=True).tolist() == [0, -1, -1, 1, -1]


class _StubSteer:
    """Draws exactly like V; records the hooks."""

    def __init__(self):
        self.puts, self.its = set(), []

    def pick_cell(self, ex, rng):
        i = ex.arch.draw(rng)
        ex.arch.chosen[i] += 1
        return i

    def pick_button(self, ex, rng, n, ctx):
        return int(rng.integers(n))

    def on_put(self, ex, i, ctx):
        assert ex.arch.used[i] and set(ctx) >= {"obs", "prev_obs", "me", "prev_me", "lives", "lost"}
        self.puts.add(int(i))

    def on_iter(self, ex, it):
        self.its.append(it)


def _short(steer):
    env = SenseEnv("pong", seed=9130, cap=60000)
    rng = np.random.default_rng(9130)
    body = Body()
    body.enter(env, rng, 9130, warm_steps=500, seed_steps=300, check_steps=100)
    ex = Explorer(body, steer=steer)
    out = ex.run(env, rng, spent=Spent)
    return ex, out


def test_stub_steer_equals_V_and_hooks_fire():
    exv, outv = _short(None)
    stub = _StubSteer()
    exs, outs = _short(stub)
    rv, rs = exv.rows.arrays(), exs.rows.arrays()
    assert all(np.array_equal(rv[k], rs[k], equal_nan=rv[k].dtype.kind == "f") for k in rv)
    iv, is_ = exv.iters.arrays(), exs.iters.arrays()
    assert all(np.array_equal(iv[k], is_[k]) for k in iv)
    assert outv["cells"] == outs["cells"]
    assert stub.its == list(range(1, outs["iters"] + 1))
    assert set(np.nonzero(exs.arch.used)[0].tolist()) <= stub.puts


def test_me_mask_keeps_neighbour_and_trail():
    from ragnarok.agent_v83.views import unpack
    bg = np.zeros((210, 160, 3), np.uint8)
    obs = bg.copy()
    obs[100:104, 80:96] = 200                      # me: a 16x4 paddle
    obs[100:104, 98:100] = 100                     # an object 2 px away, not touching
    packed, _, lost, _ = log_view(obs, None, bg, (87.5, 101.5), 64.0)
    w = unpack(packed)
    assert not lost and w[0].sum() == 8            # only the neighbour remains in 'objects'
    prev = bg.copy()
    prev[100:104, 76:92] = 200                     # where I was one step ago
    from ragnarok.agent_v83.views import fg_mask
    packed, _, _, _ = log_view(obs, fg_mask(prev, bg), bg, (87.5, 101.5), 64.0, prev_me=(83.5, 101.5))
    assert unpack(packed)[2].sum() == 0            # my trail is not 'vanished'


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
