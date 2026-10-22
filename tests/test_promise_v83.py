"""v83 promise machinery: with w = 0 the steering is exactly V's law; the placebo destroys a planted
signal; the logged views (hence W) do not depend on the palette or on button order."""
import numpy as np

from ragnarok.agent_v83.explorer import Archive
from ragnarok.agent_v83.promise import HEADS, Promise, fit_head
from ragnarok.agent_v83.views import N_FEAT, log_view


class _Ad:
    def __init__(self, slot_of):
        self.slot_of = slot_of
        self.buttons = {}
        for b, s in slot_of.items():
            self.buttons.setdefault(s, []).append(b)

    def to_effect(self, a):
        return self.slot_of[a]


class _Body:
    def __init__(self):
        self.bg = np.zeros((210, 160, 3), np.uint8)
        self.my_size = 16.0
        self.adapter = _Ad({0: 0, 1: 2, 2: 4, 3: 2})


class _Ex:
    def __init__(self, arch):
        self.arch, self.body = arch, _Body()


def _ctx(seed=0):
    r = np.random.default_rng(seed)
    obs = np.zeros((210, 160, 3), np.uint8)
    for _ in range(6):
        y, x = r.integers(10, 200), r.integers(5, 150)
        obs[y:y + 3, x:x + 4] = r.integers(1, 255)
    return dict(obs=obs, prev_obs=None, me=(80.0, 100.0), prev_me=None, lives=3, lost=False)


def test_zero_weights_reproduce_V_laws():
    env_free = type("E", (), {"free": lambda self, h: None})()
    arch = Archive(cap=50)
    rng = np.random.default_rng(1)
    for i in range(30):
        arch.put(i, i, float(rng.integers(0, 3)), int(rng.integers(1, 50)), None, 3, env_free)
        arch.chosen[arch.slot[i]] = int(rng.integers(0, 9))
    idx, w = arch.weights()
    pv = w / w.sum()
    pr = Promise()                                   # F: w0 = 0
    for h in HEADS:                                  # even active, zero weights and any intercepts
        pr.active[h] = True
        pr.b[h] = 0.7 if h == "good" else -1.3
    pr.mean, pr.scale = np.zeros(N_FEAT), np.ones(N_FEAT)
    ex = _Ex(arch)
    for i in idx:
        pr.on_put(ex, int(i), _ctx(int(i)))
    # (b) button law: exactly uniform
    p = pr.button_probs(ex, 4, _ctx(7))
    assert np.allclose(p, 0.25, atol=1e-12)
    # (a) cell law: 16 candidates from V's law, uniform pick (all z = 0) -> V's law, checked empirically
    chosen0 = arch.chosen.copy()
    counts = np.zeros(len(arch.S))
    r = np.random.default_rng(5)
    n = 6000
    for _ in range(n):
        arch.chosen[:] = chosen0                     # keep V's law fixed for the comparison
        counts[pr.pick_cell(ex, r)] += 1
    emp = counts[idx] / n
    assert np.abs(emp - pv).max() < 0.02


def test_placebo_destroys_planted_signal():
    import scripts.train_promise_v83 as tp
    r = np.random.default_rng(0)
    n_g, per = 3, 3000
    X = r.normal(size=(n_g * per, N_FEAT)).astype(np.float16)
    it = np.tile(np.repeat(np.arange(per // 25), 25), n_g)
    game = np.repeat(np.arange(n_g), per).astype(np.int16)
    signal = X[:, 5].astype(float)
    p = 1 / (1 + np.exp(-(2.0 * signal - 1.0)))
    hit = r.random(len(p)) < p
    m = dict(games=np.array(["a", "b", "c"]), game=game, seed=np.full(len(game), 9131), it=it,
             rise_lag=np.where(hit, 3, 255).astype(np.uint8), drop_lag=np.full(len(game), 255, np.uint8),
             avail=np.full(len(game), 60, np.uint8), t=np.full(len(game), 8, np.uint8), slot=np.zeros(len(game), np.int16))
    real = tp.fit(X, m, np.arange(n_g), "good")
    perm = tp.fit(X, m, np.arange(n_g), "good", shift=tp.placebo_index(m))
    assert abs(real["w"][5]) > 1.0
    assert abs(perm["w"][5]) < 0.2 * abs(real["w"][5])


def test_view_invariant_to_palette():
    r = np.random.default_rng(2)
    bg = np.zeros((210, 160, 3), np.uint8)
    bg[:, :] = (10, 20, 30)
    obs = bg.copy()
    for _ in range(8):
        y, x = r.integers(10, 200), r.integers(5, 150)
        obs[y:y + 4, x:x + 3] = r.integers(40, 255, size=3)
    lut = r.permutation(256).astype(np.uint8)       # a palette change (per channel bijection)
    a = log_view(obs, None, bg, (80.0, 100.0), 16.0)
    b = log_view(lut[obs], None, lut[bg], (80.0, 100.0), 16.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_head_fit_pulls_to_prior_with_little_data():
    r = np.random.default_rng(3)
    X = r.normal(size=(50, 4))
    y = (r.random(50) < 0.5).astype(int)
    w0 = np.array([2.0, -1.0, 0.0, 0.5])
    w, _ = fit_head(X, y, w0, n0=50000, kappa0=1.0)
    assert np.linalg.norm(w - w0) < 0.5               # 50 rows against n0 = 50,000: the prior dominates
