"""senses_v85 (docs/ARC4_V85_PROTOCOL_EN.md A.5, A.7, §13.3): shared start, sandbox, relative clock,
replay from s1, checkpoints. Seed 9800 = smoke runs and tests (§12)."""
import pickle
import zlib

import numpy as np
import pytest

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.evaluation.senses_v85 import LOAD_FRAMES, SenseEnv85, Spent, Trace85, make_blob

SEED = 9800


def make_start(game, n=300):
    """A mid-game start blob and its cumulative sensed rise."""
    e = AtariEnv(game, seed=SEED)
    e.step(0)
    rng = np.random.default_rng([SEED, 1])
    cum = 0.0
    for _ in range(n):
        before = e._true_score
        if e.step(int(rng.integers(e.n_actions)))[1]:
            break
        cum += max(0.0, e._true_score - before)
    return make_blob(e), cum


def ale_bytes(env):
    return pickle.dumps(env._e.ale.cloneState())


def drive(env, n, seed, hs, sandbox=False):
    """Random live play with restores and clones; optional sandbox passes on their own stream."""
    rng, rs = np.random.default_rng([SEED, seed]), np.random.default_rng([SEED, seed, 1])
    log = []
    for t in range(n):
        if t % 37 == 0:
            o, lv = env.restore(hs[int(rng.integers(len(hs)))])
            log.append((zlib.crc32(o.tobytes()), lv, env.billed))
        obs, rise, lives, end = env.step(int(rng.integers(env.n_actions)))
        log.append((zlib.crc32(obs.tobytes()), rise, lives, end, env.billed))
        if end:
            env.restore(hs[0])
        if t % 23 == 0:
            hs.append(env.clone())
        if sandbox and t % 31 == 0:
            before = (ale_bytes(env), env.billed, env._cum, env._depth, env._anchor, bytes(env._since))
            with env.sandbox():
                for _ in range(int(rs.integers(5, 60))):
                    env.step(int(rs.integers(env.n_actions)))
                if rs.random() < 0.5:
                    env.restore(hs[int(rs.integers(len(hs)))])
                    for _ in range(int(rs.integers(5, 60))):
                        env.step(int(rs.integers(env.n_actions)))
                for bad in (env.clone, env.reset, lambda: env.free(hs[0])):
                    with pytest.raises(RuntimeError):
                        bad()
            after = (ale_bytes(env), env.billed, env._cum, env._depth, env._anchor, bytes(env._since))
            assert after == before
    return log


def started(game, blob, S, cap=4_000_000, stop_at=None):
    tr = Trace85(stop_at)
    env = SenseEnv85(game, seed=SEED, trace=tr)
    env.start_from(blob, S, cap=cap)
    return env, tr


def test_start_from_billing_clock_and_cap():
    blob, S = make_start("breakout")
    tr = Trace85()
    env = SenseEnv85("breakout", seed=SEED, trace=tr)
    b0 = env.billed
    obs, lives = env.start_from(blob, S, cap=1000)
    b = pickle.loads(blob)
    assert env.billed == b0 + 4 + LOAD_FRAMES and env.clock_start == env.billed and env.billed_since_start == 0
    assert env._cum == S and env._depth == 0 and lives == b["lives"]
    assert zlib.crc32(obs.tobytes()) == zlib.crc32(zlib.decompress(b["obs"]))
    assert tr.bases[tr.cur] == b["state"] and tr.start_S == S
    h = env.clone()
    assert tr.records == [(0, S, h, b["true"], 0, 0)]
    with pytest.raises(Spent):
        for _ in range(1000):
            env.step(1)
    assert 1000 <= env.billed_since_start < 1004
    with env.sandbox():
        env.step(0)                       # the cap is not checked in the sandbox
    assert env.billed_since_start < 1004 and env.oracle_frames == 4


def test_evaluator_stop_on_cum():
    blob, S = make_start("breakout")
    env, tr = started("breakout", blob, S, stop_at=S)
    env.clone()
    assert tr.stopped
    with pytest.raises(Spent):
        env.step(0)


@pytest.mark.parametrize("game", ["breakout", "qbert"])
def test_sandbox_leaves_no_trace(game):
    blob, S = make_start(game)
    out = []
    for sb in (True, False):
        env, tr = started(game, blob, S)
        log = drive(env, 700, 2, [env.clone()], sandbox=sb)
        out.append((log, [r[:5] for r in tr.records], tr.rise_billed, tr.rise_sizes, env.billed, env.oracle_frames))
    (la, ra, rba, rsa, ba, oa), (lb, rb, rbb, rsb, bb, ob) = out
    assert la == lb and ra == rb and rba == rbb and rsa == rsb and ba == bb
    assert oa > 0 and ob == 0


@pytest.mark.parametrize("game", ["breakout", "qbert"])
def test_replay85_exact(game):
    blob, S = make_start(game)
    env, tr = started(game, blob, S)
    drive(env, 900, 3, [env.clone()], sandbox=True)
    assert len(tr.records) >= 1
    oracle = [r[5] for r in tr.records]
    assert oracle == sorted(oracle)
    for billed, cum, h, true, depth, _ in tr.records:
        assert tr.replay85(h) == (cum, true, depth)


def test_checkpoint_resume_identical():
    blob, S = make_start("qbert")
    env, tr = started("qbert", blob, S)
    hs = [env.clone()]
    drive(env, 300, 4, hs, sandbox=True)
    hs.append(env.clone())                                     # an expansion boundary
    ck = pickle.dumps((env.to_state(), tr.to_state(), list(hs)))
    live = drive(env, 400, 5, hs, sandbox=True)
    st, st_tr, hs2 = pickle.loads(ck)
    tr2 = Trace85()
    env2 = SenseEnv85("qbert", seed=SEED, trace=tr2)
    env2.from_state(st)
    tr2.from_state(st_tr)
    resumed = drive(env2, 400, 5, hs2, sandbox=True)
    assert live == resumed and tr.records == tr2.records and env.billed == env2.billed
    assert env.oracle_frames == env2.oracle_frames
    for r in tr2.records:
        assert tr2.replay85(r[2]) == (r[1], r[3], r[4])
