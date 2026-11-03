"""explorer2s (A.6): determinism (test 4), resume identity at an expansion boundary (test 5), side-effect-free
propose, and a dry rollout in the sandbox that leaves explorer, archive, bandit, env and trace unchanged."""
import pickle

import numpy as np
import pytest

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v83.entry import Body
from ragnarok.agent_v85.explorer2s import Explorer2s
from ragnarok.evaluation.senses_v85 import SenseEnv85, Spent, Trace85, make_blob

SEED, GAME, CAP = 9800, "breakout", 50_000


class Kill(Exception):
    pass


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
    blob = make_blob(e)
    body = Body()
    body.enter(SenseEnv85(GAME, seed=SEED), np.random.default_rng([SEED, 0]), SEED)
    return blob, S, pickle.dumps(body)


def fresh(setup, boundary=None):
    blob, S, bb = setup
    tr = Trace85()
    env = SenseEnv85(GAME, seed=SEED, trace=tr)
    obs, lives = env.start_from(blob, S, cap=CAP)
    ex = Explorer2s(pickle.loads(bb), budget=CAP, seed=SEED, boundary=boundary)
    return env, tr, ex, obs, lives, S


def outcome(tr, ex, summ):
    return pickle.dumps((tr.records, ex.exp_meta, np.array(ex.exp_rows).tobytes(), summ))


def test_determinism_and_resume(setup):
    env, tr, ex, obs, lives, S = fresh(setup)
    full = outcome(tr, ex, ex.run(env, obs, lives, S, spent=Spent))
    assert ex.exp_n > 30 and len(ex.arch) > 20 and len(tr.records) >= 1
    env, tr, ex, obs, lives, S = fresh(setup)
    assert outcome(tr, ex, ex.run(env, obs, lives, S, spent=Spent)) == full
    ck = {}

    def boundary(x, e):
        if e == 17:
            ck["s"] = pickle.dumps((x.to_state(), env.to_state(), tr.to_state()))
            raise Kill
    env, tr, ex, obs, lives, S = fresh(setup, boundary)
    with pytest.raises(Kill):
        ex.run(env, obs, lives, S, spent=Spent)
    s_ex, s_env, s_tr = pickle.loads(ck["s"])
    tr2 = Trace85()
    env2 = SenseEnv85(GAME, seed=SEED, trace=tr2)
    env2.from_state(s_env)
    tr2.from_state(s_tr)
    ex2 = Explorer2s(pickle.loads(setup[2]), budget=CAP, seed=SEED)
    ex2.from_state(s_ex)
    assert outcome(tr2, ex2, ex2.run(env2, spent=Spent, resume=True)) == full


def test_propose_and_dry_rollout_have_no_side_effects(setup):
    env, tr, ex, obs, lives, S = fresh(setup)

    def stop(x, e):
        if e == 12:
            raise Kill
    ex.boundary = stop
    with pytest.raises(Kill):
        ex.run(env, obs, lives, S, spent=Spent)
    def snap():
        es = env.to_state()
        es.pop("oracle_frames")                       # the only counter a sandbox pass may change
        return (pickle.dumps(ex.to_state()), pickle.dumps(es), pickle.dumps(tr.to_state()),
                pickle.dumps(env._e.ale.cloneState()))
    before = snap()
    props = [ex.propose(12, j) for j in range(4)]
    assert props == [ex.propose(12, j) for j in range(4)] and snap() == before
    saved_tracker = pickle.dumps(ex.body.tracker)
    for j, (i, q, slot) in enumerate(props):
        with env.sandbox():
            out = ex._rollout(env, i, q, slot, ex.stream(12, j, 1), 12, dry=True)
        ex.body.tracker = pickle.loads(saved_tracker)
        assert out["steps"] > 0
    assert snap() == before and env.oracle_frames > 0
