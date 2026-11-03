"""oracle_v85 (A.7), part 1: prefix identities of test 6 (O_1 through the sandbox and K*map with an empty
map are byte-identical to V2s apart from the oracle-frame column), O_4 and K*map on a real map."""
import pickle

import numpy as np
import pytest

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v83.entry import Body
from ragnarok.agent_v85.explorer2s import Explorer2s
from ragnarok.evaluation.oracle_v85 import KMap, OChooser, lineage_map
from ragnarok.evaluation.senses_v85 import SenseEnv85, Spent, Trace85, make_blob

SEED, GAME, CAP = 9800, "breakout", 40_000


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


def run(setup, **hooks):
    s1, bb = setup
    tr = Trace85()
    env = SenseEnv85(GAME, seed=SEED, trace=tr)
    obs, lives = env.start_from(s1["blob"], s1["S"], cap=CAP)
    ex = Explorer2s(pickle.loads(bb), budget=CAP, seed=SEED, **hooks)
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


def test_o4_and_kmap_run(setup):
    o4 = OChooser(4)
    _, env, ex = run(setup, chooser=o4)
    assert len(o4.log) >= ex.exp_n and env.oracle_frames > 0
    assert not all(t for *_, t in o4.log)
    km = KMap(lineage_map(GAME, SEED, setup[0]))
    assert len(km.sb) > 0
    _, _, ex = run(setup, steer=km)
    assert km.coins > 0 and km.coverage is not None
