"""v83 senses: rises and lives match the emulator, restores are exact and billed, the ledger never
decreases, sibling frames are billed, the cap and the evaluator stop hold, the trace replays."""
import numpy as np
import pytest
from ale_py import ALEInterface, LoggerMode, roms

from ragnarok.evaluation.senses_v83 import LOAD_FRAMES, SenseEnv, Spent, Trace

ALEInterface.setLoggerMode(LoggerMode.Error)


def _raw(game, seed):
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", 1)
    ale.loadROM(roms.get_rom_path(game))
    ale.reset_game()               # AtariEnv resets once at construction (the reset draws emulator RNG)
    return ale


@pytest.mark.parametrize("game", ["pong", "breakout", "seaquest"])
def test_rise_and_lives_match_emulator(game):
    env, ale = SenseEnv(game, seed=7), _raw(game, 7)
    acts = ale.getMinimalActionSet()
    rng = np.random.default_rng(0)
    rises = lives_changes = 0
    last_lives = ale.lives()
    for _ in range(10000):
        a = int(rng.integers(env.n_actions))
        obs, rise, lives, end = env.step(a)
        r, frames = 0.0, []
        for _ in range(4):
            r += ale.act(acts[a])
            frames.append(ale.getScreenRGB())
        assert rise == max(0.0, r)
        assert lives == ale.lives()
        assert end == ale.game_over()
        assert np.array_equal(obs, np.maximum(frames[2], frames[3]))
        rises += rise > 0
        lives_changes += lives != last_lives
        last_lives = lives
        if end:
            env.reset()
            ale.reset_game()
    assert rises > 0
    if game != "pong":
        assert lives_changes > 0


def test_restore_exact_and_billed():
    env = SenseEnv("breakout", seed=3)
    rng = np.random.default_rng(1)
    for _ in range(200):
        env.step(int(rng.integers(env.n_actions)))
    h = env.clone()
    o0, seq = env.obs().copy(), [int(rng.integers(env.n_actions)) for _ in range(80)]
    first = [env.step(a) for a in seq]
    b = env.billed
    o, _ = env.restore(h)
    assert env.billed == b + LOAD_FRAMES
    assert np.array_equal(o, o0) and np.array_equal(env.obs(), o0)
    again = [env.step(a) for a in seq]
    for x, y in zip(first, again):
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


def test_ledger_monotone_and_sibling_billed():
    env = SenseEnv("pong", seed=5)
    rng = np.random.default_rng(2)
    seen = [env.billed]
    hs = []
    for t in range(600):
        env.step(int(rng.integers(env.n_actions)))
        if t % 50 == 0:
            hs.append(env.clone())
        if t % 70 == 69:
            env.restore(hs[int(rng.integers(len(hs)))])
        seen.append(env.billed)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    sib = env.sibling(991)
    b = env.billed
    for _ in range(10):
        sib.step(0)
    assert env.billed == b + 40 and sib.billed == env.billed


def test_cap_and_evaluator_stop():
    env = SenseEnv("pong", seed=1, cap=400)
    with pytest.raises(Spent):
        for _ in range(200):
            env.step(0)
    assert env.billed == 400
    tr = Trace(stop_at=1.0)
    env = SenseEnv("breakout", seed=2, trace=tr)
    rng = np.random.default_rng(3)
    with pytest.raises(Spent):
        for _ in range(20000):
            _, rise, _, end = env.step(int(rng.integers(env.n_actions)))
            if rise > 0:
                env.clone()
            if end:
                env.reset()
    assert tr.stopped and tr.records[-1][1] >= 1.0


def test_trace_replay_matches():
    tr = Trace()
    env = SenseEnv("breakout", seed=4, trace=tr)
    rng = np.random.default_rng(4)
    env.reset()
    hs = [env.clone()]
    for it in range(60):
        env.restore(hs[int(rng.integers(len(hs)))])
        for _ in range(60):
            _, _, _, end = env.step(int(rng.integers(env.n_actions)))
            if end:
                break
        hs.append(env.clone())
    assert len(tr.records) >= 2
    for billed, true, h in tr.records:
        assert tr.replay(h) == true
    assert tr.first_reaching(tr.records[-1][1])[2] == tr.records[-1][2]


def test_replay_exact_after_root_restores_qbert():
    """Q*bert: a fresh emulator right after reset keeps a context outside the saved state; the replay
    plays one NOOP before restoring, as the live emulator always has (build notes 21)."""
    tr = Trace()
    env = SenseEnv("qbert", seed=9131, trace=tr)
    rng = np.random.default_rng(1)
    env.reset()
    root = env.clone()
    hs = []
    for _ in range(40):
        env.restore(root)
        for _ in range(int(rng.integers(1, 60))):
            _, _, _, end = env.step(int(rng.integers(env.n_actions)))
            if end:
                break
        hs.append(env.clone())
    for h in hs:
        assert tr.replay(h) == env._store[h][2]
