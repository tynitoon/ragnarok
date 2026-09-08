"""CPU instrument tests. Real ROM tests use only the declared development game Pong."""
import numpy as np
import pytest

from ragnarok.environments.frame_budget_v70 import BudgetExceeded, FrameBudget
from ragnarok.environments.emu_pool import EmulatorConfig, EmulatorPool


@pytest.mark.parametrize("fraction,ordinary", [(0.2, 512), (0.05, 2432)])
def test_full_pair_charge_exact_boundary(fraction, ordinary):
    budget = FrameBudget(experiment_fraction=fraction)
    budget.record_play(ordinary - 1, agent_steps=0)
    assert not budget.can_experiment(8, restores=2)
    before = budget.report()
    with pytest.raises(BudgetExceeded):
        budget.record_experiment(8, restores=2)
    assert budget.report() == before
    budget.record_play(1)
    budget.record_experiment(8, restores=2, agent_steps=2)
    report = budget.report()
    assert report["charged_frames"] == ordinary + 128
    assert report["emulated_frames"] == ordinary + 8
    assert report["restore_charges"] == 120
    assert report["experiment_agent_steps"] == 2


@pytest.mark.parametrize("value", [True, -1, 0.5, float("nan"), float("inf")])
def test_bad_frame_inputs_cannot_change_budget(value):
    budget = FrameBudget()
    original = budget.report()
    for call in (lambda: budget.record_play(value), lambda: budget.record_experiment(value),
                 lambda: budget.record_experiment(0, restores=value)):
        with pytest.raises(ValueError):
            call()
    assert budget.report() == original


def test_no_free_initial_experiments_and_no_snapshot_budget_alias():
    budget = FrameBudget(max_charged_frames=640)
    assert not budget.can_experiment(8, restores=2)
    budget.record_play(512)
    detached = budget.report()
    detached["ordinary_frames"] = 1_000_000
    budget.record_experiment(8, restores=2)
    assert not budget.can_play(1)
    with pytest.raises(BudgetExceeded):
        budget.record_play(1)
    assert budget.charged_frames == 640


@pytest.mark.parametrize("fraction", [0, -0.1, .31, 1, float("nan")])
def test_allocation_cannot_exceed_draft_hard_cap(fraction):
    with pytest.raises((ValueError, OverflowError)):
        FrameBudget(experiment_fraction=fraction)


@pytest.fixture
def ale_available():
    ale = pytest.importorskip("ale_py")
    if ale.__version__ != "0.12.1":
        pytest.skip("Bootstrap and replay contract audited for ALE 0.12.1 only")


def test_pixel_only_rpc_and_live_mode(ale_available):
    with EmulatorPool([EmulatorConfig(mode="live")]) as pool:
        image = pool.observe()
        assert isinstance(image, np.ndarray) and image.shape == (210, 160, 3)
        assert image.dtype == np.uint8
        pristine = image.copy()
        image[:] = 0
        assert np.array_equal(pool.observe(), pristine)
        for forbidden in ("getRAM", "lives", "reward", "game_over", "_capture"):
            with pytest.raises(ValueError, match="Unknown"):
                pool._one(0, forbidden)
        with pytest.raises(ValueError, match="disabled"):
            pool.clone()
        with pytest.raises(ValueError, match="disabled"):
            pool.branch_pair(0)
        observation = pool.step([0])[0]
        assert isinstance(observation, np.ndarray)  # no reward, done flag or info dictionary
        assert pool.accounting()["charged_frames"] == 136  # detection60 + reset72 + repeat4


def test_pair_replays_rng_and_leaves_ordinary_trajectory_unchanged(ale_available):
    configs = [EmulatorConfig(seed=733, max_charged_frames=10_000)] * 2
    with EmulatorPool(configs) as pool:
        for anchor in range(6):
            for step in range(160):
                action = (step + anchor) % 18
                left, right = pool.step([action, action])
                assert np.array_equal(left, right)
            raw, pooled = pool.observe(raw=True), pool.observe()
            before = pool.accounting()
            left, right = pool.branch_pair(3, control=3)
            assert np.array_equal(left, right), "same snapshot/action must restore sticky RNG/history"
            assert np.array_equal(pool.observe(raw=True), raw)
            assert np.array_equal(pool.observe(), pooled), "restore must not expose ALE's stale screen cache"
            after = pool.accounting()
            assert after["restores"] == before["restores"] + 2
            assert 120 <= after["charged_frames"] - before["charged_frames"] <= 128
            left, right = pool.step([4, 4])
            assert np.array_equal(left, right), "counterfactuals must leave ordinary play unchanged"
        assert pool.accounting(1)["restores"] == 0  # workers keep independent ledgers


def test_opaque_snapshot_restores_pixels_without_refunding_budget(ale_available):
    with EmulatorPool([EmulatorConfig(seed=421)]) as pool:
        for _ in range(180):
            pool.step([3])
        raw, pooled = pool.observe(raw=True), pool.observe()
        token = pool.clone()
        assert isinstance(token, str) and len(token) == 32
        for _ in range(12):
            pool.step([4])
        before = pool.accounting()
        assert before["experiment_frames"] == 48  # clone opens a typed lab session
        restored = pool.restore(token)
        after = pool.accounting()
        assert np.array_equal(restored, pooled)
        assert np.array_equal(pool.observe(raw=True), raw)
        assert after["charged_frames"] == before["charged_frames"] + 60
        assert after["ordinary_frames"] == before["ordinary_frames"]
        pool.release(token)
        with pytest.raises(ValueError):
            pool.restore(token)


def test_terminal_repeat_counts_only_real_frames_and_charges_reset(ale_available):
    with EmulatorPool([EmulatorConfig(max_episode_frames=5)]) as pool:
        pool.step([0])
        assert pool.accounting()["charged_frames"] == 132 + 4
        pool.step([0])
        assert pool.accounting()["charged_frames"] == 132 + 5  # one frame, not four
        pool.step([0])
        counts = pool.accounting()
        assert counts["charged_frames"] == 132 + 5 + 72 + 4
        assert counts["reset_emulated_frames"] == 144
        assert counts["format_detection_emulated_frames"] == 60
        assert counts["bootstrap_emulated_frames"] == 132
        assert counts["ordinary_agent_steps"] == 3


def test_worker_errors_are_drained_and_invalid_batch_is_not_executed(ale_available):
    configs = [EmulatorConfig(max_charged_frames=136), EmulatorConfig(max_charged_frames=144)]
    with EmulatorPool(configs) as pool:
        with pytest.raises(ValueError):
            pool.step([0, 99])
        assert [pool.accounting(i)["charged_frames"] for i in (0, 1)] == [132, 132]
        pool.step([0, 0])
        with pytest.raises(BudgetExceeded):
            pool.step([0, 0])
        # The second worker completed its valid action; its reply must not poison accounting.
        assert pool.accounting(0)["charged_frames"] == 136
        assert pool.accounting(1)["charged_frames"] == 140


def test_dead_worker_fails_closed_and_cleanup_reaps_owned_children(ale_available):
    pool = EmulatorPool([EmulatorConfig()])
    children = list(pool._processes)
    children[0].terminate()
    children[0].join(timeout=3)
    with pytest.raises((BrokenPipeError, EOFError, OSError)):
        pool.observe()
    assert pool._closed
    pool.close()
    assert not any(child.is_alive() for child in children)


def test_unaudited_game_is_refused_before_use(ale_available):
    with pytest.raises(ValueError, match="bootstrap"):
        EmulatorPool([EmulatorConfig(game="breakout")])


def test_bootstrap_is_admitted_before_native_construction(ale_available, monkeypatch):
    import ale_py
    from ragnarok.environments.emu_pool import _PixelEngine
    def forbidden_construction():
        raise AssertionError("Native construction must not occur below the initial charge")
    monkeypatch.setattr(ale_py, "ALEInterface", forbidden_construction)
    with pytest.raises(BudgetExceeded, match="132"):
        _PixelEngine(EmulatorConfig(max_charged_frames=131))


def test_manual_replay_cannot_bypass_experiment_fraction(ale_available):
    with EmulatorPool([EmulatorConfig()]) as pool:
        for _ in range(87):  # bootstrap132 +348 ordinary =480
            pool.step([0])
        with pytest.raises(BudgetExceeded):
            pool.branch_pair(3)
        token = pool.clone()
        pool.step([3])
        pool.restore(token)
        pool.step([0])
        with pytest.raises(BudgetExceeded):
            pool.restore(token)  # 8 + 120 over 480 + 128 would exceed 20%
        counts = pool.accounting()
        assert counts["ordinary_frames"] == 480
        assert counts["experiment_frames"] == 8 and counts["restores"] == 1
        pool.release(token)
        pool.step([0])
        assert pool.accounting()["ordinary_frames"] == 484


class FakeConnection:
    def __init__(self, *, send_error=None, poll_error=None):
        self.send_error, self.poll_error = send_error, poll_error
        self.closed = False

    def send(self, request):
        if self.send_error:
            raise self.send_error

    def poll(self, timeout):
        if self.poll_error:
            raise self.poll_error
        return True

    def recv(self):
        return True, "stale pixels"

    def close(self):
        self.closed = True


def fake_pool(connections):
    pool = EmulatorPool.__new__(EmulatorPool)
    pool._connections, pool._processes = connections, []
    pool._timeout, pool._closed = 1, False
    return pool


@pytest.mark.parametrize("operation", ["observe", "step"])
def test_interrupted_receive_closes_pool_before_stale_reply(operation):
    pool = fake_pool([FakeConnection(poll_error=KeyboardInterrupt())])
    with pytest.raises(KeyboardInterrupt):
        pool.observe() if operation == "observe" else pool.step([0])
    assert pool._closed
    with pytest.raises(RuntimeError, match="closed"):
        pool.observe()


def test_failed_batch_send_closes_pool_before_stale_reply():
    pool = fake_pool([FakeConnection(), FakeConnection(send_error=BrokenPipeError())])
    with pytest.raises(BrokenPipeError):
        pool.step([0, 0])
    assert pool._closed


def test_spawn_failure_closes_both_new_pipe_ends(monkeypatch):
    class FailingProcess:
        def start(self):
            raise OSError("cannot spawn")

    ends = [FakeConnection(), FakeConnection()]

    class Context:
        def Pipe(self):
            return tuple(ends)

        def Process(self, **kwargs):
            return FailingProcess()

    monkeypatch.setattr("ragnarok.environments.emu_pool.mp.get_context", lambda _: Context())
    with pytest.raises(OSError, match="spawn"):
        EmulatorPool([EmulatorConfig()])
    assert all(end.closed for end in ends)


def test_last_token_closes_lab_session_and_budget_refuses_before_action(ale_available):
    with EmulatorPool([EmulatorConfig()]) as pool:
        for _ in range(30):
            pool.step([3])
        first, second = pool.clone(), pool.clone()
        pool.release(first)
        pool.step([4])
        assert pool.accounting()["experiment_frames"] == 4
        pool.release(second)
        ordinary = pool.accounting()["ordinary_frames"]
        pool.step([0])
        assert pool.accounting()["ordinary_frames"] == ordinary + 4
    with EmulatorPool([EmulatorConfig(max_charged_frames=132)]) as pool:
        pool.clone()
        before_pixels, before_counts = pool.observe(raw=True), pool.accounting()
        with pytest.raises(BudgetExceeded):
            pool.step([3])
        assert np.array_equal(pool.observe(raw=True), before_pixels)
        assert pool.accounting() == before_counts


def test_restore_across_terminal_and_reset_keeps_charge_and_pixels(ale_available):
    with EmulatorPool([EmulatorConfig(seed=15, max_episode_frames=5)]) as pool:
        for _ in range(40):
            pool.step([0])
        token = pool.clone()  # terminal state; the next step resets before acting
        raw = pool.observe(raw=True)
        first = pool.step([3])[0]
        pool.restore(token)
        assert np.array_equal(pool.observe(raw=True), raw)
        second = pool.step([3])[0]
        assert np.array_equal(first, second)
        counts = pool.accounting()
        assert counts["experiment_frames"] == 152  # two resets72 + two repeats4
        assert counts["restore_charges"] == 60
        assert counts["reset_emulated_frames"] == 22 * 72
        pool.release(token)
