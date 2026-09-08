"""Cold episode construction, pre-admission and replay across native instances."""
import numpy as np
import pytest

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import ColdEmulatorPool, ROM_SPECS, _ColdPixelEngine
from ragnarok.environments.frame_budget_v70 import BudgetExceeded


@pytest.mark.parametrize("game", tuple(ROM_SPECS))
def test_each_new_episode_has_full_load_charge_and_fresh_initial_pixels(game):
    cost = ROM_SPECS[game][1]
    engine = _ColdPixelEngine(EmulatorConfig(game=game, seed=22000, max_episode_frames=5))
    assert engine.accounting()["charged_frames"] == cost
    initial = engine.observe()
    initial_ram = engine._ale.getRAM().copy()  # experimenter assertion only
    engine.step(3)
    engine.step(4)
    assert engine.accounting()["charged_frames"] == cost + 5
    old_native = engine._ale
    engine._cold_start(experimental=False)
    assert engine._ale is not old_native
    assert np.array_equal(initial, engine.observe())
    assert np.array_equal(initial_ram, engine._ale.getRAM())
    assert engine.accounting()["charged_frames"] == 2 * cost + 5
    assert engine.accounting()["format_detection_emulated_frames"] == 120
    assert engine.accounting()["cold_loads"] == 2


def test_reset_and_repeat_are_admitted_before_native_recreation(monkeypatch):
    import ale_py
    engine = _ColdPixelEngine(EmulatorConfig(max_episode_frames=1, max_charged_frames=267))
    engine.step(0)
    before = engine.accounting()
    before_pixels = engine.observe()
    def forbidden():
        raise AssertionError("A cold native must not be constructed without the full repeat budget")
    monkeypatch.setattr(ale_py, "ALEInterface", forbidden)
    #133 spent; next full request needs132+4, total269 >267.
    with pytest.raises(BudgetExceeded):
        engine.step(0)
    assert engine.accounting() == before
    assert np.array_equal(engine.observe(), before_pixels)


def test_initial_rejection_precedes_native_construction(monkeypatch):
    import ale_py
    def forbidden():
        raise AssertionError("No native construction below the initial charge")
    monkeypatch.setattr(ale_py, "ALEInterface", forbidden)
    with pytest.raises(BudgetExceeded):
        _ColdPixelEngine(EmulatorConfig(max_charged_frames=131))


def test_restoration_across_cold_instances_replays_and_never_refunds_loads():
    engine = _ColdPixelEngine(EmulatorConfig(seed=22001, max_episode_frames=5))
    for _ in range(20):
        engine.step(0)
    token = engine.clone()
    anchor = engine.observe()
    before = engine.accounting()
    first = engine.step(3)
    assert engine.accounting()["experiment_frames"] == 136
    engine.restore(token)
    assert np.array_equal(engine.observe(), anchor)
    second = engine.step(3)
    assert np.array_equal(first, second)
    after = engine.accounting()
    assert after["charged_frames"] - before["charged_frames"] == 136 + 60 + 136
    assert after["cold_loads"] == before["cold_loads"] + 2
    assert after["ordinary_frames"] == before["ordinary_frames"]
    engine.release(token)


def test_private_scorer_wrapper_survives_native_recreation():
    engine = _ColdPixelEngine(EmulatorConfig(max_episode_frames=1))
    seen = []
    def wrapper(native):
        seen.append(native)
        return native
    engine.set_native_wrapper(wrapper)
    engine.step(0)
    sticky_before = engine._sticky_rng.getstate()
    engine._cold_start(experimental=False)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert engine._sticky_rng.getstate() == sticky_before
    assert engine._last_action == 0


def test_pixel_rpc_uses_cold_backend_and_hides_scorer_hook():
    with ColdEmulatorPool([EmulatorConfig(game="seaquest", max_episode_frames=1)]) as pool:
        for forbidden in ("set_native_wrapper", "getRAM", "reset_game", "_cold_start"):
            with pytest.raises(ValueError, match="Unknown"):
                pool._one(0, forbidden)
        pool.step([0])
        pixels = pool.step([0])[0]
        assert pixels.dtype == np.uint8 and pixels.shape == (210, 160, 3)
        assert pool.accounting()["charged_frames"] == 2 * 128 + 2
