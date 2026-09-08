"""Actual separation and count checks for the distributed reference instrument."""
import numpy as np
import pytest

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.reference_pool_v70 import ReferencePool, _ReferenceEngine, _RawBranchEngine
from ragnarok.environments.emu_pool import _PixelEngine
from scripts.capture_pong_pooled_v70 import _PairRAMTap


def test_pixel_outputs_and_c_targets_do_not_depend_on_ram_labels():
    pytest.importorskip("ale_py")
    config = EmulatorConfig(seed=18101, max_charged_frames=20_000)
    left, right = _ReferenceEngine(config), _ReferenceEngine(config)
    right._decoder = lambda ram: []  # counterfactual oracle, identical real emulator and images
    for step in range(128):
        a, b = left.step((step // 4) % 18), right.step((step // 4) % 18)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
    normal, corrupt = left.report(), right.report()
    assert normal["perception"]["matched_visible"] > 0
    assert corrupt["perception"]["matched_visible"] == 0
    assert normal["self_generated_label_sha256"] == corrupt["self_generated_label_sha256"]
    assert normal["accounting"] == corrupt["accounting"]
    assert normal["branch_pairs"] == 1
    assert normal["accounting"]["ordinary_agent_steps"] == 128
    assert normal["accounting"]["experiment_agent_steps"] == 2
    assert normal["accounting"]["restores"] == 2
    assert normal["perception"]["frames"] == 128
    assert normal["perception"]["observed_raw_frames"] == 512
    assert normal["decoded_branch_frames"] == 8


def test_reference_pool_keeps_ram_and_aggregates_out_of_step_payload():
    pytest.importorskip("ale_py")
    with ReferencePool([EmulatorConfig(seed=18102, max_charged_frames=1000)]) as pool:
        for forbidden in ("getRAM", "_capture", "lives", "reward", "decode_pong_ram"):
            with pytest.raises(ValueError, match="Unknown"):
                pool._one(0, forbidden)
        image, boxes = pool.step([3])[0]
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8
        assert image.shape == (210,160,3)
        assert isinstance(boxes, np.ndarray) and boxes.shape[1] == 4
        assert pool.report()["accounting"]["charged_frames"] == 76
        processes = list(pool._processes)
    assert all(not process.is_alive() for process in processes)


def test_reference_history_resets_at_actual_ordinary_reset_not_at_branch_restore():
    pytest.importorskip("ale_py")
    engine = _ReferenceEngine(EmulatorConfig(seed=18103, max_episode_frames=5, max_charged_frames=1000))
    engine.step(0)  # 4 raw frames
    engine.step(0)  # 1 terminal raw frame
    engine.step(0)  # 72 reset +4 raw frames
    report = engine.report()
    assert report["ordinary_raw_frames"] == 9
    assert report["accounting"]["reset_emulated_frames"] == 144
    assert report["accounting"]["charged_frames"] == 153
    assert report["perception"]["observed_raw_frames"] == 9
    assert engine._episode == 1
    assert engine.reference.frames_seen == 3  # reference receives no oracle episode boundary


def test_branch_endpoints_use_the_same_raw_convention_as_ordinary_images():
    pytest.importorskip("ale_py")
    engine = _RawBranchEngine(EmulatorConfig(seed=18104, experiment_fraction=.3))
    for i in range(128):
        engine.step((i // 8) % 18)
    anchor = engine._capture()
    engine._advance(3,4,experimental=True)
    expected = engine.observe(raw=True)
    assert not np.array_equal(expected, engine.observe(raw=False)), "Fixture must distinguish raw and pooled branches"
    engine._restore(anchor)
    intervention, control = engine.branch_pair(3,control=3)
    assert np.array_equal(intervention,expected)
    assert np.array_equal(control,expected)


def test_pooled_ram_pair_stays_aligned_through_terminal_branches_and_reset():
    pytest.importorskip("ale_py")
    config = EmulatorConfig(seed=18105, max_episode_frames=509, max_charged_frames=20_000)
    subject = _ReferenceEngine(config, observation_mode="maxpooled")
    witness = _PixelEngine(config)
    witness._ale = _PairRAMTap(witness._ale)
    for step in range(129):
        action = (step // 4) % 18
        pixels, _ = subject.step(action)
        expected = witness.step(action)
        assert np.array_equal(pixels,expected)
        assert np.array_equal(subject.engine._ale.previous_ram,witness._ale.previous)
        assert np.array_equal(subject.engine._ale.current_ram,witness._ale.current)
    assert subject.report()["branch_pairs"] == 1
    assert subject.report()["decoded_branch_frames"] == 0  # pair at terminal, both restores still charged
    assert subject.report()["accounting"]["restores"] == 2
    assert subject.report()["accounting"]["reset_emulated_frames"] == 144


def test_pooled_observations_and_self_labels_ignore_changed_oracle():
    pytest.importorskip("ale_py")
    config = EmulatorConfig(seed=18106, max_charged_frames=20_000)
    left = _ReferenceEngine(config, observation_mode="maxpooled")
    right = _ReferenceEngine(config, observation_mode="maxpooled")
    right._decoder = lambda ram: []
    right._pooled_decoder = lambda previous,current: []
    for step in range(128):
        a, b = left.step((step//4)%18), right.step((step//4)%18)
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert left.report()["self_generated_label_sha256"] == right.report()["self_generated_label_sha256"]
    assert left.report()["perception"]["matched_visible"] > 0
    assert right.report()["perception"]["matched_visible"] == 0
