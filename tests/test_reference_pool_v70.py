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
        assert pool.report()["accounting"]["charged_frames"] == 136
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
    assert report["accounting"]["charged_frames"] == 213
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


@pytest.mark.parametrize("game,reset_convention", (("pong","natural_legacy"),("pong","cold"),("breakout","cold")))
def test_pooled_ram_pair_stays_aligned_through_terminal_branches_and_reset(game,reset_convention):
    pytest.importorskip("ale_py")
    config = EmulatorConfig(game=game,seed=18105, max_episode_frames=509, max_charged_frames=20_000)
    subject = _ReferenceEngine(config, observation_mode="maxpooled", reset_convention=reset_convention)
    if reset_convention == "cold":
        from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
        witness = _ColdPixelEngine(config)
        witness.set_native_wrapper(_PairRAMTap)
    else:
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


@pytest.mark.parametrize("game,reset_convention", (("pong","natural_legacy"),("pong","cold"),("breakout","cold")))
def test_pooled_observations_and_self_labels_ignore_changed_oracle(game,reset_convention):
    pytest.importorskip("ale_py")
    config = EmulatorConfig(game=game,seed=18106, max_charged_frames=20_000, max_episode_frames=300)
    left = _ReferenceEngine(config, observation_mode="maxpooled", reset_convention=reset_convention)
    right = _ReferenceEngine(config, observation_mode="maxpooled", reset_convention=reset_convention)
    right._decoder = lambda ram: []
    right._pooled_decoder = lambda previous,current: []
    for step in range(128):
        a, b = left.step((step//4)%18), right.step((step//4)%18)
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert left.report()["self_generated_label_sha256"] == right.report()["self_generated_label_sha256"]
    assert left.report()["perception"]["matched_visible"] > 0
    assert right.report()["perception"]["matched_visible"] == 0
    assert left.reference.frames_seen == right.reference.frames_seen == 128
    assert left._episode == right._episode == 1


def test_breakout_multiple_brick_instances_are_static_and_all_visible_are_matched():
    from ragnarok.evaluation.scene_adapters_v70 import pooled_breakout_scene_objects
    from ragnarok.evaluation.scene_perception_v70 import ScenePerceptionScorer, breakout_scene_objects
    ram=np.zeros(128,dtype=np.uint8)
    ram[72]=100
    ram[5+24]=1|16  # separated runs in the upper row
    judge=ScenePerceptionScorer(roi=(8,32,152,197))
    for frame in range(1,3):
        ram[72]+=1
        judge.observe_truth(breakout_scene_objects(ram),frame=frame,episode=0)
    truths=pooled_breakout_scene_objects(ram,ram)
    assert len([o for o in truths if o.kind=="bricks_0"])==2
    judge.score(np.array([o.box for o in truths]),frame_truths=truths)
    report=judge.report()
    assert report["precision_visible"]==1
    assert report["per_type"]["bricks_0"]["visible"]==2
    assert report["per_type"]["bricks_0"]["eligible"]==0
    assert report["per_type"]["player"]["eligible"]==1


def test_breakout_benchmark_requires_reserved_and_complete_calibration(tmp_path):
    import json
    from pathlib import Path
    from scripts.bench_reference_pool_v70 import benchmark
    result=json.loads(Path("craft_v6_out/v70_breakout_pooled/validation_result.json").read_text())
    path=tmp_path/"calibration.json"
    result["seed"]=17121
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match="reserved"):
        benchmark(1,256,game="breakout",observation_mode="maxpooled",calibration=path)
    result["seed"]=17122
    result["source_sha256"]={}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match="incomplete"):
        benchmark(1,256,game="breakout",observation_mode="maxpooled",calibration=path)
