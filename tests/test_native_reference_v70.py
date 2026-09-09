from pathlib import Path

import numpy as np
import pytest

from ragnarok.evaluation.native_reference_v70 import NativeReferenceExtractor
from scripts.ref_extract_v70 import ReferenceExtractor

BUILD=Path("craft_v6_out/v70_native_reference/build_v0/build.json")


def same_state(a,b):
    assert a.frames_seen==b.frames_seen and a._palette_size==b._palette_size
    for attr in ("_counts","_best","_mode","_colour_ids"):
        assert np.array_equal(getattr(a,attr),getattr(b,attr))


def test_native_fused_loop_equals_numpy_after_every_update_and_readonly_branch():
    a=ReferenceExtractor(max_colours=32)
    b=NativeReferenceExtractor(build_manifest=BUILD,max_colours=32)
    rng=np.random.default_rng(70371)
    palette=rng.integers(0,256,(24,3),dtype=np.uint8)
    for frame in range(80):
        rgb=palette[rng.integers(0,min(24,frame+1),(13,21))]
        # Exercise non-contiguous inputs as well as palette growth through capacity16.
        rgb=rgb[:,::-1]
        assert np.array_equal(a.observe(rgb),b.observe(rgb))
        same_state(a,b)
        branch=palette[rng.integers(0,24,(13,21))]
        assert np.array_equal(a.extract(branch),b.extract(branch))
        same_state(a,b)


def test_ties_capacity_error_and_counter_limit_keep_identical_state():
    a=ReferenceExtractor(max_colours=2)
    b=NativeReferenceExtractor(build_manifest=BUILD,max_colours=2)
    for value in [10,20,20,10,10,20]:
        rgb=np.full((4,6,3),value,dtype=np.uint8)
        assert np.array_equal(a.observe(rgb),b.observe(rgb))
        same_state(a,b)
    for obj in (a,b):
        with pytest.raises(ValueError,match="capacity"):
            obj.observe(np.full((4,6,3),30,dtype=np.uint8))
    same_state(a,b)
    a.frames_seen=b.frames_seen=1_000_000
    for obj in (a,b):
        with pytest.raises(ValueError,match="limit"):
            obj.observe(np.full((4,6,3),10,dtype=np.uint8))
    same_state(a,b)


def test_native_palette_indices_exceed_uint16_offsets_without_overflow():
    rng=np.random.default_rng(71091)
    # 4096 colours, maximum allowed palette. Output equality includes every count.
    colours=np.arange(4096,dtype=np.uint32).view(np.uint8).reshape(4096,4)[:,:3]
    a=ReferenceExtractor(max_colours=4096)
    b=NativeReferenceExtractor(build_manifest=BUILD,max_colours=4096)
    for order in [np.arange(4096),rng.permutation(4096)]:
        rgb=colours[order].reshape(64,64,3)
        assert np.array_equal(a.observe(rgb),b.observe(rgb))
        same_state(a,b)


def test_invalid_native_colour_is_rejected_before_mutating_counters():
    b=NativeReferenceExtractor(build_manifest=BUILD)
    b.observe(np.zeros((4,6,3),dtype=np.uint8))
    old_counts,old_mode,old_best=b._counts.copy(),b._mode.copy(),b._best.copy()
    bad=np.zeros((4,6),dtype=np.uint32)
    bad[-1,-1]=1<<24
    with pytest.raises(ValueError,match="before native mutation"):
        b._update_packed(bad)
    assert np.array_equal(b._counts,old_counts)
    assert np.array_equal(b._mode,old_mode) and np.array_equal(b._best,old_best)
    assert b.frames_seen==1


@pytest.mark.parametrize("game",["pong","breakout"])
def test_native_reference_engine_preserves_images_boxes_scores_and_branch_targets(game):
    from ragnarok.environments.emu_pool import EmulatorConfig
    from ragnarok.evaluation.reference_pool_v70 import _ReferenceEngine
    config=EmulatorConfig(game=game,seed=18106,max_charged_frames=20_000,max_episode_frames=300)
    a=_ReferenceEngine(config,observation_mode="maxpooled",reset_convention="cold")
    b=_ReferenceEngine(config,observation_mode="maxpooled",reset_convention="cold",
                       reference_backend="native",native_build=BUILD)
    for step in range(128):
        left,right=a.step((step//4)%18),b.step((step//4)%18)
        assert np.array_equal(left[0],right[0]) and np.array_equal(left[1],right[1])
    assert a.report()==b.report()
    assert a.report()["branch_pairs"]==1
    same_state(a.reference,b.reference)
