"""Exact external geometric witnesses, independent of a new game rollout."""
import numpy as np
import pytest

from ragnarok.evaluation.fast_visible_mspacman_v70 import _unoccluded_pills
from ragnarok.evaluation.vector_visible_mspacman_v70 import unoccluded_pills
from ragnarok.evaluation.mspacman_v70 import SlottedLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_temporal_groups_v70 import group_visible_sources
from ragnarok.evaluation.cached_visible_groups_v70 import CachedVisibleGroups


def test_all_256_pill_occlusion_patterns_equal_original_order_and_clipping():
    pills=[SlottedLabel('pill',(0,0,4,2),'edge'),SlottedLabel('pill',(4,2,4,2),'other')]
    for bits in range(256):
        occupied=np.zeros((4,8),dtype=bool)
        occupied[:2,:4]=np.array([(bits>>k)&1 for k in range(8)],bool).reshape(2,4)
        before=occupied.copy()
        assert unoccluded_pills(pills,occupied)==_unoccluded_pills(pills,occupied)
        np.testing.assert_array_equal(occupied,before)


def test_random_rectangle_occlusions_and_empty_population_are_exact():
    rng=np.random.default_rng(710)
    for _ in range(100):
        occupied=rng.random((20,30))<.4
        pills=[]
        for i in range(10):
            x,y=int(rng.integers(25)),int(rng.integers(15))
            w,h=int(rng.integers(1,6)),int(rng.integers(1,6))
            pills.append(SlottedLabel('power_pill',(x,y,w,h),str(i)))
        assert unoccluded_pills(pills,occupied)==_unoccluded_pills(pills,occupied)
    assert unoccluded_pills([],occupied)==[]


@pytest.mark.parametrize('game',['ms_pacman','asteroids'])
def test_cached_group_geometry_order_and_membership_match_through_changes(game):
    cache=CachedVisibleGroups(game)
    kind=cache.kinds[0]
    own=[SceneObject('a',kind,(10,30,4,2)),SceneObject('b',kind,(13,30,4,2)),SceneObject('c',kind,(40,60,2,2))]
    sequences=[own,own,own[::-1],own[:2],[],[],[own[0],own[0]],own]
    for objects in sequences:
        assert cache.group(objects)==group_visible_sources(objects,game=game)
        assert len(cache._cache)==len(cache.kinds)
    before=cache._cache
    with pytest.raises(ValueError):cache.group([SceneObject('bad',kind,(-1,30,4,2))])
    assert cache._cache is before


def test_cache_keeps_source_membership_separate_from_identical_geometry():
    cache=CachedVisibleGroups('ms_pacman')
    a=[SceneObject('a','ghost',(10,30,4,2))]
    b=[SceneObject('b','ghost',(10,30,4,2))]
    assert cache.group(a)[0].members==frozenset({'a'})
    assert cache.group(b)[0].members==frozenset({'b'})
