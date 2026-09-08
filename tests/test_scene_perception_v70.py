import numpy as np
import pytest

from ragnarok.evaluation.scene_perception_v70 import SceneObject as Obj, ScenePerceptionScorer


def test_repeated_categories_keep_independent_ids_and_motion_denominators():
    scorer=ScenePerceptionScorer(roi=(0,0,160,210))
    first=[Obj("a","car",(10,50,4,4)),Obj("b","car",(30,50,4,4))]
    second=[Obj("a","car",(12,50,4,4)),Obj("b","car",(30,50,4,4))]
    scorer.add(first,[o.box for o in first],frame=0,episode=0)
    scorer.add(second,[o.box for o in second],frame=1,episode=0)
    report=scorer.report()
    assert report["eligible_moving"]==1 and report["recalled_moving"]==1
    assert report["per_type"]["car"]["visible"]==4
    assert report["precision_visible"]==1


def test_static_split_and_shrink_are_not_motion_but_remain_visible_matches():
    scorer=ScenePerceptionScorer(roi=(0,0,160,210))
    before=[Obj("row:8","brick",(8,57,32,6),False)]
    after=[Obj("row:8","brick",(8,57,8,6),False),Obj("row:24","brick",(24,57,16,6),False)]
    scorer.add(before,[o.box for o in before],frame=0,episode=0)
    scorer.add(after,[o.box for o in after],frame=1,episode=0)
    report=scorer.report()
    assert report["matched_visible"]==3 and report["precision_visible"]==1
    assert report["eligible_moving"]==0 and report["recall_moving"] is None


def test_matching_all_visible_objects_cannot_reuse_one_prediction_for_two_ids():
    scorer=ScenePerceptionScorer(roi=(0,0,160,210))
    for frame in range(2):
        objs=[Obj("a","ball",(10+frame,50,2,2)),Obj("b","ball",(12+frame,50,2,2))]
        scorer.add(objs,[[11+frame,50,2,2]],frame=frame,episode=0)
    assert scorer.report()["eligible_moving"]==2
    assert scorer.report()["recalled_moving"]==1


def test_episode_boundaries_clear_judge_motion_and_bad_ids_leave_counts_unchanged():
    scorer=ScenePerceptionScorer(roi=(0,0,160,210))
    scorer.add([Obj("ball","ball",(10,50,2,2))],[[10,50,2,2]],frame=0,episode=0)
    scorer.add([Obj("ball","ball",(90,50,2,2))],[[90,50,2,2]],frame=1,episode=1)
    assert scorer.report()["eligible_moving"]==0
    old=scorer.report()
    with pytest.raises(ValueError,match="unique"):
        scorer.observe_truth([Obj("x","a",(0,0,2,2)),Obj("x","b",(20,20,2,2))],frame=2,episode=1)
    assert old==scorer.report()
    scorer.add([],np.empty((0,4)),frame=2,episode=1)
