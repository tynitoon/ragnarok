from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.fast_pooled_seaquest_v70 import EnvelopeGroup
from ragnarok.evaluation.seaquest_scene_v70 import SeaquestSceneScorer


def test_split_components_with_identical_members_count_twice_and_preserve_history():
    scorer=SeaquestSceneScorer()
    scorer.observe_truth([SceneObject("shark:0","shark",(10,50,8,7))],frame=0,episode=0)
    scorer.observe_truth([SceneObject("shark:0","shark",(30,50,8,7))],frame=1,episode=0)
    groups=[EnvelopeGroup(ObjectLabel("shark",box),frozenset(("shark:0",)))
            for box in ((10,50,8,7),(30,50,8,7))]
    raw_moving=scorer._moving.copy()
    scorer.score([(10,50,8,7)],frame_truths=groups)
    result=scorer.report()
    assert result["eligible_moving"]==2 and result["recalled_moving"]==1
    assert result["precision_visible"]==1
    assert scorer._moving==raw_moving
    assert result["per_type"]["player"]["recall_moving"] is None


def test_any_moving_member_and_stationary_precision_without_empty_success():
    scorer=SeaquestSceneScorer()
    for frame,x in ((0,10),(1,11)):
        scorer.observe_truth([SceneObject("shark:a","shark",(x,50,8,7)),
                              SceneObject("shark:b","shark",(19,50,8,7)),
                              SceneObject("player:p","player",(70,50,16,11))],frame=frame,episode=0)
    groups=[EnvelopeGroup(ObjectLabel("shark",(11,50,16,7)),frozenset(("shark:a","shark:b"))),
            EnvelopeGroup(ObjectLabel("player",(70,50,16,11)),frozenset(("player:p",)))]
    scorer.score([(11,50,16,7),(70,50,16,11),(100,50,8,7)],frame_truths=groups)
    result=scorer.report()
    assert result["eligible_moving"]==1 and result["recalled_moving"]==1
    assert result["matched_visible"]==2 and result["predictions"]==3
    scorer.observe_truth([SceneObject("shark:a","shark",(11,50,8,7))],frame=2,episode=1)
    scorer.score([(11,50,16,7)],frame_truths=groups[:1])
    assert scorer.report()["eligible_moving"]==1  # no motion inherited across cold reset
