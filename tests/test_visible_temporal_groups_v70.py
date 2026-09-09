"""Component topology, true source membership and static-object eligibility."""
import numpy as np
import pytest
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_temporal_groups_v70 import group_visible_sources,VisibleTemporalScorer


def obj(name,box,kind='ghost'):
    return SceneObject(f'{kind}:{name}',kind,box,kind in ('player','ghost','fruit'))


@pytest.mark.parametrize('a,b,expected',[
    ((1,1,2,2),(3,3,2,2),[(1,1,2,2),(3,3,2,2)]),
    ((1,1,2,2),(3,2,2,2),[(1,1,4,3)]),
    ((1,1,3,3),(2,2,3,3),[(1,1,4,4)]),
    ((158,10,2,4),(0,10,2,4),[(0,10,2,4),(158,10,2,4)]),
])
def test_overlap_side_contact_corner_and_screen_wrap(a,b,expected):
    groups=group_visible_sources([obj('a',a),obj('b',b)],game='ms_pacman')
    assert [g.label.box for g in groups]==expected
    assert set().union(*(g.members for g in groups))=={'ghost:a','ghost:b'}


def test_envelope_hole_does_not_create_a_fake_connection():
    groups=group_visible_sources([obj('a',(1,1,1,5)),obj('b',(1,5,5,1)),obj('c',(4,2,1,1))],game='ms_pacman')
    assert [g.label.box for g in groups]==[(1,1,5,5),(4,2,1,1)]
    assert groups[0].members==frozenset(('ghost:a','ghost:b'))
    assert groups[1].members==frozenset(('ghost:c',))


def test_identical_source_duplicates_do_not_duplicate_components_or_members():
    a=obj('a',(20,20,8,5));pill=obj('p',(20,20,4,2),'pill')
    groups=group_visible_sources([a,a,pill],game='ms_pacman')
    assert len(groups)==2 and groups[0].members==frozenset(('ghost:a',))
    assert [g.label.kind for g in groups]==['ghost','pill']


def test_pooled_component_inherits_raw_member_motion_but_pills_stay_static():
    scorer=VisibleTemporalScorer('ms_pacman')
    before=[obj('a',(10,20,8,10)),obj('b',(18,20,8,10)),obj('p',(40,20,4,2),'pill')]
    now=[obj('a',(11,20,8,10)),obj('b',(18,20,8,10)),obj('p',(41,20,3,2),'pill')]
    scorer.observe_truth(before,frame=0,episode=0)
    scorer.observe_truth(now,frame=1,episode=0)
    groups=group_visible_sources(before+now,game='ms_pacman')
    old_moving=scorer._moving.copy()
    scorer.score(np.array([g.label.box for g in groups]),frame_truths=groups)
    report=scorer.report()
    assert report['per_type']['ghost']['eligible']==1 and report['per_type']['ghost']['recalled']==1
    assert report['per_type']['pill']['eligible']==0
    assert scorer._moving==old_moving and report['per_type']['fruit']['recall_moving'] is None


def test_roi_violation_is_rejected_instead_of_clipped_differently():
    with pytest.raises(ValueError,match='outside'):
        group_visible_sources([obj('a',(1,170,2,2))],game='ms_pacman')
