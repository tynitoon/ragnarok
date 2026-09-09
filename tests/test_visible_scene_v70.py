"""Static pill occlusion must not fabricate moving-object reference recall."""
import numpy as np
from ragnarok.evaluation.mspacman_v70 import SlottedLabel
from ragnarok.evaluation import visible_scene_v70 as scene
from ragnarok.evaluation.scene_perception_v70 import ScenePerceptionScorer


def test_pills_remain_static_despite_occlusion_and_slot_names_are_category_scoped(monkeypatch):
    objects=[SlottedLabel('pill',(10,10,4,2),'0'),
             SlottedLabel('power_pill',(20,10,4,7),'0'),
             SlottedLabel('player',(30,10,8,10),'0'),
             SlottedLabel('ghost',(40,10,8,10),'draw:0:0'),
             SlottedLabel('fruit',(50,10,8,10),'0')]
    observed=[]
    def decode(current,previous):observed.append((current,previous));return objects
    monkeypatch.setattr(scene,'decode_visible_ram',decode)
    current,previous=object(),object()
    first=scene.mspacman_visible_scene(current,previous)
    assert observed==[(current,previous)]
    assert [o.can_move for o in first]==[False,False,True,True,True]
    assert len({o.instance_id for o in first})==5
    scorer=ScenePerceptionScorer(roi=(0,0,160,171))
    scorer.add(first,np.array([o.box for o in first]),frame=0,episode=0)
    objects[:]=[SlottedLabel(o.kind,(o.box[0]+1,*o.box[1:]),o.slot) for o in objects]
    second=scene.mspacman_visible_scene(current,previous)
    scorer.add(second,np.array([o.box for o in second]),frame=1,episode=0)
    report=scorer.report()
    assert report['per_type']['pill']['eligible']==0
    assert report['per_type']['power_pill']['eligible']==0
    assert report['per_type']['ghost']['eligible']==1


def test_asteroids_visible_pieces_are_not_collapsed_into_a_single_category_id(monkeypatch):
    objects=[SlottedLabel('asteroid',(0,20,2,2),'3:0'),SlottedLabel('asteroid',(158,20,2,2),'3:1'),
             SlottedLabel('player',(30,20,6,10),'0')]
    monkeypatch.setattr(scene,'decode_visible_asteroids_ram',lambda now,before:objects)
    got=scene.asteroids_visible_scene(None,None)
    assert len({o.instance_id for o in got})==3 and all(o.can_move for o in got)
    assert [o.box for o in got]==[o.box for o in objects]
