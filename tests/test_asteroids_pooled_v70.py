"""Prospective Asteroids temporal criteria, source chronology and ROI witnesses."""
import json
import numpy as np
import pytest
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation import pooled_asteroids_v70 as pooled
from scripts import capture_asteroids_pooled_v70 as capture
from scripts.calibrate_asteroids_pooled_v70 import new_stats,accumulate,finish


def test_ram_uses_both_source_predecessors(monkeypatch):
    calls=[]
    def decode(now,before):
        calls.append((int(before[0]),int(now[0])))
        return [ObjectLabel('player',(int(now[0]),20,1,1))]
    monkeypatch.setattr(pooled,'decode_visible_asteroids_ram',decode)
    r=[np.full(128,i,dtype=np.uint8) for i in (2,3,4)]
    assert pooled.pooled_asteroids_ram_objects(*r)==[ObjectLabel('player',(3,20,2,1))]
    assert calls==[(2,3),(3,4)]


def test_spatial_union_does_not_join_diagonal_or_torus_edges():
    masks=pooled._masks()
    for box in ((0,20,1,1),(1,20,1,1),(2,21,1,1),(159,20,1,1)):
        pooled._add(masks,ObjectLabel('asteroid',box))
    assert {o.box for o in pooled._objects(masks)}=={(0,20,2,1),(2,21,1,1),(159,20,1,1)}


@pytest.mark.parametrize('box',[(10,15,1,1),(10,194,1,1),(10,193,1,2),(159,20,2,1)])
def test_roi_exclusive194_is_enforced(box):
    with pytest.raises(ValueError,match='ROI'):pooled._add(pooled._masks(),ObjectLabel('player',box))


def test_raw_rgb_decoder_only_sees_source_images_and_contributors_are_reboxed(monkeypatch):
    a=np.zeros((210,160,3),dtype=np.uint8);b=a.copy()
    a[20,10]=a[20,13]=pooled.PLAYER_COLOUR
    seen=[]
    def raw(rgb):
        seen.append(rgb)
        return [ObjectLabel('player',(10,20,4,1))] if rgb is a else []
    monkeypatch.setattr(pooled,'screen_visible_asteroids_objects',raw)
    assert pooled.pooled_asteroids_screen_objects(a,b)==[ObjectLabel('player',(10,20,4,1))]
    assert seen[0] is a and seen[1] is b


@pytest.mark.parametrize('other,present',[((255,255,255),False),((240,128,128),True),((255,0,0),True)])
def test_max_channel_contribution(other,present,monkeypatch):
    a=np.zeros((210,160,3),dtype=np.uint8);b=a.copy()
    a[20,10]=pooled.PLAYER_COLOUR;b[20,10]=other
    monkeypatch.setattr(pooled,'screen_visible_asteroids_objects',
        lambda rgb:[ObjectLabel('player',(10,20,1,1))] if rgb is a else [])
    assert bool(pooled.pooled_asteroids_screen_objects(a,b))==present


def test_empty_categories_and_sparse_false_negatives_do_not_pass():
    s=new_stats()
    for _ in range(9900):accumulate(s,[],[])
    finish(s)
    assert s['positive_presence_fraction'] is None and s['box_fraction'] is None and not s['passed']
    box=ObjectLabel('player',(10,20,2,2))
    for _ in range(98):accumulate(s,[box],[box])
    for _ in range(2):accumulate(s,[box],[])
    finish(s)
    assert s['presence_fraction']>.99 and s['positive_presence_fraction']==.98 and not s['passed']


@pytest.mark.parametrize('count,errors,passed',[(49,0,False),(50,0,True),(100,1,True),(99,1,False)])
def test_fixed_minimum_and_threshold(count,errors,passed):
    s=new_stats();a=ObjectLabel('player',(10,20,2,2));b=ObjectLabel('player',(20,30,2,2))
    for i in range(count):accumulate(s,[a],[b if i<errors else a])
    finish(s)
    assert s['passed']==passed


def test_durable_failed_capture_records_partial_cost_and_cannot_repeat(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,'DESTINATION',tmp_path/'raw')
    monkeypatch.setattr(capture,'ATTEMPT',tmp_path/'attempt.json')
    monkeypatch.setattr(capture,'check',lambda:dict(proof='checked'))
    monkeypatch.setattr(capture,'git',lambda *a:'test-commit')
    class Engine:
        def set_native_wrapper(self,_):
            assert json.loads(capture.ATTEMPT.read_text())['status']=='started'
            raise RuntimeError('synthetic hook failure')
        def accounting(self):return dict(charged_frames=132,cold_loads=1)
    monkeypatch.setattr(capture,'_ColdPixelEngine',lambda _:Engine())
    with pytest.raises(RuntimeError,match='hook failure'):capture.capture()
    record=json.loads(capture.ATTEMPT.read_text())
    assert record['status']=='failed' and record['accounting']['charged_frames']==132
    with pytest.raises(ValueError,match='earlier attempt'):capture.capture()


def test_failed_score_records_error_separately(tmp_path,monkeypatch):
    from scripts import calibrate_asteroids_pooled_v70 as scorer
    raw=tmp_path/'raw';raw.mkdir();(raw/'manifest.json').write_text('{}')
    monkeypatch.setattr(scorer,'DESTINATION',raw)
    out=tmp_path/'score.json'
    def reject():raise ValueError('synthetic unknown RAM')
    monkeypatch.setattr(scorer,'calibrate',reject)
    with pytest.raises(ValueError,match='unknown RAM'):scorer.run(out)
    assert json.loads((tmp_path/'score_attempt.json').read_text())['status']=='failed'
    assert not out.exists()
