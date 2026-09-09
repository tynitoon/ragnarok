"""Causal decoder wiring, contributor topology, gate thresholds and fail-closed capture."""
import json
import numpy as np
import pytest
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation import pooled_mspacman_v70 as pooled
from scripts import capture_mspacman_pooled_v70 as capture
from scripts.calibrate_mspacman_pooled_v70 import new_stats,accumulate,finish


def test_ram_gets_each_source_predecessor_and_no_rgb(monkeypatch):
    calls=[]
    def decode(now,before):
        calls.append((int(before[0]),int(now[0])))
        return [ObjectLabel('player',(int(now[0]),10,1,1))]
    monkeypatch.setattr(pooled,'decode_visible_ram',decode)
    r=[np.full(128,n,dtype=np.uint8) for n in (2,3,4)]
    labels=pooled.pooled_mspacman_ram_objects(*r)
    assert calls==[(2,3),(3,4)]
    assert labels==[ObjectLabel('player',(3,10,2,1))]


def test_union_joins_side_contact_but_not_diagonal_or_opposite_edges():
    masks=pooled._masks()
    for box in ((0,10,1,1),(1,10,1,1),(2,11,1,1),(159,10,1,1)):
        pooled._add(masks,ObjectLabel('player',box))
    assert {o.box for o in pooled._objects(masks)}=={(0,10,2,1),(2,11,1,1),(159,10,1,1)}


class FakeRaw:
    def __init__(self,labels):self.labels=labels;self.seen=[]
    def step(self,rgb):self.seen.append(rgb);return self.labels[len(self.seen)-1]


def test_contributors_are_reboxed_and_filled_before_connecting():
    a=np.zeros((210,160,3),dtype=np.uint8);b=a.copy()
    a[10,10]=a[10,13]=pooled.COLOURS['player'][0]
    judge=pooled.MsPacmanPooledPixelJudge()
    judge.raw=FakeRaw([[ObjectLabel('player',(10,10,4,1))],[]])
    assert judge.step(a,b)==[ObjectLabel('player',(10,10,4,1))]
    assert judge.raw.seen[0] is a and judge.raw.seen[1] is b


@pytest.mark.parametrize('other,expected',[( (255,255,255),False),((210,164,74),True),((255,0,0),True)])
def test_source_needs_one_winning_channel_or_all_channel_tie(other,expected):
    a=np.zeros((210,160,3),dtype=np.uint8);b=a.copy()
    a[10,10]=pooled.COLOURS['player'][0];b[10,10]=other
    judge=pooled.MsPacmanPooledPixelJudge();judge.raw=FakeRaw([[ObjectLabel('player',(10,10,1,1))],[]])
    assert bool(judge.step(a,b))==expected


def test_sparse_rgb_history_persists_across_windows_without_episode_input():
    frames=[np.zeros((210,160,3),dtype=np.uint8) for _ in range(3)]
    judge=pooled.MsPacmanPooledPixelJudge();judge.raw=FakeRaw([[],[],[],[]])
    judge.step(frames[0],frames[1]);judge.step(frames[1],frames[2])
    assert all(a is b for a,b in zip(judge.raw.seen,[frames[0],frames[1],frames[1],frames[2]]))


def test_empty_category_keeps_null_and_fails():
    s=new_stats()
    for _ in range(100):accumulate(s,[],[])
    finish(s)
    assert s['presence_fraction']==1 and s['positive_presence_fraction'] is None
    assert s['box_fraction'] is None and not s['passed']


@pytest.mark.parametrize('matches,errors,passed',[(49,0,False),(50,0,True),(100,1,True),(99,1,False)])
def test_minimum_and_one_percent_boundary(matches,errors,passed):
    s=new_stats();box=ObjectLabel('player',(10,10,2,2));far=ObjectLabel('player',(20,20,2,2))
    for i in range(matches):accumulate(s,[box],[far if i<errors else box])
    finish(s)
    assert s['passed']==passed


def test_absent_windows_cannot_dilute_positive_count_failures():
    s=new_stats();box=ObjectLabel('player',(10,10,2,2))
    for _ in range(9900):accumulate(s,[],[])
    for _ in range(98):accumulate(s,[box],[box])
    for _ in range(2):accumulate(s,[box],[])
    finish(s)
    assert s['presence_fraction']>.99 and s['positive_presence_fraction']==.98 and not s['passed']


def test_failed_constructor_is_recorded_and_cannot_be_retried(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,'DESTINATION',tmp_path/'capture')
    monkeypatch.setattr(capture,'ATTEMPT',tmp_path/'attempt.json')
    monkeypatch.setattr(capture,'check',lambda:dict(proof='checked'))
    monkeypatch.setattr(capture,'git',lambda *args:'test-commit')
    count=[]
    def failed(config):
        count.append(config)
        assert json.loads(capture.ATTEMPT.read_text())['status']=='started'
        raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(capture,'_ColdPixelEngine',failed)
    with pytest.raises(RuntimeError,match='synthetic'):capture.capture()
    assert json.loads(capture.ATTEMPT.read_text())['status']=='failed'
    with pytest.raises(ValueError,match='earlier attempt'):capture.capture()
    assert len(count)==1


def test_failed_prerequisite_never_constructs_or_burns_seed(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,'DESTINATION',tmp_path/'capture')
    monkeypatch.setattr(capture,'ATTEMPT',tmp_path/'attempt.json')
    def reject():raise ValueError('raw proof missing')
    monkeypatch.setattr(capture,'check',reject)
    monkeypatch.setattr(capture,'_ColdPixelEngine',lambda _:pytest.fail('emulator constructed'))
    with pytest.raises(ValueError,match='raw proof'):capture.capture()
    assert not capture.ATTEMPT.exists() and not capture.DESTINATION.exists()


def test_failure_after_construction_keeps_partial_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,'DESTINATION',tmp_path/'capture')
    monkeypatch.setattr(capture,'ATTEMPT',tmp_path/'attempt.json')
    monkeypatch.setattr(capture,'check',lambda:dict(proof='checked'))
    monkeypatch.setattr(capture,'git',lambda *args:'test-commit')
    class Engine:
        def set_native_wrapper(self,_):raise RuntimeError('synthetic hook failure')
        def accounting(self):return dict(charged_frames=132,cold_loads=1)
    monkeypatch.setattr(capture,'_ColdPixelEngine',lambda _:Engine())
    with pytest.raises(RuntimeError,match='hook failure'):capture.capture()
    record=json.loads(capture.ATTEMPT.read_text())
    assert record['status']=='failed' and record['completed_decisions']==0
    assert record['accounting_status']=='partial_live_ledger' and record['accounting']['charged_frames']==132


def test_failed_scoring_has_its_own_durable_record(tmp_path,monkeypatch):
    from scripts import calibrate_mspacman_pooled_v70 as scorer
    raw=tmp_path/'raw';raw.mkdir();(raw/'manifest.json').write_text('{}')
    monkeypatch.setattr(scorer,'DESTINATION',raw)
    out=tmp_path/'score.json'
    def reject():
        assert json.loads((tmp_path/'score_attempt.json').read_text())['status']=='started'
        raise ValueError('synthetic malformed fixture')
    monkeypatch.setattr(scorer,'calibrate',reject)
    with pytest.raises(ValueError,match='malformed'):scorer.run(out)
    record=json.loads((tmp_path/'score_attempt.json').read_text())
    assert record['status']=='failed' and record['fixture_manifest_sha256']==scorer.sha(raw/'manifest.json')
    assert not out.exists() and (raw/'manifest.json').read_text()=='{}'
    with pytest.raises(ValueError,match='previous result'):scorer.run(out)
