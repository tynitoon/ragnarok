"""Synthetic end-to-end ordinary/branch chain with RGB-only observations."""
from collections import deque
import numpy as np
import pytest
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.restorable_ram_history_v70 import _RestorableTripleRAMTap
from ragnarok.evaluation import visible_reference_pool_v70 as pool
from tests.test_restorable_ram_history_v70 import engine,Native,same_history


class Reference:
    def __init__(self):self.frames_seen=0
    def observe(self,rgb):self.frames_seen+=1;return self.extract(rgb)
    def extract(self,rgb):return np.array([[10+int(rgb[0,0,0])%100,20,8,10]])


def factory(monkeypatch,terminal=999):
    def create(config,callback):
        obj,_=engine(terminal);obj._ale._on_frame=callback;obj.loads=1
        obj.accounting=lambda:dict(obj._budget.report(),cold_loads=obj.loads)
        def cold_start(*,experimental):
            assert experimental is False
            obj._budget.record_play(132,agent_steps=0);obj.loads+=1
            native=Native(terminal);obj._ale=_RestorableTripleRAMTap(native,callback)
            obj._frames=deque([native.getScreenRGB(),native.getScreenRGB()],maxlen=2);obj._last_action=0
        obj._cold_start=cold_start;obj._bootstrap_frames=132
        return obj
    monkeypatch.setattr(pool,'_ScorerHistoryColdEngine',create)
    monkeypatch.setattr(pool,'ReferenceExtractor',Reference)
    monkeypatch.setattr(pool,'asteroids_visible_scene',lambda now,previous:
        [SceneObject('player:0','player',(10+int(now[0])%100,20,8,10))])


def test_removing_oracle_labels_cannot_change_rgb_boxes_or_branch_targets(monkeypatch):
    factory(monkeypatch)
    config=EmulatorConfig(game='asteroids')
    left,right=pool.VisibleReferenceEngine(config),pool.VisibleReferenceEngine(config)
    right._decoder=lambda now,previous:[]
    for i in range(128):
        a,b=left.step(i%2),right.step(i%2)
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    a,b=left.report(),right.report()
    assert a['self_generated_label_sha256']==b['self_generated_label_sha256']
    assert a['accounting']==b['accounting']
    assert a['perception']['matched_visible']>0 and b['perception']['matched_visible']==0
    assert a['ordinary_raw_frames']==512 and a['decoded_branch_frames']==8
    assert a['branch_pairs']==1 and left.reference.frames_seen==right.reference.frames_seen==128
    same_history(left.engine._ale.export_history(),right.engine._ale.export_history())


def test_terminal_branch_and_new_cold_first_act_keep_cached_sources_aligned(monkeypatch):
    factory(monkeypatch,terminal=509)
    subject=pool.VisibleReferenceEngine(EmulatorConfig(game='asteroids'))
    for i in range(128):subject.step(i%2)
    assert subject._ordinary_frames==509 and subject._lab_decoded==0
    assert subject.engine._ale.frame_numbers==(507,508,509)
    subject.step(0)
    assert subject._episode==1 and subject._ordinary_frames==513
    assert subject.engine._ale.frame_numbers==(2,3,4)
    assert len(subject._source_labels)==2 and subject.reference.frames_seen==129
    assert subject.scorer._episode==1


@pytest.mark.parametrize('game',['boxing','frostbite','riverraid','kangaroo'])
def test_only_two_declared_train_games_are_constructible(monkeypatch,game):
    monkeypatch.setattr(pool,'_ScorerHistoryColdEngine',lambda *args:pytest.fail('opened a held-out emulator'))
    with pytest.raises(ValueError,match='declared'):
        pool.VisibleReferenceEngine(EmulatorConfig(game=game))


def test_one_decoder_failure_permanently_stops_actions_but_keeps_partial_report(monkeypatch):
    factory(monkeypatch)
    subject=pool.VisibleReferenceEngine(EmulatorConfig(game='asteroids'))
    original=subject._decoder
    def fail_once(now,previous):
        subject._decoder=original
        raise RuntimeError('Synthetic transient decoder failure')
    subject._decoder=fail_once
    before=subject.engine._budget.charged_frames
    with pytest.raises(RuntimeError,match='transient'):subject.step(1)
    paid=subject.report()
    assert paid['accounting']['charged_frames']==before+1 and paid['failure']['error_type']=='RuntimeError'
    with pytest.raises(ValueError,match='Trajectory stopped'):subject.step(1)
    assert subject.report()==paid and subject.engine._ale.frame_numbers==(0,0,1)


def test_first_cold_transition_is_explicitly_outside_moving_eligibility(monkeypatch):
    factory(monkeypatch,terminal=1)
    subject=pool.VisibleReferenceEngine(EmulatorConfig(game='asteroids'))
    subject.step(0)
    report=subject.report()['perception']
    assert report['frames']==1 and report['per_type']['player']['visible']==1
    assert report['per_type']['player']['eligible']==0 and report['recall_moving'] is None


def test_pool_does_not_advance_other_workers_after_a_failed_batch(monkeypatch):
    subject=pool.VisibleReferencePool.__new__(pool.VisibleReferencePool);subject._step_failed=False
    sent=[]
    def failed(self,actions):sent.append(actions);raise ValueError('poisoned child')
    monkeypatch.setattr(pool.EmulatorPool,'step',failed)
    with pytest.raises(ValueError,match='poisoned child'):subject.step([1,0])
    with pytest.raises(ValueError,match='trajectory stopped'):subject.step([1,0])
    assert sent==[[1,0]]
