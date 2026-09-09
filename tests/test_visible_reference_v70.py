"""Reference receives RGB only; fresh proof and failure ledgers are enforced."""
import json
import numpy as np
import pytest
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from scripts import visible_reference_v70 as runner


def test_ram_episode_reset_never_resets_or_informs_the_rgb_reference(monkeypatch):
    seen=[];history=[]
    class Reference:
        def observe(self,rgb):
            seen.append(rgb)
            return np.array([[10+int(rgb[0,0,0]),20,2,2]])
    def decode(ram,previous):
        history.append(int(previous[0]))
        return [SceneObject('player:0','player',(10+int(ram[0]),20,2,2))]
    monkeypatch.setattr(runner,'ReferenceExtractor',Reference)
    monkeypatch.setattr(runner,'mspacman_visible_scene',decode)
    rows=[]
    for v,episode,prior in ((0,0,9),(1,0,0),(2,1,7)):
        rgb=np.zeros((210,160,3),dtype=np.uint8);rgb[0,0,0]=v
        rows.append(dict(rgb=rgb,ram=np.full(128,v,dtype=np.uint8),episode=episode,
            ram_previous=np.full(128,prior,dtype=np.uint8)))
    result=runner.measure_frames('ms_pacman',rows)
    assert history==[9,0,7] and len(seen)==3
    assert all(a is b['rgb'] for a,b in zip(seen,rows))
    assert result['metrics']['frames']==3
    assert result['metrics']['per_type']['player']['eligible']==1
    assert result['metrics']['per_type']['player']['recalled']==1
    assert result['metrics']['per_type']['fruit']['recall_moving'] is None


def fake_paths(tmp_path,monkeypatch):
    values=(tmp_path/'raw',tmp_path/'capture_attempt.json',tmp_path/'result.json')
    monkeypatch.setattr(runner,'paths',lambda game:values)
    monkeypatch.setattr(runner,'git',lambda *args:'test-commit')
    return values


def test_actual_raw_history_reader_feeds_python_episode_indices_to_scorer(tmp_path,monkeypatch):
    from tests.test_raw_history_fixtures_v70 import fixture
    fixture(tmp_path)
    class Reference:
        def observe(self,rgb):return np.array([[11,20,2,2]])
    monkeypatch.setattr(runner,'ReferenceExtractor',Reference)
    monkeypatch.setattr(runner,'mspacman_visible_scene',lambda ram,previous:
        [SceneObject('player:0','player',(10+int(ram[0]),20,2,2))])
    result=runner.measure_frames('ms_pacman',runner.raw_history_fixture_frames(tmp_path))
    assert result['metrics']['frames']==3 and result['metrics']['per_type']['player']['eligible']==1


def test_missing_prerequisite_never_constructs_or_burns_seed(tmp_path,monkeypatch):
    directory,receipt,_=fake_paths(tmp_path,monkeypatch)
    def reject(game):raise ValueError('raw proof missing')
    monkeypatch.setattr(runner,'check',reject)
    monkeypatch.setattr(runner,'_ColdPixelEngine',lambda _:pytest.fail('constructed emulator'))
    with pytest.raises(ValueError,match='raw proof'):runner.capture('ms_pacman')
    assert not receipt.exists() and not directory.exists()


def test_constructor_failure_burns_attempt_before_construction(tmp_path,monkeypatch):
    directory,receipt,_=fake_paths(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'check',lambda game:dict(proof='checked'))
    def failed(config):
        assert json.loads(receipt.read_text())['status']=='started'
        raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(runner,'_ColdPixelEngine',failed)
    with pytest.raises(RuntimeError,match='constructor failure'):runner.capture('asteroids')
    record=json.loads(receipt.read_text())
    assert record['status']=='failed' and record['accounting_status']=='constructor_did_not_return_cost_unknown_no_retry'
    with pytest.raises(ValueError,match='earlier captures'):runner.capture('asteroids')


def test_partial_failed_act_remains_charged(tmp_path,monkeypatch):
    _,receipt,_=fake_paths(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'check',lambda game:dict(proof='checked'))
    class Engine:
        frames=132
        def set_native_wrapper(self,wrapper):pass
        def accounting(self):return dict(cold_loads=1,charged_frames=self.frames)
        def step(self,action):
            self.frames+=1
            raise RuntimeError('synthetic post-act failure')
    monkeypatch.setattr(runner,'_ColdPixelEngine',lambda _:Engine())
    with pytest.raises(RuntimeError,match='post-act'):runner.capture('asteroids')
    record=json.loads(receipt.read_text())
    assert record['accounting']['charged_frames']==133 and record['completed_frames']==0
    assert record['accounting_status']=='partial_live_ledger'


def test_score_failure_gets_a_separate_marker_and_never_changes_capture(tmp_path,monkeypatch):
    _,capture_receipt,out=fake_paths(tmp_path,monkeypatch)
    capture_receipt.write_text('{"status":"complete"}')
    original=capture_receipt.read_bytes()
    def fail(game):raise ValueError('proof changed')
    monkeypatch.setattr(runner,'check',fail)
    with pytest.raises(ValueError,match='proof changed'):runner.score('ms_pacman')
    assert capture_receipt.read_bytes()==original and not out.exists()
    assert json.loads((tmp_path/'result_score_attempt.json').read_text())['status']=='failed'
    with pytest.raises(ValueError,match='earlier reference'):runner.score('ms_pacman')


@pytest.mark.parametrize('game',['boxing','frostbite','riverraid','kangaroo'])
def test_held_out_games_remain_closed(game):
    with pytest.raises(ValueError,match='prospectively'):runner.paths(game)
