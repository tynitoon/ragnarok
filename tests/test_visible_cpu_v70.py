"""Prospective CPU proof barriers and paid-failure evidence; no ROM is opened."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import bench_visible_reference_v70 as bench
from scripts import check_visible_cpu_v70 as check
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject


def _paths(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,'ROOT',tmp_path)
    monkeypatch.setattr(bench,'base',lambda game:tmp_path)
    monkeypatch.setattr(bench,'preflight',lambda game:{'proof':'synthetic'})
    monkeypatch.setattr(bench,'require_geometry',lambda game,proof:'geometry')
    monkeypatch.setattr(bench,'require_committed',lambda *args:None)
    (tmp_path/'cpu_wiring_v0.json').write_text(json.dumps(dict(
        kind='visible_cpu_wiring_check',passed=True,proof='synthetic',geometry_sha256='geometry')))


@pytest.mark.parametrize('game',['ms_pacman','asteroids'])
def test_geometry_compares_both_actual_source_pairs_in_order(monkeypatch,game):
    calls=[]
    def scene(now,before):
        calls.append((int(now[0]),int(before[0])))
        return [SceneObject('player:0','player',(10+int(now[0]),20,2,2))]
    monkeypatch.setattr(check,'mspacman_visible_scene' if game=='ms_pacman' else 'asteroids_visible_scene',scene)
    monkeypatch.setattr(check,'pooled_mspacman_ram_objects' if game=='ms_pacman' else 'pooled_asteroids_ram_objects',
        lambda older,previous,current:[ObjectLabel('player',(11,20,3,2))])
    row={k:np.full(128,v,dtype=np.uint8) for v,k in enumerate(
        ('ram_before_previous','ram_previous','ram_current'))}
    assert check._same_window(row,game) and calls==[(1,0),(2,1)]
    row['ram_current'][:]=9
    assert not check._same_window(row,game)


def test_missing_final_reference_proof_cannot_open_cpu_seeds(monkeypatch):
    monkeypatch.setattr(bench,'REFERENCE_REVIEW_SHA256',None)
    monkeypatch.setattr(bench.reference,'check',lambda game:pytest.fail('went beyond missing proof'))
    with pytest.raises(ValueError,match='independently reviewed'):bench.require_reference('asteroids')


def test_missing_fast_source_review_cannot_open_cpu_seeds(monkeypatch):
    monkeypatch.setattr(bench,'FAST_REVIEW_SHA256',None)
    monkeypatch.setattr(bench.fast_check,'hashes',lambda:pytest.fail('went beyond missing review'))
    with pytest.raises(ValueError,match='independent fast source'):bench.require_fast_equivalence()


@pytest.mark.parametrize('defect',['missing_window','missing_frame_zero','missing_archive_review','none'])
def test_fast_source_proof_requires_every_archived_pair_and_reviewed_file(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(bench,'ROOT',tmp_path)
    monkeypatch.setattr(bench,'require_committed',lambda p:None)
    archive=tmp_path/'archive.npz';archive.write_bytes(b'synthetic evidence, not an emulator fixture')
    proof=dict(source_sha256={'source.py':'synthetic'},evidence_sha256={'archive.npz':bench.sha(archive)})
    monkeypatch.setattr(bench.fast_check,'hashes',lambda:proof)
    out=tmp_path/'equivalence.json';attempt=tmp_path/'equivalence_attempt.json';attempt.write_text('{}')
    monkeypatch.setattr(bench.fast_check,'OUTPUT',out)
    collections={name:dict(windows=8192,source_pairs=pairs) for name,pairs in
        [('development',16384),('reserved',16384),('fresh_raw',8197)]}
    if defect=='missing_window':collections['reserved']['windows']=8191
    if defect=='missing_frame_zero':collections['fresh_raw']['source_pairs']=8192
    out.write_text(json.dumps(dict(kind='offline_ordered_mspacman_source_equivalence',passed=True,
        **proof,collections=collections)))
    evidence={p.name:bench.sha(p) for p in (archive,out,attempt)}
    if defect=='missing_archive_review':evidence.pop(archive.name)
    review=tmp_path/'review.json'
    review.write_text(json.dumps(dict(passed=True,source_sha256=proof['source_sha256'],
        result_sha256=bench.sha(out),evidence_sha256=evidence)))
    monkeypatch.setattr(bench,'FAST_REVIEW',review.name)
    monkeypatch.setattr(bench,'FAST_REVIEW_SHA256',bench.sha(review))
    if defect=='none':assert bench.require_fast_equivalence()==dict(result_sha256=bench.sha(out),review_sha256=bench.sha(review))
    else:
        with pytest.raises(ValueError,match='equivalence|evidence'):bench.require_fast_equivalence()


@pytest.mark.parametrize('mode',['geometry','wiring','benchmark'])
def test_preflight_failure_does_not_burn_attempt_or_construct(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(bench,'base',lambda game:tmp_path)
    def rejected(game):raise ValueError('review incomplete')
    monkeypatch.setattr(bench,'preflight',rejected)
    monkeypatch.setattr(check,'VisibleReferenceEngine',lambda c:pytest.fail('opened emulator'))
    monkeypatch.setattr(bench,'VisibleReferencePool',lambda *a,**k:pytest.fail('opened worker'))
    with pytest.raises(ValueError,match='review incomplete'):
        if mode=='benchmark':bench.benchmark('asteroids','numpy')
        else:getattr(check,mode)('asteroids')
    assert list(tmp_path.iterdir())==[]


def test_wiring_constructor_failure_keeps_prior_engine_cost_and_never_retries(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch);calls=[]
    # The output marker used by benchmark setup must not pre-exist for wiring.
    monkeypatch.setattr(bench,'base',lambda game:tmp_path/'new')
    (tmp_path/'new').mkdir()
    def create_at_new(config):
        calls.append(config.seed)
        assert json.loads((tmp_path/'new/cpu_wiring_v0_attempt.json').read_text())['status']=='started'
        if len(calls)==2:raise RuntimeError('synthetic second constructor failure')
        return SimpleNamespace(report=lambda:dict(accounting=dict(charged_frames=132)))
    monkeypatch.setattr(check,'VisibleReferenceEngine',create_at_new)
    with pytest.raises(RuntimeError,match='second constructor'):check.wiring('asteroids')
    record=json.loads((tmp_path/'new/cpu_wiring_v0_attempt.json').read_text())
    assert record['known_charged_frames']==132 and record['constructor_without_return']=='oracle_removed'
    assert record['accounting_complete'] is False and calls==[18176,18176]
    with pytest.raises(ValueError,match='no retry'):check.wiring('asteroids')
    assert len(calls)==2


def test_benchmark_postclose_failure_keeps_all_completed_costs(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch);checks=[];closed=[]
    def proof(game):
        checks.append(game)
        if len(checks)>1:raise ValueError('source changed after paid work')
        return {'proof':'synthetic'}
    monkeypatch.setattr(bench,'preflight',proof)
    class Pool:
        def __init__(self,configs,**kwargs):
            assert json.loads((tmp_path/'cpu_chain_numpy_v0_attempt.json').read_text())['status']=='started'
            assert [c.seed for c in configs]==list(range(18000,18016))
            self.steps=0
        def step(self,actions):
            assert actions==[(self.steps//4+i)%18 for i in range(16)]
            self.steps+=1
            return [(np.zeros((210,160,3),dtype=np.uint8),np.zeros((0,4),dtype=np.int64))]*16
        def report(self,i):return dict(decisions=self.steps,failure=None,accounting=dict(charged_frames=9000+i))
        def close(self):closed.append(True)
    monkeypatch.setattr(bench,'VisibleReferencePool',Pool)
    with pytest.raises(ValueError,match='after paid work'):bench.benchmark('asteroids','numpy')
    record=json.loads((tmp_path/'cpu_chain_numpy_v0_attempt.json').read_text())
    assert record['status']=='failed' and record['accounting_status']=='collected_worker_reports'
    assert len(record['worker_reports'])==16 and record['charged_frames']==sum(range(9000,9016))
    assert closed==[True]
    with pytest.raises(ValueError,match='no retry'):bench.benchmark('asteroids','numpy')


def test_benchmark_midbatch_failure_preserves_each_workers_paid_report(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch);closed=[]
    class Pool:
        def __init__(self,*args,**kwargs):pass
        def step(self,actions):raise ValueError('one decoder failed after frame one')
        def report(self,i):return dict(accounting=dict(charged_frames=133 if i==0 else 136),
            failure={'error':'decoder'} if i==0 else None)
        def close(self):closed.append(True)
    monkeypatch.setattr(bench,'VisibleReferencePool',Pool)
    with pytest.raises(ValueError,match='decoder failed'):bench.benchmark('asteroids','numpy')
    record=json.loads((tmp_path/'cpu_chain_numpy_v0_attempt.json').read_text())
    assert record['accounting_status']=='partial_reports_when_transport_available'
    assert [r['report']['accounting']['charged_frames'] for r in record['partial_reports']]==[133]+[136]*15
    assert closed==[True]


def test_native_cannot_construct_without_prior_numpy_result(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch)
    monkeypatch.setattr(bench,'VisibleReferencePool',lambda *a,**k:pytest.fail('opened worker'))
    with pytest.raises(FileNotFoundError):bench.benchmark('asteroids','native')
    assert not (tmp_path/'cpu_chain_native_v0_attempt.json').exists()


def test_geometry_creates_missing_output_directory_only_after_valid_preflight(tmp_path,monkeypatch):
    folder=tmp_path/'missing'
    monkeypatch.setattr(bench,'base',lambda game:folder)
    monkeypatch.setattr(bench,'preflight',lambda game:{'proof':'synthetic'})
    def broken_fixture(directory):
        assert json.loads((folder/'cpu_geometry_v0_attempt.json').read_text())['status']=='started'
        raise ValueError('synthetic bad fixture')
    monkeypatch.setattr(check,'triple_pooled_fixture_frames',broken_fixture)
    with pytest.raises(ValueError,match='bad fixture'):check.geometry('asteroids')
    record=json.loads((folder/'cpu_geometry_v0_attempt.json').read_text())
    assert record['status']=='failed' and record['new_charged_frames']==0


def test_later_report_transport_failure_never_loses_already_received_costs(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch);seen=[]
    class Pool:
        def __init__(self,*args,**kwargs):pass
        def step(self,actions):
            return [(np.zeros((210,160,3),dtype=np.uint8),np.zeros((0,4),dtype=np.int64))]*16
        def report(self,i):
            seen.append(i)
            if i>=2:raise RuntimeError('transport closed')
            return dict(decisions=2048,failure=None,accounting=dict(charged_frames=9100+i))
        def close(self):pass
    monkeypatch.setattr(bench,'VisibleReferencePool',Pool)
    with pytest.raises(RuntimeError,match='transport closed'):bench.benchmark('asteroids','numpy')
    record=json.loads((tmp_path/'cpu_chain_numpy_v0_attempt.json').read_text())
    assert record['known_charged_frames']==18201 and record['accounting_complete'] is False
    assert seen==list(range(16))
    assert [r['report']['accounting']['charged_frames'] for r in record['partial_reports'][:2]]==[9100,9101]
    assert all('accounting_error' in r for r in record['partial_reports'][2:])


def test_requesting_asteroids_still_requires_the_full_mspacman_reference(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch)
    monkeypatch.setattr(bench,'REFERENCE_REVIEW','reference_review.json')
    proof=dict(source_sha256={'source':'checked'},temporal_proof_sha256='temporal')
    monkeypatch.setattr(bench.reference,'check',lambda game:proof)
    evidence={};result_hashes={};paths={}
    for game in bench.reference.REGISTRY:
        directory=tmp_path/game;directory.mkdir()
        manifest=directory/'manifest.json';manifest.write_text('{}')
        attempt=directory/'capture.json';attempt.write_text('{}')
        out=directory/'result.json'
        score=directory/'result_score_attempt.json';score.write_text('{}')
        out.write_text(json.dumps(dict(kind='fresh_visible_raw_reference_diagnostic',**proof,
            fixture_manifest_sha256=bench.sha(manifest),metrics=dict(frames=8192))))
        result_hashes[game]=bench.sha(out);paths[game]=(directory,attempt,out)
        evidence.update({p.relative_to(tmp_path).as_posix():bench.sha(p) for p in (manifest,attempt,out,score)})
    review=tmp_path/'reference_review.json'
    review.write_text(json.dumps(dict(passed=True,source_sha256=proof['source_sha256'],
        result_sha256=result_hashes,evidence_sha256=evidence)))
    monkeypatch.setattr(bench,'REFERENCE_REVIEW_SHA256',bench.sha(review))
    monkeypatch.setattr(bench.reference,'paths',lambda game:paths[game])
    seen=[]
    def read(directory):
        seen.append(directory.name)
        return iter(range(8191 if directory.name=='ms_pacman' else 8192))
    monkeypatch.setattr(bench,'raw_history_fixture_frames',read)
    with pytest.raises(ValueError,match='Every fresh raw'):bench.require_reference('asteroids')
    assert seen==['ms_pacman']
    review.write_text(json.dumps(dict(passed=True,result_sha256={'asteroids':'only_one'})))
    monkeypatch.setattr(bench,'REFERENCE_REVIEW_SHA256',bench.sha(review))
    with pytest.raises(ValueError,match='both completed'):bench.require_reference('asteroids')


def test_native_build_requires_committed_provenance_and_exact_binary(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,'ROOT',tmp_path);seen=[]
    source=tmp_path/'ragnarok/native/reference_update_v70.c';source.parent.mkdir(parents=True)
    source.write_bytes(b'not executed C source')
    build=tmp_path/'build/build.json';build.parent.mkdir()
    dll=build.parent/'reference_update_v70.dll';dll.write_bytes(b'not executed binary')
    build.write_text(json.dumps(dict(source_sha256=bench.sha(source),dll_sha256=bench.sha(dll))))
    monkeypatch.setattr(bench,'NATIVE_BUILD',build)
    monkeypatch.setattr(bench,'require_committed',lambda p:seen.append(p))
    assert bench.require_native_build()==bench.sha(build)
    assert seen==['build/build.json','build/reference_update_v70.dll']
    dll.write_bytes(b'changed binary')
    with pytest.raises(ValueError,match='provenance changed'):bench.require_native_build()


def _complete_pool(monkeypatch,close=lambda:None,digest='unchanged'):
    class Pool:
        def __init__(self,*args,**kwargs):pass
        def step(self,actions):
            return [(np.zeros((210,160,3),dtype=np.uint8),np.zeros((0,4),dtype=np.int64))]*16
        def report(self,i):return dict(decisions=2048,failure=None,ordinary_raw_frames=8192,
            accounting=dict(charged_frames=9000,emulated_frames=8400),ordered_reference_output_sha256=digest)
        def close(self):close()
    monkeypatch.setattr(bench,'VisibleReferencePool',Pool)
    return Pool().report(0)


def test_changed_wiring_after_paid_work_is_not_bound_as_a_valid_new_proof(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch)
    _complete_pool(monkeypatch,close=lambda:(tmp_path/'cpu_wiring_v0.json').write_text('{"passed":false}'))
    with pytest.raises(ValueError,match='wiring/geometry evidence changed'):bench.benchmark('asteroids','numpy')
    record=json.loads((tmp_path/'cpu_chain_numpy_v0_attempt.json').read_text())
    assert record['charged_frames']==144000 and record['status']=='failed'
    assert not (tmp_path/'cpu_chain_numpy_v0.json').exists()


def test_native_output_mismatch_is_retained_as_negative_equivalence(tmp_path,monkeypatch):
    _paths(tmp_path,monkeypatch)
    current=_complete_pool(monkeypatch,digest='native_changed')
    old=dict(current,ordered_reference_output_sha256='numpy_original')
    monkeypatch.setattr(bench,'_numpy_pair',lambda *args:({'worker_reports':[old]*16},'paired_sha'))
    result=bench.benchmark('asteroids','native')
    assert result['engineering_equivalence_pass'] is False and result['worker_reports_exactly_equal'] is False
    assert result['charged_frames']==144000 and result['paired_numpy_result_sha256']=='paired_sha'
    assert (tmp_path/'cpu_chain_native_v0.json').exists()
    with pytest.raises(ValueError,match='no retry'):bench.benchmark('asteroids','native')
