"""Negative development, missing review, provenance mutations and seal chronology."""
import json
import pytest
from scripts import temporal_validation_seal_v70 as seal


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(seal,'ROOT',tmp_path)
    monkeypatch.setattr(seal,'runtime',lambda:{'synthetic':'no emulator'})
    monkeypatch.setattr(seal,'require_committed',lambda path,commit='HEAD':None)
    monkeypatch.setattr(seal,'git',lambda *args:'frozen-commit')
    monkeypatch.setattr(seal.subprocess,'check_call',lambda args:0)
    monkeypatch.setattr(seal,'triple_pooled_fixture_frames',lambda path:iter([None]*8192))
    def write(path,value):
        p=tmp_path/path;p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(value),encoding='utf-8')
    (tmp_path/'model.py').write_text('version=1',encoding='utf-8')
    write('protocol.json',{'seed':17422})
    hashes={p:seal.sha(tmp_path/p) for p in ('model.py','protocol.json')}
    runtime=seal.runtime();raw='raw-proof'
    account=dict(charged_frames=33416,cold_loads=5)
    manifest=dict(game='ms_pacman',seed=17421,partition='development',decisions=8192,
        source_sha256={'model.py':hashes['model.py']},source_commit='capture-commit',runtime=runtime,accounting=account)
    write('fixture/manifest.json',manifest);mh=seal.sha(tmp_path/'fixture/manifest.json')
    stats=dict(frames=8192,presence_count_agreement=8192,positive_frames=8192,positive_count_agreement=8192,
        ram_objects=8192,pixel_objects=8192,joint_matches=8192,boxes_within1=8192,boxes_exact=8192,
        presence_fraction=1.,positive_presence_fraction=1.,box_fraction=1.,passed=True)
    result=dict(seed=17421,decoder_revision='test-v1',source_sha256={'model.py':hashes['model.py']},
        raw_calibration_proof_sha256=raw,fixture_manifest_sha256=mh,calibration_pass=True,per_type={'ghost':stats})
    write('result.json',result);rh=seal.sha(tmp_path/'result.json')
    write('capture.json',dict(status='complete',seed=17421,source_sha256=manifest['source_sha256'],
        source_commit=manifest['source_commit'],runtime=runtime,accounting=account,
        raw_calibration_proof_sha256=raw,manifest_sha256=mh))
    write('score.json',dict(status='complete',seed=17421,calibration_pass=True,result_sha256=rh,fixture_manifest_sha256=mh))
    write('review.json',dict(passed=True,development_calibration_pass_confirmed=True,result_sha256=rh))
    write('preflight.json',dict(passed=True,source_sha256=hashes,runtime=runtime,development_result_sha256=rh))
    spec=dict(game='ms_pacman',development_seed=17421,reserved_seed=17422,kinds=('ghost',),revision='test-v1',
        protocol='protocol.json',source_paths=('model.py','protocol.json'),development_sources=('model.py',),
        capture_sources=('model.py',),development_fixture='fixture',development_result='result.json',
        development_capture_attempt='capture.json',development_score_attempt='score.json',
        development_review='review.json',preflight_review='preflight.json',seal='seal.json',
        metadata_paths=('fixture/manifest.json','result.json','capture.json','score.json','review.json','preflight.json'))
    return spec,lambda:raw,write


def test_complete_reviewed_development_can_be_sealed_once_and_verified(tmp_path,monkeypatch):
    spec,raw,_=fixture(tmp_path,monkeypatch)
    created=seal.create_seal(spec,raw)
    assert created['source_commit']=='frozen-commit' and created['seed']==17422
    got=seal.verify_seal(spec,raw)
    assert got['seal_sha256']==seal.sha(tmp_path/'seal.json') and got['raw_calibration_proof_sha256']=='raw-proof'
    with pytest.raises(ValueError,match='existing validation seal'):seal.create_seal(spec,raw)


@pytest.mark.parametrize('file,key,value',[
    ('capture.json','seed',17422),('capture.json','status','failed'),
    ('capture.json','source_commit','unrelated'),('capture.json','accounting',{}),
    ('score.json','fixture_manifest_sha256','changed'),('score.json','status','failed'),
    ('score.json','result_sha256','changed'),('review.json','passed',False),
    ('review.json','result_sha256','changed'),('preflight.json','source_sha256',{}),
    ('preflight.json','runtime',{}),('result.json','calibration_pass',False),
])
def test_incomplete_or_inconsistent_proof_never_creates_a_seal(tmp_path,monkeypatch,file,key,value):
    spec,raw,write=fixture(tmp_path,monkeypatch)
    data=json.loads((tmp_path/file).read_text());data[key]=value;write(file,data)
    with pytest.raises(ValueError):seal.create_seal(spec,raw)
    assert not (tmp_path/'seal.json').exists()


@pytest.mark.parametrize('mutate',[
    lambda s:s.update(joint_matches=49),
    lambda s:s.update(positive_frames=0),
    lambda s:s.update(positive_count_agreement=8100,positive_presence_fraction=8100/8192),
    lambda s:s.update(boxes_within1=8100,boxes_exact=8100,box_fraction=8100/8192),
    lambda s:s.update(box_fraction=.999),
    lambda s:s.update(positive_frames=9000,positive_count_agreement=9000),
    lambda s:s.update(positive_frames=8100,positive_count_agreement=8099,positive_presence_fraction=8099/8100),
    lambda s:s.update(ram_objects=-1),
    lambda s:s.update(boxes_exact=8192.0),
])
def test_null_sparse_failed_or_falsified_denominators_cannot_pass(tmp_path,monkeypatch,mutate):
    fixture(tmp_path,monkeypatch)
    report=json.loads((tmp_path/'result.json').read_text());mutate(report['per_type']['ghost'])
    with pytest.raises(ValueError):seal.check_statistics(report,('ghost',))


def test_fixture_reader_failure_prevents_seal(tmp_path,monkeypatch):
    spec,raw,_=fixture(tmp_path,monkeypatch)
    def reject(path):raise ValueError('synthetic uncharged window')
    monkeypatch.setattr(seal,'triple_pooled_fixture_frames',reject)
    with pytest.raises(ValueError,match='uncharged'):seal.create_seal(spec,raw)


def test_old_seal_cannot_follow_mutated_runtime(tmp_path,monkeypatch):
    spec,raw,_=fixture(tmp_path,monkeypatch);seal.create_seal(spec,raw)
    monkeypatch.setattr(seal,'runtime',lambda:{'synthetic':'changed'})
    with pytest.raises(ValueError):seal.verify_seal(spec,raw)


def test_uncommitted_source_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(seal,'ROOT',tmp_path)
    monkeypatch.setattr(seal,'git',lambda *args:'working' if args[0]=='hash-object' else 'committed')
    with pytest.raises(ValueError,match='Committed proof differs'):seal.require_committed('model.py')


def test_metadata_must_exist_unchanged_at_the_announced_source_commit(tmp_path,monkeypatch):
    spec,raw,_=fixture(tmp_path,monkeypatch);seal.create_seal(spec,raw)
    def committed(path,commit='HEAD'):
        if path=='review.json' and commit=='frozen-commit':
            raise ValueError('Review only committed after the announced source commit')
    monkeypatch.setattr(seal,'require_committed',committed)
    with pytest.raises(ValueError,match='Review only committed after'):
        seal.verify_seal(spec,raw)
