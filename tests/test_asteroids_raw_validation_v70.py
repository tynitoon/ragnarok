"""Single-attempt, immutable-evidence and true-seal guards without an emulator."""
import json
import pytest
from scripts import validate_asteroids_raw_v70 as validation


def setup_run(monkeypatch,tmp_path):
    monkeypatch.setattr(validation,'BASE',tmp_path)
    monkeypatch.setattr(validation,'verify_seal',lambda:dict(_file_sha256='seal'))
    monkeypatch.setattr(validation,'git',lambda *args:'committed')


@pytest.mark.parametrize('occupied',[validation.DESTINATION,validation.ATTEMPT,validation.RESULT])
def test_any_existing_artifact_prevents_recollection(monkeypatch,tmp_path,occupied):
    setup_run(monkeypatch,tmp_path)
    (tmp_path/occupied).write_text('preserve')
    monkeypatch.setattr(validation,'capture',lambda *a,**kw:pytest.fail('No emulator allowed'))
    with pytest.raises(ValueError,match='already attempted'):validation.run()
    assert len(list(tmp_path.iterdir()))==1 and (tmp_path/occupied).read_text()=='preserve'


def test_failed_constructor_has_durable_attempt_and_is_never_retried(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    calls=[]
    def failure(destination,**kwargs):
        record=json.loads((tmp_path/validation.ATTEMPT).read_text())
        assert record['status']=='started' and record['seed']==17512
        calls.append(kwargs)
        raise RuntimeError('construction failure')
    monkeypatch.setattr(validation,'capture',failure)
    with pytest.raises(RuntimeError,match='construction failure'):validation.run()
    assert json.loads((tmp_path/validation.ATTEMPT).read_text())['status']=='failed'
    with pytest.raises(ValueError,match='already attempted'):validation.run()
    assert len(calls)==1


def test_scoring_failure_retains_capture_and_records_failure(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    def fake_capture(destination,**kwargs):
        destination.mkdir()
        data=dict(accounting=dict(charged_frames=8324,cold_loads=1,restores=0,experiment_frames=0))
        (destination/'manifest.json').write_text(json.dumps(data))
        (destination/'raw-witness').write_bytes(b'captured')
        return data
    def failing_score(directory):
        record=json.loads((tmp_path/validation.ATTEMPT).read_text())
        assert record['status']=='scoring' and record['manifest_sha256']==validation.sha(directory/'manifest.json')
        raise ValueError('unknown decoder state')
    from scripts import calibrate_asteroids_reserved_v70 as scorer
    monkeypatch.setattr(validation,'capture',fake_capture)
    monkeypatch.setattr(scorer,'calibrate',failing_score)
    with pytest.raises(ValueError,match='unknown decoder state'):validation.run()
    record=json.loads((tmp_path/validation.ATTEMPT).read_text())
    assert record['status']=='failed' and 'unknown decoder state' in record['error']
    assert (tmp_path/validation.DESTINATION/'raw-witness').read_bytes()==b'captured'
    assert not (tmp_path/validation.RESULT).exists()
    with pytest.raises(ValueError,match='already attempted'):validation.run()


@pytest.mark.parametrize('mutation',['none','source','metadata','runtime','seed','head','source_commit','seal_blob','ancestor'])
def test_real_verify_seal_checks_each_binding(monkeypatch,tmp_path,mutation):
    monkeypatch.setattr(validation,'ROOT',tmp_path)
    monkeypatch.setattr(validation,'SEAL','seal.json')
    monkeypatch.setattr(validation,'SOURCE_PATHS',('code.py',))
    monkeypatch.setattr(validation,'METADATA_PATHS',('proof.json',))
    (tmp_path/'code.py').write_text('source')
    (tmp_path/'proof.json').write_text('proof')
    expected_runtime=dict(python='test',versions={},ale_binary_sha256='binary')
    monkeypatch.setattr(validation,'runtime',lambda:expected_runtime)
    seal=dict(source_commit='frozen',source_sha256={'code.py':validation.sha(tmp_path/'code.py')},
              metadata_sha256={'proof.json':validation.sha(tmp_path/'proof.json')},runtime=expected_runtime,seed=17512)
    if mutation=='runtime':seal['runtime']={'wrong':True}
    if mutation=='seed':seal['seed']=17511
    (tmp_path/'seal.json').write_text(json.dumps(seal))
    if mutation=='source':(tmp_path/'code.py').write_text('changed')
    if mutation=='metadata':(tmp_path/'proof.json').write_text('changed')
    def fake_git(*args):
        if args[0]=='hash-object':return 'blob:'+args[1][7:]
        ref,path=args[1].split(':',1)
        if mutation=='head' and ref=='HEAD' and path=='code.py':return 'different'
        if mutation=='source_commit' and ref=='frozen':return 'different'
        if mutation=='seal_blob' and path=='seal.json':return 'different'
        return 'blob:'+path
    monkeypatch.setattr(validation,'git',fake_git)
    def ancestry(argv):
        if mutation=='ancestor':raise validation.subprocess.CalledProcessError(1,argv)
        return 0
    monkeypatch.setattr(validation.subprocess,'check_call',ancestry)
    if mutation=='none':
        assert validation.verify_seal()['_file_sha256']==validation.sha(tmp_path/'seal.json')
    else:
        with pytest.raises((ValueError,validation.subprocess.CalledProcessError)):
            validation.verify_seal()
