"""Reserved-attempt guards and uncovered-category semantics, without ALE."""
import copy
import json
import pytest

from scripts import validate_mspacman_raw_v70 as validation


def scores(fruit_count=0):
    uniform = {'per_type': {k:dict(passed=True) for k in ('player','ghost','power_pill','pill')}}
    uniform['per_type']['fruit'] = dict(passed=False,positive_frames=fruit_count,
        ram_objects=fruit_count,pixel_objects=fruit_count,joint_matches=fruit_count,
        positive_presence_fraction=None if not fruit_count else 1.,box_fraction=None if not fruit_count else 1.)
    uniform['calibration_pass'] = False
    coverage = copy.deepcopy(uniform)
    coverage['per_type']['fruit'].update(passed=True,positive_frames=100,ram_objects=100,pixel_objects=100,joint_matches=100)
    coverage['calibration_pass'] = True
    return uniform,coverage


def test_absent_uniform_fruit_stays_null_and_fail():
    uniform,coverage = scores()
    before = copy.deepcopy(uniform)
    verdict = validation.pair_verdict(uniform,coverage)
    assert verdict['pair_pass'] and verdict['uniform_fruit_strictly_absent']
    assert uniform == before and not verdict['uniform_calibration_pass']


@pytest.mark.parametrize('field',['positive_frames','ram_objects','pixel_objects','joint_matches'])
def test_any_nonzero_uniform_fruit_requires_full_pass(field):
    uniform,coverage = scores()
    uniform['per_type']['fruit'][field] = 1
    verdict = validation.pair_verdict(uniform,coverage)
    assert not verdict['pair_pass'] and not verdict['uniform_fruit_strictly_absent']


@pytest.mark.parametrize('side,kind',[(0,'ghost'),(1,'fruit'),(1,'pill')])
def test_other_category_failures_are_never_exempted(side,kind):
    pair = scores()
    pair[side]['per_type'][kind]['passed'] = False
    assert not validation.pair_verdict(*pair)['pair_pass']


def setup_run(monkeypatch,tmp_path):
    monkeypatch.setattr(validation,'BASE',tmp_path)
    monkeypatch.setattr(validation,'verify_seal',lambda:dict(_file_sha256='seal'))
    monkeypatch.setattr(validation,'git',lambda *args:'committed')


@pytest.mark.parametrize('occupied',[
    'validation_uniform_v3','validation_coverage_v3','validation_uniform_v3_attempt.json',
    'validation_coverage_v3_attempt.json','uniform_reserved_visible_v3.json',
    'coverage_reserved_visible_v3.json','raw_reserved_pair_v3.json'])
def test_existing_any_pair_artifact_refuses_before_emulation(monkeypatch,tmp_path,occupied):
    setup_run(monkeypatch,tmp_path)
    (tmp_path/occupied).write_text('preserve',encoding='utf-8')
    monkeypatch.setattr(validation,'capture_uniform',lambda *a,**kw:pytest.fail('No emulator may be opened'))
    with pytest.raises(ValueError,match='already attempted'):
        validation.run()
    assert (tmp_path/occupied).read_text(encoding='utf-8') == 'preserve'
    assert len(list(tmp_path.iterdir())) == 1


def test_attempt_is_durable_before_construction_and_failure_is_not_retried(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    calls = []
    def failing_constructor(destination,**kwargs):
        record=json.loads((tmp_path/'validation_uniform_v3_attempt.json').read_text())
        assert record['status']=='started' and record['seed']==17412 and record['seal_sha256']=='seal'
        calls.append(kwargs)
        raise RuntimeError('emulator construction failed')
    monkeypatch.setattr(validation,'capture_uniform',failing_constructor)
    with pytest.raises(RuntimeError,match='construction failed'):
        validation.run()
    record=json.loads((tmp_path/'validation_uniform_v3_attempt.json').read_text())
    assert record['status']=='failed' and 'construction failed' in record['error']
    with pytest.raises(ValueError,match='already attempted'):
        validation.run()
    assert len(calls)==1 and not (tmp_path/'validation_coverage_v3_attempt.json').exists()


def test_uncommitted_seal_refuses_before_emulation(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    def bad_seal():
        raise ValueError('Commit every source and seal before capture')
    monkeypatch.setattr(validation,'verify_seal',bad_seal)
    monkeypatch.setattr(validation,'capture_uniform',lambda *a,**kw:pytest.fail('No emulator may be opened'))
    with pytest.raises(ValueError,match='Commit every source'):
        validation.run()
    assert not list(tmp_path.iterdir())
