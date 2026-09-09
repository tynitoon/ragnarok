"""Prospective native barriers/failure artifacts; fixtures never open an emulator."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import bench_cached_visible_v70 as bench


def report(steps=2048):
    pairs=steps//128
    return dict(failure=None,decisions=steps,ordinary_raw_frames=steps*4,
                ordered_reference_output_sha256='synthetic-identical',
                accounting=dict(charged_frames=132+steps*4+pairs*128,emulated_frames=132+steps*4+pairs*8))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,'ROOT',tmp_path)
    monkeypatch.setattr(bench,'FOLDER',tmp_path)
    for name,filename in (('OUTPUT','native_chain_v0.json'),('WIRING','wiring_v0.json'),
                          ('WIRING_REVIEW','wiring_review_v0.json'),('BASELINE','baseline.json'),
                          ('PREFLIGHT','preflight.json')):
        monkeypatch.setattr(bench,name,tmp_path/filename)
    monkeypatch.setattr(bench,'preflight',lambda:dict(source_sha256={'synthetic':'fixed'},preflight_review_sha256='review'))
    monkeypatch.setattr(bench,'require_wiring',lambda proof:dict(wiring_result_sha256='synthetic-wiring'))
    monkeypatch.setattr(bench,'require_committed',lambda *args:None)
    bench.BASELINE.write_text(json.dumps(dict(worker_reports=[report() for _ in range(16)])))
    created=[]
    class Pool:
        def __init__(self,configs,**kwargs):
            self.steps=0;self.closed=False
            self._processes=[SimpleNamespace(is_alive=lambda:False)]
            self.output=(np.zeros((210,160,3),np.uint8),np.zeros((0,4),np.int64))
            self.configs=configs
            created.append(self)
        def step(self,actions):
            assert actions==[(self.steps//4+i)%18 for i in range(16)]
            self.steps+=1
            return [self.output]*16
        def report(self,index=0):return report(self.steps)
        def close(self):self.closed=True
    monkeypatch.setattr(bench,'CachedVisiblePool',Pool)
    return created,Pool


@pytest.mark.parametrize('mode',['wiring','benchmark'])
def test_preflight_rejection_precedes_attempt_and_all_native_constructors(tmp_path,monkeypatch,mode):
    created,_=setup(tmp_path,monkeypatch)
    def fail():raise ValueError('Independent preflight incomplete')
    monkeypatch.setattr(bench,'preflight',fail)
    monkeypatch.setattr(bench,'VisibleReferenceEngine',lambda *a,**k:pytest.fail('native constructor'))
    monkeypatch.setattr(bench,'CachedVisibleEngine',lambda *a,**k:pytest.fail('native constructor'))
    with pytest.raises(ValueError,match='preflight'):getattr(bench,mode)()
    assert not created and list(tmp_path.iterdir())==[bench.BASELINE]


def test_wiring_review_rejection_precedes_benchmark_attempt_or_workers(tmp_path,monkeypatch):
    created,_=setup(tmp_path,monkeypatch)
    def fail(proof):raise ValueError('Independent wiring review missing')
    monkeypatch.setattr(bench,'require_wiring',fail)
    with pytest.raises(ValueError,match='wiring review'):bench.benchmark()
    assert not created and not bench._attempt_path(bench.OUTPUT).exists()


def test_complete_identical_synthetic_reports_fixed_commands_and_no_retry(tmp_path,monkeypatch):
    created,_=setup(tmp_path,monkeypatch)
    result=bench.benchmark()
    assert result['engineering_equivalence_pass'] and result['worker_reports_exactly_equal']
    assert result['charged_frames']==16*report()['accounting']['charged_frames']
    assert result['agent_training_allowed'] is False and result['gate_result'] is None
    assert len(created)==1 and created[0].steps==2048 and created[0].closed
    assert [c.seed for c in created[0].configs]==list(range(18000,18016))
    assert all(c.max_charged_frames==100000 for c in created[0].configs)
    before=bench.OUTPUT.read_bytes(),bench._attempt_path(bench.OUTPUT).read_bytes()
    with pytest.raises(ValueError,match='no retry'):bench.benchmark()
    assert before==(bench.OUTPUT.read_bytes(),bench._attempt_path(bench.OUTPUT).read_bytes())
    assert len(created)==1


def test_complete_digest_mismatch_is_preserved_as_negative_equivalence(tmp_path,monkeypatch):
    created,_=setup(tmp_path,monkeypatch)
    baseline=json.loads(bench.BASELINE.read_text())
    baseline['worker_reports'][9]['ordered_reference_output_sha256']='synthetic-mismatch'
    bench.BASELINE.write_text(json.dumps(baseline))
    result=bench.benchmark()
    assert not result['engineering_equivalence_pass'] and not result['worker_reports_exactly_equal']
    assert created[0].closed and bench.OUTPUT.exists()
    attempt=json.loads(bench._attempt_path(bench.OUTPUT).read_text())
    assert attempt['status']=='complete' and not attempt['engineering_equivalence_pass']


def test_paid_worker_failure_keeps_all_available_costs_and_stops_without_retry(tmp_path,monkeypatch):
    created,Pool=setup(tmp_path,monkeypatch)
    def fail(self,actions):
        self.steps+=1
        raise ValueError('synthetic paid worker batch failure')
    monkeypatch.setattr(Pool,'step',fail)
    with pytest.raises(ValueError,match='paid worker'):bench.benchmark()
    attempt=json.loads(bench._attempt_path(bench.OUTPUT).read_text())
    assert attempt['status']=='failed' and attempt['accounting_complete'] and attempt['no_retry']
    assert len(attempt['partial_reports'])==16 and attempt['known_charged_frames']==16*136
    assert not bench.OUTPUT.exists() and len(created)==1 and created[0].steps==1 and created[0].closed
    with pytest.raises(ValueError,match='no retry'):bench.benchmark()


def test_constructor_failure_declares_unknown_cost_instead_of_zero_total(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError('synthetic partial startup')
    monkeypatch.setattr(bench,'CachedVisiblePool',fail)
    with pytest.raises(RuntimeError):bench.benchmark()
    attempt=json.loads(bench._attempt_path(bench.OUTPUT).read_text())
    assert not attempt['accounting_complete'] and attempt['unknown_cost_note']
    assert attempt['known_charged_frames']==0 and 'charged_frames' not in attempt


def test_wiring_second_constructor_failure_retains_first_constructor_cost(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    monkeypatch.setattr(bench,'VisibleReferenceEngine',lambda *a,**k:SimpleNamespace(report=lambda:report(0)))
    def fail(*args,**kwargs):raise ValueError('synthetic second native construction')
    monkeypatch.setattr(bench,'CachedVisibleEngine',fail)
    with pytest.raises(ValueError,match='second native'):bench.wiring()
    attempt=json.loads(bench._attempt_path(bench.WIRING).read_text())
    assert attempt['known_charged_frames']==132 and len(attempt['partial_reports'])==1
    assert attempt['constructor_without_return']=='cached_19176' and not attempt['accounting_complete']
    assert attempt['no_retry'] and not bench.WIRING.exists()


def test_postpaid_proof_change_keeps_reports_and_cannot_publish_success(tmp_path,monkeypatch):
    created,_=setup(tmp_path,monkeypatch)
    calls=[]
    def proof():
        calls.append(1)
        return dict(source_sha256={'synthetic':'fixed' if len(calls)==1 else 'changed'},preflight_review_sha256='review')
    monkeypatch.setattr(bench,'preflight',proof)
    with pytest.raises(ValueError,match='changed during paid'):bench.benchmark()
    attempt=json.loads(bench._attempt_path(bench.OUTPUT).read_text())
    assert attempt['accounting_complete'] and attempt['known_charged_frames']==16*report()['accounting']['charged_frames']
    assert created[0].closed and not bench.OUTPUT.exists()


def test_independent_wiring_review_must_bind_attempt_result_and_current_sources(tmp_path,monkeypatch):
    actual_require=bench.require_wiring
    setup(tmp_path,monkeypatch)
    proof=bench.preflight()
    bench.WIRING.write_text(json.dumps(dict(kind='cached_mspacman_cpu_wiring',passed=True,**proof,wiring=bench.WIRING_CASES)))
    attempt=bench._attempt_path(bench.WIRING)
    attempt.write_text(json.dumps(dict(status='complete',result_sha256=bench.sha(bench.WIRING))))
    review=dict(kind='cached_mspacman_cpu_wiring_independent_review',passed=True,
        wiring_result_sha256=bench.sha(bench.WIRING),wiring_attempt_sha256=bench.sha(attempt),**proof)
    bench.WIRING_REVIEW.write_text(json.dumps(review))
    assert actual_require(proof)['wiring_attempt_sha256']==bench.sha(attempt)
    for key in ('wiring_result_sha256','wiring_attempt_sha256','preflight_review_sha256','source_sha256'):
        changed=dict(review);changed[key]='changed'
        bench.WIRING_REVIEW.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='independent review'):actual_require(proof)
