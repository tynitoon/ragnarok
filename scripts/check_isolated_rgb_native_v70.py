"""One prospective fidelity check of the explicit-interpreter RGB transport.

The old native result remains immutable. Same commands/seeds are deliberately
replayed as an engineering comparison, not an independent learning population.
This parent has Torch installed but imports no model, Torch or ALE runtime.
"""
from collections import Counter
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.isolated_rgb_pool_v70 import IsolatedRGBPool,IsolatedWorkerError
from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='docs/ARC4_ISOLATED_RGB_NATIVE_PROTOCOL.json'
BASE_PROTOCOL='docs/ARC4_OWNED_ORDINARY_NATIVE_PROTOCOL.json'
BASE_RESULT='craft_v6_out/v70_owned_ordinary_native/wiring_v0.json'
FOLDER=ROOT/'craft_v6_out/v70_isolated_rgb_native'
PREFLIGHT=FOLDER/'preflight_v0.json'
ATTEMPT=FOLDER/'wiring_v0_attempt.json'
OUTPUT=FOLDER/'wiring_v0.json'
REVIEWS={
    'craft_v6_out/v70_isolated_rgb_transport/source_review_v0.json':
        '97a3fc7d172d41b0dcb861d12872396bff17fecdaa62ebe1fade361d675a2414',
    'craft_v6_out/v70_owned_ordinary_native/wiring_review_v0.json':
        '8b4105ce479e7d990293a89ce8ea3789506600729f3e1f721dbf59653f04a668'}
NEW_SOURCES=(PROTOCOL,'docs/ARC4_ISOLATED_RGB_NATIVE_DRAFT.md',
    'scripts/check_isolated_rgb_native_v70.py','tests/test_isolated_rgb_native_protocol_v70.py')
SPEC=dict(case_names=['pong','breakout','freeway','seaquest','ms_pacman','asteroids',
    'pong_prefix_1','pong_prefix_2','pong_prefix_3'],max_constructions=9,max_charge_per_engine=2000,
    max_total_charge=18000,expected_success_charge=7454,ordinary_decisions=1266,branch_pairs=6,
    selected_indices=[0,1,2,3,126,127,128,129],peer_timeout_seconds=30,
    max_work_seconds=300,optimizer_updates=0,model_inferences=0)
CHILD=dict(executable=r'F:\dev\ragnarok\.venv-arc4\Scripts\python.exe',
    python='3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]',
    numpy='2.4.4',ale_py='0.12.1',
    ale_binary_sha256='b7e29d7c2662be5a7d49d9e7151637a7842547ecc149b2b2c6771b9e02159d7c',
    numerical_threads={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
    cuda_visible_devices='')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def require_committed(path):
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('hash-object',f'--path={path}',str(ROOT/path))!=git('rev-parse',f'HEAD:{path}'):
        raise ValueError('Current source/proof differs from its committed version')


def runtime():
    return dict(parent_executable=str(Path(sys.executable).resolve()),python=sys.version,
        numpy=np.__version__,torch=importlib.metadata.version('torch'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        numerical_threads={k:os.environ.get(k) for k in CHILD['numerical_threads']})


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'));current=runtime()
    if (protocol.get('kind')!='prospective_isolated_rgb_native_v0' or protocol.get('spec')!=SPEC
            or protocol.get('child_runtime')!=CHILD or protocol.get('parent_runtime')!=current
            or protocol.get('agent_training_allowed') is not False or protocol.get('gate_result') is not None
            or current['cuda_visible_devices']!='' or any(v!='1' for v in current['numerical_threads'].values())):
        raise ValueError('Exact prospective protocol and CPU runtime required')
    sources={};evidence={}
    for path,expected in REVIEWS.items():
        if sha(ROOT/path)!=expected:raise ValueError('Pinned independent prerequisite changed')
        review=json.loads((ROOT/path).read_text(encoding='utf-8'))
        if review.get('passed') is not True:raise ValueError('Successful independent prerequisite required')
        evidence[path]=expected
        for field,destination in (('source_sha256',sources),('evidence_sha256',evidence)):
            for p,h in review[field].items():
                if sha(ROOT/p)!=h or p in destination and destination[p]!=h:
                    raise ValueError('Reviewed source/evidence changed or conflicts')
                destination[p]=h
    base=json.loads((ROOT/BASE_RESULT).read_text(encoding='utf-8'))
    plan=json.loads((ROOT/BASE_PROTOCOL).read_text(encoding='utf-8'))
    if (evidence.get(BASE_RESULT)!=sha(ROOT/BASE_RESULT) or sources.get(BASE_PROTOCOL)!=sha(ROOT/BASE_PROTOCOL)
            or not base['passed'] or not base['accounting_complete']
            or [c['name'] for c in base['cases']]!=SPEC['case_names']
            or [c['name'] for c in plan['cases']]!=SPEC['case_names']
            or sum(c['accounting']['charged_frames'] for c in base['cases'])!=SPEC['expected_success_charge']):
        raise ValueError('Complete immutable baseline and its exact plan required')
    for case in base['cases']:
        p=case['selected_trace_archive'];h=case['selected_trace_sha256']
        if sha(ROOT/p)!=h or evidence.get(p)!=h:raise ValueError('Exact baseline archive required')
    for p in NEW_SOURCES:sources[p]=sha(ROOT/p)
    for p in {*sources,*evidence}:require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=current)


def preflight():
    proof=proofs();review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='isolated_rgb_native_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before native initialization')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


def trace_digest(digest,trace):
    # Byte-for-byte encoding from the sealed ordinary-native witness runner.
    for field in ('start_pair','raw','pooled','endpoint'):
        value=getattr(trace,field)
        digest.update(json.dumps([field,list(value.shape),str(value.dtype)],separators=(',',':')).encode()+b'\n')
        digest.update(value.tobytes(order='C'))


def same_array(actual,expected):
    if actual.dtype!=expected.dtype or actual.shape!=expected.shape or not np.array_equal(actual,expected):
        raise ValueError('Explicit-interpreter RGB differs from immutable native baseline')


class Reports:
    """Keep completed and failing case ledgers, with explicit unknown costs."""
    def __init__(self):self.started=[];self.known={};self.errors={}
    def begin(self,name):self.started.append(name)
    def remember(self,name,report):
        old=self.known.get(name);account=report.get('accounting')
        previous=old.get('accounting') if old else None
        if previous is not None and (account is None or account['charged_frames']<previous['charged_frames']):
            report=dict(report,accounting=previous,accounting_complete=False)
        self.known[name]=report
    def collect(self):
        unknown=[]
        for name in self.started:
            report=self.known.get(name);account=report.get('accounting') if report else None
            if (account is None or report.get('accounting_complete') is False
                    or account.get('accounting_complete') is False or report.get('failure') is not None
                    or name in self.errors):unknown.append(name)
        return dict(constructor_count=len(self.started),engine_reports=self.known,report_errors=self.errors,
            uncertain_engine_reports=unknown,accounting_complete=not unknown,
            known_charged_frames=sum(r['accounting']['charged_frames'] for r in self.known.values()
                                    if r.get('accounting') is not None))


def config(case):
    return EmulatorConfig(game=case['game'],seed=case['seed'],max_episode_frames=case['limit'],
        max_charged_frames=2000,frame_skip=4,sticky_probability=.25,experiment_fraction=.2,mode='lab')


def run_case(case,baseline,reports,deadline):
    name=case['name'];pool=owner=None;peers=[]
    digest=hashlib.sha256();histogram=Counter();selected=[];branch_row=None
    # Load/validate the existing archive before any paid constructor.
    with np.load(ROOT/baseline['selected_trace_archive'],allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    try:
        if time.monotonic()>deadline:raise ValueError('Native work ceiling reached before construction')
        reports.begin(name)
        pool=IsolatedRGBPool([config(case)],python_executable=CHILD['executable'],expected_runtime=CHILD,
            timeout=SPEC['peer_timeout_seconds'])
        peers=list(pool._peers);reports.remember(name,pool.report(0))
        owner=OwnedRGBCollector(pool,1,session_id='isolated-native-v0/'+name)
        initial=owner.current_anchor();same_array(initial.pair,arrays['s000_start_pair'])
        previous=initial.pair[-1].copy();epoch=frame=0
        for index in range(case['decisions']):
            if time.monotonic()>deadline:raise ValueError('Native work ceiling reached before next command')
            action=(index//4)%18;operation=owner.step([action]);record=owner.take(operation)[0];trace=record.trace
            length=len(trace.raw);histogram[length]+=1
            changed=not np.array_equal(previous,trace.start_pair[-1])
            if changed:epoch+=1;frame=0
            timing=record.chronology
            if (timing.epoch,timing.start_frame,timing.end_frame,timing.visual_discontinuity)!=(epoch,frame,frame+length,changed):
                raise ValueError('Own delivered-image chronology differs')
            frame+=length;previous=(trace.raw[-1] if length else trace.start_pair[-1]).copy()
            anchor=owner.current_anchor()
            if (record.operation_id!=operation or record.requested_action!=action or record.worker!=0
                    or not np.array_equal(record.effect_assumption,np.zeros(12,np.float32))
                    or anchor.token!=(operation[0],operation[1],0) or (anchor.epoch,anchor.frame)!=(epoch,frame)):
                raise ValueError('Own record or anchor is detached from the paid command')
            trace_digest(digest,trace)
            if index in SPEC['selected_indices']:
                for field in fields(OrdinaryRGBTrace):same_array(getattr(trace,field.name),arrays[f's{index:03d}_{field.name}'])
                meta=dict(index=index,action=action,operation_id=['owned-ordinary-v0/'+name,operation[1]],
                    epoch=epoch,start_frame=timing.start_frame,end_frame=frame,visual_discontinuity=changed,
                    actual_frames=length,effect_assumption=[0.]*12)
                selected.append(meta)
            if case['branch_after']==index:
                if time.monotonic()>deadline:raise ValueError('Native work ceiling reached before branch')
                operation=owner.branch(5,expected_anchor=anchor.token,frames=8);branch=owner.take(operation)[0]
                same_array(branch.anchor.pair,arrays['branch_start_pair'])
                for branch_name,trace in (('intervention',branch.intervention),('control',branch.control)):
                    for field in ('raw','pooled','endpoint'):same_array(getattr(trace,field),arrays[f'branch_{branch_name}_{field}'])
                after=owner.current_anchor()
                same_array(after.pair,anchor.pair)
                if after.token!=anchor.token or (after.epoch,after.frame)!=(epoch,frame):
                    raise ValueError('Branch modified ordinary anchor or time')
                branch_row=dict(operation_id=['owned-ordinary-v0/'+name,operation[1]],
                    intervention_frames=len(branch.intervention.raw),control_frames=len(branch.control.raw),
                    anchor_frame=frame,anchor_epoch=epoch,requested_action=5,control_action=0,requested_frames=8,
                    effect_assumption=[0.]*12)
        report=pool.report(0);reports.remember(name,report)
        result=dict(name=name,passed=True,decisions=case['decisions'],length_counts={str(k):v for k,v in histogram.items()},
            final_epoch=epoch,final_delivered_frame=frame,ordered_trace_sha256=digest.hexdigest(),
            selected_steps=selected,branch=branch_row,accounting=report['accounting'])
        if any(value!=baseline[key] for key,value in result.items()):
            raise ValueError('Full ordered RGB digest, metadata or paid accounting differs from baseline')
        if report.get('failure') is not None:raise ValueError('Worker reported a native failure')
        return result
    except IsolatedWorkerError as error:
        if 0 in error.known_worker_reports:reports.remember(name,error.known_worker_reports[0])
        raise
    finally:
        if pool is not None:
            try:reports.remember(name,pool.report(0))
            except BaseException as error:reports.errors[name]=type(error).__name__
            finally:
                if owner is not None:owner.close()
                else:pool.close()
        if any(peer.process.poll() is None or any(t.is_alive() for t in peer._threads) for peer in peers):
            raise ValueError('Explicit child or transport threads did not stop')


def run():
    if ATTEMPT.exists() or OUTPUT.exists():raise ValueError('Preserve every attempt/result; no retry')
    proof=preflight();FOLDER.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',started_utc=utc(),**proof);write_new(ATTEMPT,record)
    reports=Reports();completed=[];deadline=time.monotonic()+SPEC['max_work_seconds']
    try:
        plan=json.loads((ROOT/BASE_PROTOCOL).read_text(encoding='utf-8'))
        baseline=json.loads((ROOT/BASE_RESULT).read_text(encoding='utf-8'))
        for case,old in zip(plan['cases'],baseline['cases'],strict=True):
            record['current_case']=case['name']
            result=run_case(case,old,reports,deadline);completed.append(result)
            print(json.dumps(dict(name=result['name'],passed=True,decisions=result['decisions'])),flush=True)
        accounts=reports.collect()
        if (not accounts['accounting_complete'] or accounts['constructor_count']!=9
                or accounts['known_charged_frames']!=7454 or time.monotonic()>deadline or preflight()!=proof):
            raise ValueError('Complete expected costs, bounded work and unchanged proof required')
        result=dict(kind='isolated_rgb_native_wiring',passed=True,created_utc=utc(),**proof,**accounts,
            cases=completed,agent_training_allowed=False,gpu_used=False,gate_result=None,
            model_inferences=0,optimizer_updates=0,
            limitations=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        write_new(OUTPUT,result);record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        raise
    finally:
        record.update(completed_cases=completed,**reports.collect())
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    result=run();print(json.dumps(dict(passed=result['passed'],charged_frames=result['known_charged_frames'])))
