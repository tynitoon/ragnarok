"""One prospective CPU numerical check on fixed, reviewed own RGB archives."""
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
import torch

from ragnarok.environments.owned_rgb_collector_v70 import StepExperience
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace,RGBChronologyStep,validate_trace
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.learning.parser_v70 import CounterfactualParser
from ragnarok.learning.archived_pixel_diagnostics_v70 import ArchivedPixelDiagnostics,weight_sha256

# This runner deliberately does not import the ALE-dependent seal helper: the
# installed CPU Torch interpreter and frozen emulator interpreter are distinct.
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='docs/ARC4_ARCHIVED_PIXEL_DIAGNOSTICS_PROTOCOL.json'
FOLDER=ROOT/'craft_v6_out/v70_archived_pixel_diagnostics'
OUTPUT=FOLDER/'cpu_v0.json'
ATTEMPT=FOLDER/'cpu_v0_attempt.json'
PREFLIGHT=FOLDER/'preflight_v0.json'
NATIVE_RESULT='craft_v6_out/v70_owned_ordinary_native/wiring_v0.json'
NATIVE_RESULT_SHA='5db50e4ee27fedf0ec77a62fab9c13efd309749f7a127e47d73ec4f56ae886af'
PINNED_REVIEWS={
    'craft_v6_out/v70_owned_ordinary_native/wiring_review_v0.json':
        '8b4105ce479e7d990293a89ce8ea3789506600729f3e1f721dbf59653f04a668',
    'craft_v6_out/v70_parser_synthetic/gradient_guard_review_v0.json':
        'c5119bea17ac84b0db60cb94b2a0f54049e8e2e63b08f21b6d9371e1a4f59bd4'}
SOURCE_REVIEW='craft_v6_out/v70_parser_synthetic/archived_pixel_diagnostics_review_v0.json'
DIAGNOSTIC_SOURCES=('ragnarok/learning/archived_pixel_diagnostics_v70.py',
    'tests/test_archived_pixel_diagnostics_v70.py','docs/ARC4_ARCHIVED_PIXEL_DIAGNOSTICS_DRAFT.md')
NEW_SOURCES=(*DIAGNOSTIC_SOURCES,'scripts/check_archived_pixel_diagnostics_v70.py',
    'tests/test_archived_pixel_diagnostic_protocol_v70.py',PROTOCOL)
SPEC=dict(cases=['pong','breakout','freeway','seaquest','ms_pacman','asteroids',
    'pong_prefix_1','pong_prefix_2','pong_prefix_3'],ordinary_indices=[0,1,2,3,126,127,128,129],
    ordinary_records=72,branch_pairs=6,parser_seed=764,reset_identical_weights_per_case=True,
    max_tracks=512,max_groups=512,gradient_norm_bound=1.,cpu_threads=1,
    max_work_seconds=900,optimizer_updates=0,new_emulated_frames=0)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def require_committed(path):
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('hash-object',f'--path={path}',str(ROOT/path))!=git('rev-parse',f'HEAD:{path}'):
        raise ValueError('A current source/proof differs from its committed version')


def runtime():
    return dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,
        scipy=importlib.metadata.version('scipy'),torch_threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        numerical_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    current=runtime()
    if (protocol.get('kind')!='prospective_archived_pixel_cpu_diagnostics_v0' or protocol.get('spec')!=SPEC
            or protocol.get('agent_training_allowed') is not False or protocol.get('gate_result') is not None
            or current['torch_threads']!=1 or current['cuda_visible_devices']!=''
            or any(value!='1' for value in current['numerical_threads'].values())):
        raise ValueError('Exact prospective CPU-only diagnostic protocol and one numerical thread required')
    sources={};evidence={}
    for path,expected in [*PINNED_REVIEWS.items(),(SOURCE_REVIEW,None)]:
        value=sha(ROOT/path)
        if expected is not None and value!=expected:raise ValueError('Pinned independent prerequisite changed')
        review=json.loads((ROOT/path).read_text(encoding='utf-8'))
        if review.get('passed') is not True:raise ValueError('Passing independent prerequisite required')
        if path==SOURCE_REVIEW and any(review.get('source_sha256',{}).get(p)!=sha(ROOT/p) for p in DIAGNOSTIC_SOURCES):
            raise ValueError('Current archived diagnostic helper must be independently reviewed')
        evidence[path]=value
        for field,destination in (('source_sha256',sources),('evidence_sha256',evidence)):
            for p,h in review[field].items():
                if sha(ROOT/p)!=h or p in destination and destination[p]!=h:
                    raise ValueError('Reviewed source or proof changed or conflicts')
                destination[p]=h
    if sha(ROOT/NATIVE_RESULT)!=NATIVE_RESULT_SHA:raise ValueError('The declared native archive result changed')
    evidence[NATIVE_RESULT]=NATIVE_RESULT_SHA
    native=json.loads((ROOT/NATIVE_RESULT).read_text(encoding='utf-8'))
    if (native.get('passed') is not True or native.get('accounting_complete') is not True
            or [c['name'] for c in native['cases']]!=SPEC['cases'] or native.get('known_charged_frames')!=14908):
        raise ValueError('Complete declared native RGB wiring result required')
    for case in native['cases']:
        path=case['selected_trace_archive']
        if (Path(path).parent.as_posix()!='craft_v6_out/v70_owned_ordinary_native'
                or [s['index'] for s in case['selected_steps']]!=SPEC['ordinary_indices']
                or sha(ROOT/path)!=case['selected_trace_sha256']):
            raise ValueError('Exact preselected own RGB trace archive required')
        evidence[path]=case['selected_trace_sha256']
    for p in NEW_SOURCES:sources[p]=sha(ROOT/p)
    for p in {*sources,*evidence}:require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=current)


def preflight():
    proof=proofs()
    review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='archived_pixel_cpu_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before any real-archive inference')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


def read_steps(case,arrays):
    expected={f's{index:03d}_{f.name}' for index in SPEC['ordinary_indices'] for f in fields(OrdinaryRGBTrace)}
    if case['branch'] is not None:
        expected.add('branch_start_pair')
        expected.update(f'branch_{name}_{f.name}' for name in ('intervention','control') for f in fields(RGBBranchTrace))
    if set(arrays)!=expected:raise ValueError('Archive contains missing or extra fields')
    result=[]
    for meta in case['selected_steps']:
        index=meta['index']
        trace=OrdinaryRGBTrace(*(arrays[f's{index:03d}_{f.name}'].copy() for f in fields(OrdinaryRGBTrace)))
        length=validate_trace(trace)
        if (trace.endpoint.shape!=(210,160,3) or length!=meta['actual_frames']
                or meta['action']!=(index//4)%18 or meta['effect_assumption']!=[0.]*12
                or meta['operation_id']!=['owned-ordinary-v0/'+case['name'],index+1]):
            raise ValueError('Native own record differs from its fixed source command or RGB domain')
        chronology=RGBChronologyStep(meta['epoch'],meta['start_frame'],meta['end_frame'],meta['visual_discontinuity'])
        result.append((index,StepExperience(tuple(meta['operation_id']),0,meta['action'],
            np.array(meta['effect_assumption'],dtype=np.float32),chronology,trace)))
    return result


def run():
    if OUTPUT.exists() or ATTEMPT.exists():raise ValueError('Preserve every CPU diagnostic result and attempt; no retry')
    proof=preflight()
    record=dict(status='started',started_utc=utc(),**proof,new_emulated_frames=0,optimizer_updates=0)
    FOLDER.mkdir(parents=True,exist_ok=True);write_new(ATTEMPT,record)
    completed=[];initial_weights=None;ordinary_count=branch_count=0;started=time.perf_counter()
    try:
        native=json.loads((ROOT/NATIVE_RESULT).read_text(encoding='utf-8'))
        for case in native['cases']:
            record['current_case']=case['name'];record['current_phase']='load_and_initialize'
            with np.load(ROOT/case['selected_trace_archive'],allow_pickle=False) as archive:
                arrays={name:archive[name] for name in archive.files}
            steps=read_steps(case,arrays)
            torch.manual_seed(SPEC['parser_seed'])
            with torch.device('cpu'):parser=CounterfactualParser()
            weights=weight_sha256(parser)
            if initial_weights is None:initial_weights=weights
            if weights!=initial_weights:raise ValueError('Every case must begin with exactly the same untrained weights')
            diagnostic=ArchivedPixelDiagnostics(parser,stream_id='native-wiring-archive/'+case['name'])
            case_rows=[];branch_row=None
            record['current_case_rows']=case_rows;record['current_branch']=None
            for index,saved in steps:
                if time.perf_counter()-started>SPEC['max_work_seconds']:raise ValueError('Declared CPU diagnostic work ceiling reached')
                record.update(current_phase='ordinary',current_index=index)
                row=diagnostic.step(index,saved);case_rows.append(row);ordinary_count+=1
            if case['branch'] is not None:
                if time.perf_counter()-started>SPEC['max_work_seconds']:raise ValueError('Declared CPU diagnostic work ceiling reached')
                record['current_phase']='branch'
                meta=case['branch']
                if (meta['requested_action'],meta['control_action'],meta['requested_frames'],meta['effect_assumption'])!=(5,0,8,[0.]*12):
                    raise ValueError('The archived counterfactual command contract changed')
                def trace(name):return RGBBranchTrace(*(arrays[f'branch_{name}_{f.name}'].copy() for f in fields(RGBBranchTrace)))
                branch_row=diagnostic.branch(tuple(meta['operation_id']),start_pair=arrays['branch_start_pair'],
                    effect=np.array(meta['effect_assumption'],dtype=np.float32),
                    intervention=trace('intervention'),control=trace('control'),requested_frames=8)
                record['current_branch']=branch_row
                branch_count+=1
            if weight_sha256(parser)!=initial_weights:raise ValueError('CPU diagnostics modified model weights')
            completed.append(dict(case=case['name'],initial_weight_sha256=weights,ordinary=case_rows,branch=branch_row))
            print(json.dumps(dict(case=case['name'],ordinary_records=len(case_rows),
                prediction_eligible=sum(r['prediction_eligible'] for r in case_rows),
                identity_anchors=sum(r['identity_anchors'] for r in case_rows),
                c_supervised_pixels=branch_row['supervised_pixels'] if branch_row else None)),flush=True)
        if time.perf_counter()-started>SPEC['max_work_seconds']:raise ValueError('Declared CPU diagnostic work ceiling reached')
        if ordinary_count!=72 or branch_count!=6 or preflight()!=proof:
            raise ValueError('Complete fixed objective population and unchanged source/evidence required')
        result=dict(kind='archived_pixel_cpu_numerical_diagnostics',passed=True,created_utc=utc(),**proof,
            cases=completed,ordinary_records=ordinary_count,branch_pairs=branch_count,
            initial_weight_sha256=initial_weights,optimizer_updates=0,new_emulated_frames=0,
            gpu_used=False,agent_training_allowed=False,gate_result=None,
            limitations=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        raise
    finally:
        record.update(completed_cases=completed,completed_ordinary_records=ordinary_count,completed_branch_pairs=branch_count)
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1)
    result=run()
    print(json.dumps(dict(passed=result['passed'],ordinary_records=result['ordinary_records'],branch_pairs=result['branch_pairs'])))
