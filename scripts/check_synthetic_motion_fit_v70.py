"""Prospective two-scene P-only fitting diagnostic; not an Atari training gate."""
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

from ragnarok.learning.parser_v70 import CounterfactualParser,max_pool_rgb
from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.pixel_prediction_v70 import prediction_terms
from ragnarok.learning.gradient_guard_v70 import admit_adam_gradients
from ragnarok.learning.archived_pixel_diagnostics_v70 import weight_sha256

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='docs/ARC4_SYNTHETIC_MOTION_FIT_PROTOCOL.json'
FOLDER=ROOT/'craft_v6_out/v70_synthetic_motion_fit'
PREFLIGHT=FOLDER/'preflight_v0.json'
ATTEMPT=FOLDER/'fit_v0_attempt.json'
OUTPUT=FOLDER/'fit_v0.json'
REVIEWS={
    'craft_v6_out/v70_parser_synthetic/ordered_motion_review_v0.json':
        '138a3049877c38636a584cfa0de065c1c28d461aefcc5fa5b203e87a8b4b692f',
    'craft_v6_out/v70_parser_synthetic/gradient_guard_review_v0.json':
        'c5119bea17ac84b0db60cb94b2a0f54049e8e2e63b08f21b6d9371e1a4f59bd4',
    'craft_v6_out/v70_parser_synthetic/archived_pixel_diagnostics_review_v0.json':
        '968d7f029ba1f971a518168cd630f6108fa861e55ab618c0254e89ed5eaf8a5b'}
NEW_SOURCES=(PROTOCOL,'docs/ARC4_SYNTHETIC_MOTION_FIT_DRAFT.md',
    'scripts/check_synthetic_motion_fit_v70.py','tests/test_synthetic_motion_fit_protocol_v70.py',
    'tests/test_prediction_loss_translation_limits_v70.py','docs/ARC4_PREDICTION_LOSS_LIMITS.md')
SPEC=dict(arms=['pooled','ordered'],seed=764,updates_per_arm=256,batch_size=2,
    objective='prediction_image_l1_only',learning_rate=.0003,betas=[.9,.999],epsilon=1e-8,
    weight_decay=0.,max_gradient_norm=1.,measure_at=[0,1,8,32,128,256],max_work_seconds=300,
    pooled_common_prediction_lower_bound=.0234375,ordered_fit_threshold=.01171875,
    cpu_threads=1,emulated_frames=0,gpu_used=False,independent_generalization_test=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def require_committed(path):
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('hash-object',f'--path={path}',str(ROOT/path))!=git('rev-parse',f'HEAD:{path}'):
        raise ValueError('Current source/proof differs from its committed version')


def runtime():
    return dict(executable=str(Path(sys.executable).resolve()),python=sys.version,numpy=np.__version__,
        torch=str(torch.__version__),scipy=importlib.metadata.version('scipy'),
        torch_threads=torch.get_num_threads(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        numerical_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'));current=runtime()
    if (protocol.get('kind')!='prospective_synthetic_motion_fit_v0' or protocol.get('spec')!=SPEC
            or protocol.get('runtime')!=current or protocol.get('gate_result') is not None
            or current['torch_threads']!=1 or current['cuda_visible_devices']!=''
            or any(v!='1' for v in current['numerical_threads'].values())):
        raise ValueError('Exact prospective synthetic CPU fitting protocol required')
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
    for p in NEW_SOURCES:sources[p]=sha(ROOT/p)
    for p in {*sources,*evidence}:require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=current)


def preflight():
    proof=proofs();review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='synthetic_motion_fit_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before any fitting update')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


def scenes():
    def image(x):
        value=torch.zeros((1,3,16,16),dtype=torch.uint8,device='cpu')
        value[:,:,6:8,x:x+2]=255
        return value
    previous=torch.cat((image(4),image(5)))
    current=torch.cat((image(5),image(4)))
    target=torch.cat((max_pool_rgb(image(8),image(9)),max_pool_rgb(image(1),image(0))))
    return previous,current,target


def infer(model,arm,previous,current):
    unknown=torch.zeros((len(previous),12),dtype=torch.float32,device='cpu')
    if arm=='ordered':output=model(previous,current,unknown)
    elif arm=='pooled':output=model(max_pool_rgb(previous,current),unknown)
    else:raise ValueError('Only the two declared architectures are admitted')
    for name in ('embedding','foreground_logits','displacement','controllability_logits','solidity_logits'):
        if not torch.isfinite(getattr(output,name)).all():raise ValueError('Nonfinite synthetic parser head')
    return output


def measure(model,arm,data,updates):
    previous,current,target=data
    with torch.no_grad():
        output=infer(model,arm,previous,current)
        terms=prediction_terms(max_pool_rgb(previous,current),target,output)
        errors=(terms['reconstructed']-target.float()/255).abs().mean(dim=(1,2,3))
        if not torch.isfinite(errors).all():raise ValueError('Nonfinite synthetic measurement')
        # Architecture invariant, not a fitted criterion: the two pooled inputs
        # are identical, hence their deterministic reconstructions must agree.
        if arm=='pooled':
            if (not torch.equal(terms['reconstructed'][0],terms['reconstructed'][1])
                    or float(errors.mean())<SPEC['pooled_common_prediction_lower_bound']-1e-7):
                raise ValueError('Pooled ambiguity invariant violated')
        return dict(updates=updates,image_l1=float(errors.mean()),scene_l1=errors.tolist(),
            reconstruction_difference=float((terms['reconstructed'][0]-terms['reconstructed'][1]).abs().mean()),
            foreground_target_mean=float(terms['foreground_target'].mean()))


def run():
    if ATTEMPT.exists() or OUTPUT.exists():raise ValueError('Preserve every fit attempt/result; no retry')
    proof=preflight();FOLDER.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',started_utc=utc(),**proof,completed_updates={},current_arm=None,
                optimizer_step_in_progress=None)
    write_new(ATTEMPT,record);completed=[];deadline=time.monotonic()+SPEC['max_work_seconds']
    try:
        data=scenes()
        for arm in SPEC['arms']:
            record.update(current_arm=arm,current_rows=[]);record['completed_updates'][arm]=0
            torch.manual_seed(SPEC['seed'])
            with torch.device('cpu'):model=CounterfactualParser() if arm=='pooled' else OrderedMotionParser()
            initial_weights=weight_sha256(model)
            optimizer=torch.optim.Adam(model.parameters(),lr=SPEC['learning_rate'],betas=tuple(SPEC['betas']),
                eps=SPEC['epsilon'],weight_decay=SPEC['weight_decay'],foreach=False,fused=False)
            rows=record['current_rows'];rows.append(measure(model,arm,data,0))
            for update in range(1,SPEC['updates_per_arm']+1):
                if time.monotonic()>deadline:raise ValueError('Synthetic fitting work ceiling reached')
                optimizer.zero_grad(set_to_none=True)
                previous,current,target=data;output=infer(model,arm,previous,current)
                loss=prediction_terms(max_pool_rgb(previous,current),target,output)['image_l1']
                if not torch.isfinite(loss):raise ValueError('Nonfinite synthetic training loss')
                loss.backward();admit_adam_gradients(model,max_norm=SPEC['max_gradient_norm'])
                # Optimizer failure can occur after a partial parameter change.
                # Keep this marker if step raises; never report that attempt as
                # a completed update or as certainly having changed no weights.
                record['optimizer_step_in_progress']=dict(arm=arm,update=update)
                optimizer.step();record['completed_updates'][arm]=update
                record['optimizer_step_in_progress']=None
                if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite updated parameters')
                if update in SPEC['measure_at']:rows.append(measure(model,arm,data,update))
            optimizer.zero_grad(set_to_none=True)
            result=dict(arm=arm,completed_updates=SPEC['updates_per_arm'],initial_weight_sha256=initial_weights,
                final_weight_sha256=weight_sha256(model),measurements=rows)
            completed.append(result)
            print(json.dumps(dict(arm=arm,updates=result['completed_updates'],final_image_l1=rows[-1]['image_l1'])),flush=True)
        if time.monotonic()>deadline or preflight()!=proof:raise ValueError('Bounded fit and unchanged proof required')
        ordered=completed[1]['measurements'][-1]['image_l1']
        result=dict(kind='synthetic_motion_fit_result',execution_complete=True,created_utc=utc(),**proof,
            arms=completed,ordered_fit_below_declared_threshold=ordered<SPEC['ordered_fit_threshold'],
            optimizer_updates=512,new_emulated_frames=0,gpu_used=False,gate_result=None,
            limitations=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        write_new(OUTPUT,result);record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        raise
    finally:
        record['completed_arms']=completed
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1)
    result=run();print(json.dumps(dict(execution_complete=True,
        ordered_fit_below_declared_threshold=result['ordered_fit_below_declared_threshold'])))
