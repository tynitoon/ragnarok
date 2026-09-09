"""Prospective separate loss candidate on the original two synthetic scenes.

Use the existing ordered/native-P result as an immutable baseline. Do not rerun
either old arm. The success criterion is still native-image L1, not blurred L1.
"""
import json
import time

import torch

from scripts import check_synthetic_motion_fit_v70 as base
from ragnarok.learning.blurred_prediction_loss_v70 import blurred_prediction_loss

ROOT=base.ROOT
PROTOCOL='docs/ARC4_BLURRED_MOTION_FIT_PROTOCOL.json'
FOLDER=ROOT/'craft_v6_out/v70_blurred_motion_fit'
PREFLIGHT=FOLDER/'preflight_v0.json'
ATTEMPT=FOLDER/'fit_v0_attempt.json'
OUTPUT=FOLDER/'fit_v0.json'
BASE_RESULT='craft_v6_out/v70_synthetic_motion_fit/fit_v0.json'
BASE_RESULT_SHA='9d049a632816855b5dd2497681c9ff900bdbe04d7d8efbb68bb0cbb56514b82f'
REVIEWS=('craft_v6_out/v70_synthetic_motion_fit/fit_review_v0.json',
         'craft_v6_out/v70_parser_synthetic/blurred_prediction_review_v0.json')
LOSS_SOURCES=('ragnarok/learning/blurred_prediction_loss_v70.py',
    'tests/test_blurred_prediction_loss_v70.py','docs/ARC4_BLURRED_PREDICTION_DRAFT.md')
NEW_SOURCES=(PROTOCOL,'docs/ARC4_BLURRED_MOTION_FIT_DRAFT.md',
    'scripts/check_blurred_motion_fit_v70.py','tests/test_blurred_motion_fit_protocol_v70.py')
SPEC=dict(arm='ordered_blurred',parser_seed=764,updates=256,batch_size=2,
    objective='mean_rgb_l1_box_widths_1_3_7',learning_rate=.0003,betas=[.9,.999],epsilon=1e-8,
    weight_decay=0.,max_gradient_norm=1.,measure_at=[0,1,8,32,128,256],max_work_seconds=300,
    success_metric='native_image_l1',native_fit_threshold=.01171875,
    baseline_reexecuted=False,cpu_threads=1,emulated_frames=0,gpu_used=False,
    independent_generalization_test=False)


def baseline():
    if base.sha(ROOT/BASE_RESULT)!=BASE_RESULT_SHA:raise ValueError('The original negative fit result changed')
    result=json.loads((ROOT/BASE_RESULT).read_text(encoding='utf-8'))
    if (result.get('execution_complete') is not True or result.get('optimizer_updates')!=512
            or result.get('ordered_fit_below_declared_threshold') is not False
            or [arm['arm'] for arm in result['arms']]!=['pooled','ordered']):
        raise ValueError('The complete disclosed original two-arm result is required')
    return result['arms'][1]


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'));original=base.proofs()
    if (protocol.get('kind')!='prospective_blurred_motion_fit_v0' or protocol.get('spec')!=SPEC
            or protocol.get('runtime')!=original['runtime'] or protocol.get('gate_result') is not None):
        raise ValueError('Exact prospective CPU loss-candidate protocol required')
    sources=dict(original['source_sha256']);evidence=dict(original['evidence_sha256'])
    for path in REVIEWS:
        review=json.loads((ROOT/path).read_text(encoding='utf-8'))
        if review.get('passed') is not True:raise ValueError('Independent result/loss review required')
        evidence[path]=base.sha(ROOT/path)
        for field,destination in (('source_sha256',sources),('evidence_sha256',evidence)):
            for p,h in review[field].items():
                if base.sha(ROOT/p)!=h or p in destination and destination[p]!=h:
                    raise ValueError('Reviewed source/evidence changed or conflicts')
                destination[p]=h
        if path==REVIEWS[1] and any(review['source_sha256'].get(p)!=base.sha(ROOT/p) for p in LOSS_SOURCES):
            raise ValueError('The complete current loss candidate must be reviewed')
        if path==REVIEWS[0] and review['evidence_sha256'].get(BASE_RESULT)!=BASE_RESULT_SHA:
            raise ValueError('The original fit result must be covered by its independent review')
    baseline();evidence[BASE_RESULT]=BASE_RESULT_SHA
    for p in NEW_SOURCES:sources[p]=base.sha(ROOT/p)
    for p in {*sources,*evidence}:base.require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=original['runtime'])


def preflight():
    proof=proofs();review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='blurred_motion_fit_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before any candidate update')
    base.require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=base.sha(PREFLIGHT))


def measure(model,data,updates):
    # Preserve the original native-image metric and all its diagnostics.
    row=base.measure(model,'ordered',data,updates)
    previous,current,target=data
    with torch.no_grad():
        output=base.infer(model,'ordered',previous,current)
        terms=base.prediction_terms(base.max_pool_rgb(previous,current),target,output)
        candidate=blurred_prediction_loss(terms['reconstructed'],target.float()/255)
        row['candidate_objective']=float(candidate['loss'])
        row['scale_l1']={str(k):float(v) for k,v in candidate['scale_l1'].items()}
    return row


def run():
    if ATTEMPT.exists() or OUTPUT.exists():raise ValueError('Preserve every candidate attempt/result; no retry')
    proof=preflight();old=baseline();FOLDER.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',started_utc=base.utc(),**proof,completed_updates=0,
                optimizer_step_in_progress=None,measurements=[])
    base.write_new(ATTEMPT,record);deadline=time.monotonic()+SPEC['max_work_seconds']
    try:
        data=base.scenes();torch.manual_seed(SPEC['parser_seed'])
        with torch.device('cpu'):model=base.OrderedMotionParser()
        initial_weights=base.weight_sha256(model)
        if initial_weights!=old['initial_weight_sha256']:raise ValueError('Candidate must start from identical virgin ordered weights')
        first=measure(model,data,0)
        if {k:first[k] for k in old['measurements'][0]}!=old['measurements'][0]:
            raise ValueError('Initial native measurements must exactly match the ordered baseline')
        rows=record['measurements'];rows.append(first)
        optimizer=torch.optim.Adam(model.parameters(),lr=SPEC['learning_rate'],betas=tuple(SPEC['betas']),
            eps=SPEC['epsilon'],weight_decay=SPEC['weight_decay'],foreach=False,fused=False)
        for update in range(1,SPEC['updates']+1):
            if time.monotonic()>deadline:raise ValueError('Candidate work ceiling reached')
            optimizer.zero_grad(set_to_none=True)
            previous,current,target=data;output=base.infer(model,'ordered',previous,current)
            terms=base.prediction_terms(base.max_pool_rgb(previous,current),target,output)
            loss=blurred_prediction_loss(terms['reconstructed'],target.float()/255)['loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite candidate objective')
            loss.backward();base.admit_adam_gradients(model,max_norm=SPEC['max_gradient_norm'])
            record['optimizer_step_in_progress']=update
            optimizer.step();record['completed_updates']=update;record['optimizer_step_in_progress']=None
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite candidate parameters')
            if update in SPEC['measure_at']:rows.append(measure(model,data,update))
        optimizer.zero_grad(set_to_none=True)
        if time.monotonic()>deadline or preflight()!=proof:raise ValueError('Bounded candidate fit and unchanged proof required')
        final_error=rows[-1]['image_l1']
        result=dict(kind='blurred_motion_fit_result',execution_complete=True,created_utc=base.utc(),**proof,
            optimizer_updates=256,initial_weight_sha256=initial_weights,final_weight_sha256=base.weight_sha256(model),
            measurements=rows,native_fit_below_declared_threshold=final_error<SPEC['native_fit_threshold'],
            baseline_final_native_l1=old['measurements'][-1]['image_l1'],baseline_result_sha256=BASE_RESULT_SHA,
            new_emulated_frames=0,gpu_used=False,gate_result=None,
            limitations=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        base.write_new(OUTPUT,result);record.update(status='complete',completed_utc=base.utc(),result_sha256=base.sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=base.utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        raise
    finally:ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1)
    result=run();print(json.dumps(dict(execution_complete=True,
        final_native_image_l1=result['measurements'][-1]['image_l1'],
        native_fit_below_declared_threshold=result['native_fit_below_declared_threshold'])))
