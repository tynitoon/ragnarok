"""Prospective native ordinary RGB provenance; bounded CPU wiring, no learner."""
from collections import Counter
import hashlib
import json
import os

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.environments.ordinary_rgb_pool_v70 import OrdinaryRGBPool
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace
from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from scripts.temporal_validation_seal_v70 import ROOT,sha,utc,runtime,write_new,require_committed

PROTOCOL='docs/ARC4_OWNED_ORDINARY_NATIVE_PROTOCOL.json'
FOLDER=ROOT/'craft_v6_out/v70_owned_ordinary_native'
PREFLIGHT=FOLDER/'preflight_v0.json'
OUTPUT=FOLDER/'wiring_v0.json'
ATTEMPT=FOLDER/'wiring_v0_attempt.json'
PREREQUISITES={
    'craft_v6_out/v70_ordinary_rgb_trace/collector_review_v0.json':
        '6bf43d46eb874018c5e66096d70c6e85e257d9e81bac57daa08e2f00603e6b61',
    'craft_v6_out/v70_rgb_branch_trace/result_review_v0.json':
        '64e82d1b7f6b543ab8db7530771cca11590a0eaab9f03ca5f87dc88401b69902'}
NEW_SOURCES=('scripts/check_owned_ordinary_native_v70.py','tests/test_owned_ordinary_native_v70.py',
    'tests/test_rgb_branch_trace_wiring_v70.py',
    PROTOCOL,'docs/ARC4_OWNED_ORDINARY_NATIVE_DRAFT.md')
GAMES=('pong','breakout','freeway','seaquest','ms_pacman','asteroids')
CASES=[dict(name=game,game=game,seed=19200,limit=108000,decisions=146,branch_after=143,
            required_short_length=None) for game in GAMES]+[
    dict(name=f'pong_prefix_{length}',game='pong',seed=19210+length,limit=508+length,
         decisions=130,branch_after=None,required_short_length=length) for length in (1,2,3)]
SHARED=dict(frame_skip=4,sticky_probability=.25,experiment_fraction=.2,mode='lab',
    cap_per_engine=2000,branch_action=5,control_action=0,branch_frames=8,
    selected_ordinary_indices=[0,1,2,3,126,127,128,129],max_constructions=18,max_total_charge=36000)


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol.get('kind')!='prospective_owned_ordinary_native_v0' or protocol.get('cases')!=CASES
            or protocol.get('shared')!=SHARED or protocol.get('agent_training_allowed') is not False
            or protocol.get('gate_result') is not None):
        raise ValueError('Exact prospective native ordinary protocol required')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('One CPU numerical thread required')
    sources={};evidence={}
    for path,value in PREREQUISITES.items():
        if sha(ROOT/path)!=value:raise ValueError('Pinned independent prerequisite changed')
        review=json.loads((ROOT/path).read_text(encoding='utf-8'))
        if review.get('passed') is not True:raise ValueError('Successful independent prerequisite required')
        evidence[path]=value
        for field,destination in (('source_sha256',sources),('evidence_sha256',evidence)):
            for p,h in review[field].items():
                if sha(ROOT/p)!=h or p in destination and destination[p]!=h:
                    raise ValueError('Reviewed source/evidence changed or conflicts')
                destination[p]=h
    for p in NEW_SOURCES:sources[p]=sha(ROOT/p)
    for p in {*sources,*evidence}:require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=runtime())


def preflight():
    proof=proofs()
    review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='owned_ordinary_native_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before any native construction')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


class _RGBReadTap:
    """Private witness: copy the already-requested screen after the old engine paid."""
    def __init__(self,native,owner):self.native,self.owner=native,owner
    def __getattr__(self,name):return getattr(self.native,name)
    def getScreenRGB(self):
        pixels=self.native.getScreenRGB()
        if self.owner._witness_raw is not None:self.owner._witness_raw.append(pixels.copy())
        return pixels


class _WitnessEngine(_ColdPixelEngine):
    """Keep the original step/advance/act path; record its existing RGB read calls."""
    def __init__(self,config):
        self._witness_raw=None;self.last_trace=None
        super().__init__(config)
        self.set_native_wrapper(lambda native:_RGBReadTap(native,self))

    def _advance(self,action,frames,*,experimental):
        start=np.stack(tuple(self._frames))
        self._witness_raw=[];self.last_trace=None
        try:
            endpoint=super()._advance(action,frames,experimental=experimental)
            raw=np.stack(self._witness_raw) if self._witness_raw else np.empty((0,*endpoint.shape),np.uint8)
            pooled=[];previous=start[-1]
            for current in raw:
                pooled.append(np.maximum(previous,current));previous=current
            pooled=np.stack(pooled) if pooled else np.empty_like(raw)
            self.last_trace=OrdinaryRGBTrace(start,raw,pooled,endpoint)
            return endpoint
        finally:self._witness_raw=None


def compare_trace(actual,expected):
    for field in ('start_pair','raw','pooled','endpoint'):
        a,b=getattr(actual,field),getattr(expected,field)
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):
            raise ValueError(f'Independent native ordinary RGB {field} mismatch')


def trace_digest(digest,trace):
    for field in ('start_pair','raw','pooled','endpoint'):
        value=getattr(trace,field)
        digest.update(json.dumps([field,list(value.shape),str(value.dtype)],separators=(',',':')).encode()+b'\n')
        digest.update(value.tobytes(order='C'))


class _Reports:
    """Cache every successful report immediately; never lose it to a peer failure."""
    def __init__(self):self.readers={};self.known={};self.errors={};self.constructing=None;self.count=0
    def begin(self,name):self.constructing=name;self.count+=1
    def register(self,name,reader):self.readers[name]=reader;self.constructing=None
    def read(self,name):
        if name in self.known:return self.known[name]
        if name in self.errors:raise ValueError('This report already failed; no ambiguous reread')
        try:report=self.readers[name]()
        except BaseException as error:
            self.errors[name]=type(error).__name__;raise
        self.known[name]=report
        return report
    def collect(self):
        for name in self.readers:
            if name not in self.known and name not in self.errors:
                try:self.read(name)
                except BaseException:pass
        return dict(constructor_count=self.count,constructor_without_return=self.constructing,
            engine_reports=self.known,report_errors=self.errors,
            accounting_complete=self.constructing is None and len(self.known)==self.count,
            known_charged_frames=sum(r['accounting']['charged_frames'] for r in self.known.values()))


def config(case):
    return EmulatorConfig(game=case['game'],seed=case['seed'],max_episode_frames=case['limit'],
        max_charged_frames=2000,frame_skip=4,sticky_probability=.25,experiment_fraction=.2,mode='lab')


def run_case(case,reports):
    name=case['name'];pool=None;owner=None;arrays={};selected=[]
    digest=hashlib.sha256();histogram=Counter();epoch=frame=0
    try:
        reports.begin(name+'/witness');witness=_WitnessEngine(config(case))
        reports.register(name+'/witness',lambda:dict(accounting=witness.accounting(),failure=None))
        reports.begin(name+'/worker');pool=OrdinaryRGBPool([config(case)])
        reports.register(name+'/worker',lambda:pool.report(0))
        owner=OwnedRGBCollector(pool,1,session_id='owned-ordinary-v0/'+name)
        initial=owner.current_anchor()
        previous=witness.observe(raw=True)
        if not np.array_equal(initial.pair,np.stack(tuple(witness._frames))):raise ValueError('Initial RGB differs')
        branch_record=None
        for index in range(case['decisions']):
            action=(index//4)%18
            witness.step(action)
            operation=owner.step([action]);records=owner.take(operation)
            if len(records)!=1:raise ValueError('One own worker record required')
            record=records[0];trace=record.trace
            compare_trace(trace,witness.last_trace)
            length=len(trace.raw);histogram[length]+=1
            changed=not np.array_equal(previous,trace.start_pair[-1])
            if changed:epoch+=1;frame=0
            timing=record.chronology
            if (timing.epoch,timing.start_frame,timing.end_frame,timing.visual_discontinuity)!=(epoch,frame,frame+length,changed):
                raise ValueError('Owned chronology does not follow only delivered raw images')
            frame+=length;previous=(trace.raw[-1] if length else trace.start_pair[-1]).copy()
            anchor=owner.current_anchor()
            if (record.operation_id!=operation or record.requested_action!=action or record.worker!=0
                    or not np.array_equal(record.effect_assumption,np.zeros(12,np.float32))
                    or anchor.token!=(operation[0],operation[1],0) or (anchor.epoch,anchor.frame)!=(epoch,frame)
                    or not np.array_equal(anchor.pair,np.stack(tuple(witness._frames)))):
                raise ValueError('Owned record or endpoint anchor differs from its paid command')
            trace_digest(digest,trace)
            if index in SHARED['selected_ordinary_indices']:
                for field in ('start_pair','raw','pooled','endpoint'):
                    arrays[f's{index:03d}_{field}']=getattr(trace,field).copy()
                selected.append(dict(index=index,action=action,operation_id=list(operation),
                    epoch=epoch,start_frame=timing.start_frame,end_frame=frame,visual_discontinuity=changed,
                    actual_frames=length,effect_assumption=[0.]*12))
            if case['branch_after']==index:
                expected=witness.branch_pair(5,control=0,frames=8)
                branch_operation=owner.branch(5,expected_anchor=anchor.token,frames=8)
                branch=owner.take(branch_operation)[0]
                for actual,pixels in zip((branch.intervention,branch.control),expected,strict=True):
                    if not np.array_equal(actual.endpoint,pixels):raise ValueError('Branch endpoint mismatch')
                after=owner.current_anchor()
                if (after.token!=anchor.token or (after.epoch,after.frame)!=(epoch,frame)
                        or not np.array_equal(after.pair,anchor.pair)
                        or not np.array_equal(after.pair,np.stack(tuple(witness._frames)))):
                    raise ValueError('Branch changed ordinary anchor or delivered-image time')
                branch_record=dict(operation_id=list(branch_operation),intervention_frames=len(branch.intervention.raw),
                    control_frames=len(branch.control.raw),anchor_frame=frame,anchor_epoch=epoch)
        required=case['required_short_length']
        if required is not None and histogram[required]<1:raise ValueError('Predeclared terminal prefix was not covered')
        left=reports.read(name+'/witness');right=reports.read(name+'/worker')
        if left!=right:raise ValueError('Full paid accounting differs between independent native paths')
        path=FOLDER/(name+'_rgb_v0.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        result=dict(name=name,passed=True,decisions=case['decisions'],length_counts=dict(histogram),
            final_epoch=epoch,final_delivered_frame=frame,ordered_trace_sha256=digest.hexdigest(),
            selected_trace_archive=path.relative_to(ROOT).as_posix(),selected_trace_sha256=sha(path),
            selected_steps=selected,branch=branch_record,accounting=left['accounting'])
        processes=list(pool._processes);owner.close();owner=None;pool=None
        if any(p.is_alive() for p in processes):raise ValueError('Native ordinary workers did not exit')
        return result
    finally:
        # Read while workers are alive, even when validation/copy/archive failed.
        reports.collect()
        if owner is not None:owner.close()
        elif pool is not None:pool.close()


def run():
    artifacts=[OUTPUT,ATTEMPT,*(FOLDER/(c['name']+'_rgb_v0.npz') for c in CASES)]
    if any(p.exists() for p in artifacts):raise ValueError('Preserve every native ordinary attempt/artifact; no retry')
    proof=preflight()
    FOLDER.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',started_utc=utc(),**proof)
    write_new(ATTEMPT,record)
    reports=_Reports();completed=[]
    try:
        for case in CASES:
            result=run_case(case,reports);completed.append(result)
            print(json.dumps({k:result[k] for k in ('name','passed','decisions','length_counts')}),flush=True)
        accounting=reports.collect()
        if (not accounting['accounting_complete'] or accounting['constructor_count']!=18
                or accounting['known_charged_frames']>36000 or preflight()!=proof):
            raise ValueError('Complete native ordinary accounting and unchanged proof required')
        result=dict(kind='owned_ordinary_native_wiring',passed=True,created_utc=utc(),**proof,
            **accounting,cases=completed,agent_training_allowed=False,gpu_used=False,gate_result=None,
            limitations=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        raise
    finally:
        record.update(completed_cases=completed,**reports.collect())
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(passed=result['passed'],charged_frames=result['known_charged_frames'])))
