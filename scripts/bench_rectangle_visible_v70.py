"""Prospective rectangle-only MsPacman native wiring and one fixed CPU comparison."""
import argparse
import json
import os
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine
from ragnarok.evaluation.rectangle_visible_pool_v70 import RectangleVisibleEngine,RectangleVisiblePool
from scripts import bench_visible_reference_v70 as original
from scripts import check_rectangle_groups_v70 as grouping_audit
from scripts.check_visible_cpu_v70 import _same_history
from scripts.temporal_validation_seal_v70 import ROOT,sha,runtime,utc,write_new,require_committed

PROTOCOL='docs/ARC4_RECTANGLE_CPU_PROTOCOL.json'
FOLDER=ROOT/'craft_v6_out/v70_mspacman_rectangle_cpu'
PREFLIGHT=FOLDER/'preflight_review_v0.json'
WIRING=FOLDER/'wiring_v0.json'
WIRING_REVIEW=FOLDER/'wiring_review_v0.json'
OUTPUT=FOLDER/'native_chain_v0.json'
BASELINE=ROOT/'craft_v6_out/v70_mspacman_cpu/cpu_chain_native_v0.json'
CPU_REVIEW='craft_v6_out/v70_mspacman_cpu/cpu_chain_result_review_v0.json'
CPU_REVIEW_SHA='40f640d4fa5648c5eff98de7a723f3f093bc19783b11a2fc74f1fa9c532742ff'
GROUP_REVIEW='craft_v6_out/v70_rectangle_groups/offline_review_v0.json'
GROUP_REVIEW_SHA='5fba09e17802c0b7617a75fac34309ce515b1f5b15bc1acdbe27eec46e471de8'
GROUP_RESULT_SHA='97569b0191c259c3758d5581d5870f10af92240aef41b0db44cab7966de27440'
ADAPTER_REVIEW='craft_v6_out/v70_rectangle_groups/source_review_cpu_v0.json'
ADAPTER_SOURCES=('ragnarok/evaluation/rectangle_visible_pool_v70.py',
    'tests/test_rectangle_visible_pool_v70.py','docs/ARC4_RECTANGLE_CPU_DRAFT.md')
SOURCE_PATHS=tuple(dict.fromkeys((*original.SOURCE_PATHS,*grouping_audit.prior.SOURCE_PATHS,
    *grouping_audit.NEW_SOURCES,'ragnarok/evaluation/__init__.py','tests/__init__.py',
    'ragnarok/evaluation/rectangle_visible_pool_v70.py','tests/test_rectangle_visible_pool_v70.py',
    'ragnarok/evaluation/cached_visible_pool_v70.py','ragnarok/evaluation/cached_visible_groups_v70.py',
    'tests/test_cached_visible_pool_v70.py','docs/ARC4_RECTANGLE_CPU_DRAFT.md',
    'scripts/bench_rectangle_visible_v70.py','tests/test_rectangle_cpu_protocol_v70.py',PROTOCOL)))
BENCHMARK=dict(workers=16,ordinary_decisions_per_worker=2048,seed_first=18000,action_repeat=4,
    branch_every=128,episode_limit=108000,max_charged_frames_per_worker=100000,
    sticky_probability=.25,experiment_fraction=.2,backend='native')
WIRING_CASES=[dict(seed=19376,episode_limit=300,decisions=128,engines=['original_native','rectangle_native']),
              dict(seed=19377,episode_limit=509,decisions=129,engines=['original_native','rectangle_native']),
              dict(seed=19378,episode_limit=108000,decisions=1,engines=['rectangle_native_spawn'])]


def _checked_review(path,expected=None):
    actual=sha(ROOT/path)
    if expected is not None and actual!=expected:raise ValueError('Pinned independent prerequisite review changed')
    review=json.loads((ROOT/path).read_text(encoding='utf-8'))
    if review.get('passed') is not True:raise ValueError('Independent prerequisite must pass')
    evidence={path:actual}
    for field in ('source_sha256','evidence_sha256'):
        mapping=review.get(field)
        if not isinstance(mapping,dict) or not mapping:raise ValueError('Incomplete prerequisite evidence')
        for p,h in mapping.items():
            if sha(ROOT/p)!=h:raise ValueError('Reviewed prerequisite source/evidence changed')
            if p in evidence and evidence[p]!=h:raise ValueError('Conflicting prerequisite fingerprint')
            evidence[p]=h
    for p in evidence:require_committed(p)
    return review,evidence


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol.get('kind')!='prospective_mspacman_rectangle_groups_cpu_v0'
            or protocol.get('game')!='ms_pacman' or protocol.get('benchmark')!=BENCHMARK
            or protocol.get('wiring')!=WIRING_CASES or protocol.get('agent_training_allowed') is not False
            or protocol.get('gate_result') is not None):
        raise ValueError('Unexpected prospective rectangle CPU protocol')
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or
            any(os.environ.get(name)!='1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))):
        raise ValueError('Declare one CPU thread per numerical runtime')
    cpu,evidence=_checked_review(CPU_REVIEW,CPU_REVIEW_SHA)
    rectangle,more=_checked_review(GROUP_REVIEW,GROUP_REVIEW_SHA)
    adapter,adapter_evidence=_checked_review(ADAPTER_REVIEW)
    for mapping in (more,adapter_evidence):
        for p,h in mapping.items():
            if p in evidence and evidence[p]!=h:raise ValueError('Prerequisite reviews disagree')
            evidence[p]=h
    adapter_bound={**adapter['evidence_sha256'],**adapter['source_sha256']}
    if any(adapter_bound.get(p)!=sha(ROOT/p) for p in ADAPTER_SOURCES):
        raise ValueError('Current complete native adapter must be independently reviewed')
    offline=json.loads(grouping_audit.OUTPUT.read_text(encoding='utf-8'))
    offline_path=grouping_audit.OUTPUT.relative_to(ROOT).as_posix()
    if (sha(grouping_audit.OUTPUT)!=GROUP_RESULT_SHA or offline.get('passed') is not True
            or offline.get('kind')!='offline_rectangle_groups_equivalence'
            or rectangle['source_sha256']!=offline.get('source_sha256')
            or rectangle['evidence_sha256'].get(offline_path)!=GROUP_RESULT_SHA
            or set(offline['collections'])!={'ms_pacman','asteroids'}
            or any((r['raw_frames'],r['episodes'],r['paired_groups'],r['complete_ordered_groups_exact'])
                   !=(8192,5,8187,True) for r in offline['collections'].values())):
        raise ValueError('Exact complete independently reviewed offline grouping result required')
    baseline=json.loads(BASELINE.read_text(encoding='utf-8'))
    if (cpu['source_sha256']!=original.source_hashes()
            or cpu.get('result_sha256',{}).get('native')!=sha(BASELINE)
            or baseline.get('benchmark')!=original.BENCHMARK or baseline.get('backend')!='native'
            or baseline.get('worker_reports_exactly_equal') is not True
            or baseline.get('runtime')!=runtime() or len(baseline.get('worker_reports',[]))!=16
            or any(r.get('failure') is not None or r.get('decisions')!=2048 for r in baseline['worker_reports'])):
        raise ValueError('Complete original chain, rectangle equivalence, source closure and runtime must agree')
    source={p:sha(ROOT/p) for p in SOURCE_PATHS}
    for reviewed in (cpu,rectangle,adapter):
        for p,h in reviewed['source_sha256'].items():
            if p in source and source[p]!=h:raise ValueError('Reviewed source closure conflicts')
            source[p]=h
    for p in source:require_committed(p)
    return dict(source_sha256=source,evidence_sha256=evidence,runtime=runtime(),
                native_build_manifest_sha256=original.require_native_build(),
                baseline_result_sha256=sha(BASELINE),rectangle_equivalence_review_sha256=GROUP_REVIEW_SHA,
                native_adapter_review_sha256=sha(ROOT/ADAPTER_REVIEW),
                original_cpu_review_sha256=CPU_REVIEW_SHA)


def preflight():
    proof=proofs()
    review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='rectangle_mspacman_cpu_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Current-source independent preflight required before any rectangle native construction')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


def _accounting_known(rows):
    return all(row.get('accounting_complete') is not False and row.get('failure') is None for row in rows)


def _attempt_path(output):return output.with_name(output.stem+'_attempt.json')


def _begin(output):
    attempt=_attempt_path(output)
    if output.exists() or attempt.exists():raise ValueError('Preserve every rectangle CPU result and attempt; no retry')
    proof=preflight()
    return attempt,proof


def _config(seed,limit,cap=20000):
    return EmulatorConfig(game='ms_pacman',seed=seed,frame_skip=4,mode='lab',max_episode_frames=limit,
                          max_charged_frames=cap,sticky_probability=.25,experiment_fraction=.2)


def _same_outputs(left,right):
    if len(left)!=2 or len(right)!=2 or any(not np.array_equal(a,b) for a,b in zip(left,right,strict=True)):
        raise ValueError('Rectangle chain changed ordinary RGB or reference boxes')


def wiring():
    attempt,proof=_begin(WIRING)
    FOLDER.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',started_utc=utc(),**proof)
    write_new(attempt,record)
    objects=[];pool=None;constructing=None;private_report=None
    known_reports={};report_errors={}
    def create(name,cls,config):
        nonlocal constructing
        constructing=name
        obj=cls(config,reference_backend='native',native_build=original.NATIVE_BUILD)
        objects.append((name,obj));constructing=None
        return obj
    def read_report(name,obj):
        if name in known_reports:return known_reports[name]
        if name in report_errors:raise ValueError(report_errors[name])
        try:known_reports[name]=obj.report()
        except BaseException as error:
            report_errors[name]=f'{type(error).__name__}: {error}'
            raise
        return known_reports[name]
    def reports():
        rows=[]
        for name,obj in objects:
            try:rows.append(dict(name=name,report=read_report(name,obj)))
            except BaseException as error:rows.append(dict(name=name,accounting_error=str(error)))
        if private_report is not None:rows.append(dict(name='rectangle_spawn',report=private_report))
        elif pool is not None:
            try:rows.append(dict(name='rectangle_spawn',report=read_report('rectangle_spawn',pool)))
            except BaseException as error:rows.append(dict(name='rectangle_spawn',accounting_error=str(error)))
        return rows
    try:
        for case in WIRING_CASES[:2]:
            config=_config(case['seed'],case['episode_limit'])
            left=create(f"original_{case['seed']}",VisibleReferenceEngine,config)
            right=create(f"rectangle_{case['seed']}",RectangleVisibleEngine,config)
            for step in range(case['decisions']):
                action=(step//4)%18
                _same_outputs(left.step(action),right.step(action))
            left_report=read_report(f"original_{case['seed']}",left)
            right_report=read_report(f"rectangle_{case['seed']}",right)
            if left_report!=right_report or not _same_history(left.engine._ale,right.engine._ale):
                raise ValueError('Complete rectangle/native reports or three-frame native histories differ')
            report=right_report
            if report['accounting']['cold_loads']!=2 or report['branch_pairs']!=1:
                raise ValueError('Both paired wiring cases require a cold transition and one paid pair')
            if case['seed']==19377 and (report['decoded_branch_frames']!=0
                    or report['ordinary_raw_frames']!=513 or right.engine._ale.frame_numbers!=(2,3,4)):
                raise ValueError('Terminal no-act branch and next cold history must remain exact')
        constructing='rectangle_spawn'
        pool=RectangleVisiblePool([_config(19378,108000)],reference_backend='native',native_build=original.NATIVE_BUILD)
        constructing=None
        for forbidden in ('getRAM','_capture','clone','restore','branch_pair','lives','reward','decode_visible_ram'):
            try:pool._one(0,forbidden)
            except ValueError as error:
                if 'Unknown' not in str(error):raise
            else:raise ValueError('Oracle/state RPC unexpectedly available')
        rgb,boxes=pool.step([3])[0]
        if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8 or boxes.ndim!=2 or boxes.shape[1]!=4:
            raise ValueError('Malformed rectangle reference observation')
        private_report=read_report('rectangle_spawn',pool);processes=list(pool._processes)
        pool.close();pool=None
        if any(p.is_alive() for p in processes):raise ValueError('Rectangle worker did not exit')
        rows=reports()
        if (len(rows)!=5 or any('accounting_error' in r for r in rows)
                or not _accounting_known([r['report'] for r in rows])):
            raise ValueError('All five complete engine costs required')
        if preflight()!=proof:raise ValueError('Sources/proofs changed during rectangle native wiring')
        result=dict(kind='rectangle_mspacman_cpu_wiring',passed=True,created_utc=utc(),**proof,
            wiring=WIRING_CASES,engine_reports=rows,new_charged_frames=sum(r['report']['accounting']['charged_frames'] for r in rows),
            agent_training_allowed=False,gate_result=None)
        write_new(WIRING,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(WIRING))
    except BaseException as error:
        rows=reports()
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),
            partial_reports=rows,known_charged_frames=sum(r['report']['accounting']['charged_frames'] for r in rows if 'report' in r),
            constructor_without_return=constructing,accounting_complete=constructing is None and all('report' in r for r in rows) and _accounting_known([r['report'] for r in rows]),no_retry=True)
        raise
    finally:
        try:
            if pool is not None:pool.close()
        finally:attempt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


def require_wiring(proof):
    result=json.loads(WIRING.read_text(encoding='utf-8'))
    review=json.loads(WIRING_REVIEW.read_text(encoding='utf-8'))
    attempt=json.loads(_attempt_path(WIRING).read_text(encoding='utf-8'))
    if (result.get('kind')!='rectangle_mspacman_cpu_wiring' or result.get('passed') is not True
            or any(result.get(k)!=v for k,v in proof.items()) or result.get('wiring')!=WIRING_CASES
            or review.get('kind')!='rectangle_mspacman_cpu_wiring_independent_review'
            or review.get('passed') is not True or review.get('wiring_result_sha256')!=sha(WIRING)
            or review.get('source_sha256')!=proof['source_sha256']
            or review.get('preflight_review_sha256')!=proof['preflight_review_sha256']
            or attempt.get('status')!='complete' or attempt.get('result_sha256')!=sha(WIRING)
            or review.get('wiring_attempt_sha256')!=sha(_attempt_path(WIRING))):
        raise ValueError('Completed wiring and its independent review required before benchmark')
    for path in (WIRING,WIRING_REVIEW,_attempt_path(WIRING)):require_committed(path.relative_to(ROOT).as_posix())
    return dict(wiring_result_sha256=sha(WIRING),wiring_review_sha256=sha(WIRING_REVIEW),
                wiring_attempt_sha256=sha(_attempt_path(WIRING)))


def benchmark():
    attempt,proof=_begin(OUTPUT)
    wiring_proof=require_wiring(proof)
    baseline=json.loads(BASELINE.read_text(encoding='utf-8'))
    record=dict(status='started',started_utc=utc(),**proof,**wiring_proof)
    write_new(attempt,record)
    pool=None;known={};errors={};reports=None
    started=time.perf_counter()
    try:
        pool=RectangleVisiblePool([_config(18000+i,108000,100000) for i in range(16)],
                               reference_backend='native',native_build=original.NATIVE_BUILD)
        startup=time.perf_counter()-started;tick=time.perf_counter()
        for step in range(2048):
            outputs=pool.step([(step//4+i)%18 for i in range(16)])
            if len(outputs)!=16 or any(rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8 or boxes.ndim!=2
                                      or boxes.shape[1]!=4 for rgb,boxes in outputs):
                raise ValueError('Malformed rectangle reference worker batch')
        work=time.perf_counter()-tick
        for i in range(16):
            try:known[i]=pool.report(i)
            except BaseException as error:errors[i]=str(error);raise
        reports=[known[i] for i in range(16)]
        if not _accounting_known(reports) or any(r.get('decisions')!=2048 for r in reports):
            raise ValueError('Every rectangle worker must finish the fixed trajectory')
        processes=list(pool._processes);pool.close();pool=None
        if any(p.is_alive() for p in processes):raise ValueError('Rectangle workers did not all exit')
        if preflight()!=proof or require_wiring(proof)!=wiring_proof:
            raise ValueError('Rectangle CPU sources, prerequisites or wiring changed during paid work')
        equal=reports==baseline['worker_reports']
        result=dict(kind='rectangle_mspacman_native_cpu_benchmark',created_utc=utc(),**proof,**wiring_proof,
            benchmark=BENCHMARK,startup_seconds=startup,work_seconds=work,ordinary_agent_steps_per_second=32768/work,
            ordinary_raw_frames_per_second=sum(r['ordinary_raw_frames'] for r in reports)/work,
            emulated_frames_in_work=sum(r['accounting']['emulated_frames']-132 for r in reports),
            charged_frames=sum(r['accounting']['charged_frames'] for r in reports),worker_reports=reports,
            worker_reports_exactly_equal=equal,engineering_equivalence_pass=equal,
            gpu_used=False,agent_training_allowed=False,gate_result=None,
            limitations=['Only external grouping uses the stateless rectangle graph; original decoder and native build preserved, no cache or gather combination.',
                'Includes emulator/IPC/reference/judge/movement/grouping/scoring/digests/branches/restores.',
                'Old baseline and this run occur under different concurrent ARC3 load; no controlled universal speedup.',
                'Not a learner, gate0 or new independent raw reference calibration.'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT),engineering_equivalence_pass=equal)
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),no_retry=True)
        if pool is not None:
            for i in range(16):
                if i not in known and i not in errors:
                    try:known[i]=pool.report(i)
                    except BaseException as error:errors[i]=str(error)
        record.update(partial_reports=[dict(worker=i,report=r) for i,r in sorted(known.items())],
            report_errors=errors,known_charged_frames=sum(r['accounting']['charged_frames'] for r in known.values()),
            accounting_complete=len(known)==16 and _accounting_known(known.values()),
            unknown_cost_note=None if len(known)==16 and _accounting_known(known.values()) else 'Missing or failed worker accounting is a known lower bound only; no retry')
        raise
    finally:
        try:
            if pool is not None:pool.close()
        finally:attempt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('wiring','benchmark'))
    args=parser.parse_args()
    result=(wiring if args.mode=='wiring' else benchmark)()
    print(json.dumps({k:result[k] for k in ('kind','new_charged_frames','charged_frames',
        'ordinary_agent_steps_per_second','engineering_equivalence_pass') if k in result},indent=2))
