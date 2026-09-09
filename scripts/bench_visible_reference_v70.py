"""Closed, prospective CPU benchmarks after both reviewed fresh reference diagnostics."""
import argparse
import json
import time
import numpy as np
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferencePool
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from scripts import visible_reference_v70 as reference
from scripts import check_fast_visible_mspacman_v70 as fast_check
from scripts.temporal_validation_seal_v70 import ROOT,sha,runtime,utc,write_new,require_committed

PROTOCOL='docs/ARC4_VISIBLE_CPU_PROTOCOL.json'
PREFLIGHT_REVIEW='craft_v6_out/v70_visible_cpu/preflight_review_v0.json'
REFERENCE_REVIEW='craft_v6_out/v70_visible_reference/result_review_v0.json'
REFERENCE_REVIEW_SHA256='cefb04be064767ded9303e2256abcf1381505657b12ec7519c7d306bbfe1a6cd'
FAST_REVIEW='craft_v6_out/v70_visible_cpu/fast_mspacman_equivalence_review_v0.json'
FAST_REVIEW_SHA256=None  # Complete ordered equivalence and independent review required before freezing.
NATIVE_BUILD=ROOT/'craft_v6_out/v70_native_reference/build_v0/build.json'
SOURCE_PATHS=tuple(dict.fromkeys((*reference.SOURCE_PATHS,*fast_check.SOURCE_PATHS,PROTOCOL,
    'scripts/bench_visible_reference_v70.py','scripts/check_visible_cpu_v70.py',
    'ragnarok/evaluation/restorable_ram_history_v70.py','ragnarok/evaluation/visible_reference_pool_v70.py',
    'ragnarok/evaluation/visible_temporal_groups_v70.py','ragnarok/evaluation/native_reference_v70.py',
    'ragnarok/native/reference_update_v70.c','scripts/build_reference_native_v70.py',
    'tests/test_restorable_ram_history_v70.py','tests/test_visible_reference_pool_v70.py',
    'tests/test_visible_temporal_groups_v70.py','tests/test_visible_cpu_v70.py')))
BENCHMARK=dict(workers=16,ordinary_decisions_per_worker=2048,seed_first=18000,action_repeat=4,
    branch_every=128,episode_limit=108000,max_charged_frames_per_worker=100000,sticky_probability=.25,
    experiment_fraction=.2,backends=['numpy','native'])


def base(game):
    if game not in reference.REGISTRY:raise ValueError('Only the two declared training games are enabled')
    folder='mspacman' if game=='ms_pacman' else 'asteroids'
    return ROOT/f'craft_v6_out/v70_{folder}_cpu'


def source_hashes():return {p:sha(ROOT/p) for p in SOURCE_PATHS}


def _reference_result(game,review):
    proof=reference.check(game)
    directory,attempt,out=reference.paths(game)
    result=json.loads(out.read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('source_sha256')!=proof['source_sha256']
            or review.get('result_sha256',{}).get(game)!=sha(out)
            or result.get('kind')!='fresh_visible_raw_reference_diagnostic'
            or result.get('source_sha256')!=proof['source_sha256']
            or result.get('temporal_proof_sha256')!=proof['temporal_proof_sha256']
            or result.get('fixture_manifest_sha256')!=sha(directory/'manifest.json')
            or result.get('metrics',{}).get('frames')!=8192):
        raise ValueError('Completed fresh reference result and independent review must agree')
    required={p.relative_to(ROOT).as_posix() for p in (out,attempt,directory/'manifest.json',
        out.with_name(out.stem+'_score_attempt.json'))}
    evidence=review.get('evidence_sha256',{})
    if not required<=set(evidence):raise ValueError('Reference review lacks required attempt/result evidence')
    if sum(1 for _ in raw_history_fixture_frames(directory))!=8192:
        raise ValueError('Every fresh raw reference frame must remain intact')
    return dict(reference_result_sha256=sha(out),temporal_result_sha256=proof['temporal_proof_sha256'])


def require_reference(game):
    base(game)
    if not isinstance(REFERENCE_REVIEW_SHA256,str) or len(REFERENCE_REVIEW_SHA256)!=64:
        raise ValueError('Both final fresh reference results must be independently reviewed before CPU work')
    review_path=ROOT/REFERENCE_REVIEW
    if sha(review_path)!=REFERENCE_REVIEW_SHA256:raise ValueError('Independent reference review changed')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if set(review.get('result_sha256',{}))!=set(reference.REGISTRY):
        raise ValueError('Independent review must contain both completed reference results')
    for p,h in review.get('evidence_sha256',{}).items():
        if sha(ROOT/p)!=h:raise ValueError('Reference evidence changed')
        require_committed(p)
    require_committed(REFERENCE_REVIEW)
    proofs={g:_reference_result(g,review) for g in reference.REGISTRY}
    return proofs[game]


def require_native_build():
    manifest=json.loads(NATIVE_BUILD.read_text(encoding='utf-8'))
    if (manifest.get('source_sha256')!=sha(ROOT/'ragnarok/native/reference_update_v70.c')
            or manifest.get('dll_sha256')!=sha(NATIVE_BUILD.parent/'reference_update_v70.dll')):
        raise ValueError('Native build source/binary provenance changed')
    require_committed(NATIVE_BUILD.relative_to(ROOT).as_posix())
    require_committed((NATIVE_BUILD.parent/'reference_update_v70.dll').relative_to(ROOT).as_posix())
    return sha(NATIVE_BUILD)


def require_fast_equivalence():
    if not isinstance(FAST_REVIEW_SHA256,str) or len(FAST_REVIEW_SHA256)!=64:
        raise ValueError('Complete independent fast source equivalence review required before CPU work')
    path=ROOT/FAST_REVIEW
    if sha(path)!=FAST_REVIEW_SHA256:raise ValueError('Fast source equivalence review changed')
    review=json.loads(path.read_text(encoding='utf-8'))
    result=json.loads(fast_check.OUTPUT.read_text(encoding='utf-8'))
    proof=fast_check.hashes()
    if (review.get('passed') is not True or result.get('passed') is not True
            or result.get('kind')!='offline_ordered_mspacman_source_equivalence'
            or review.get('result_sha256')!=sha(fast_check.OUTPUT)
            or review.get('source_sha256')!=proof['source_sha256']
            or any(result.get(k)!=v for k,v in proof.items())
            or set(result.get('collections',{}))!={'development','reserved','fresh_raw'}
            or any(result['collections'][name].get('windows')!=8192 for name in result['collections'])
            or [result['collections'][name].get('source_pairs') for name in
                ('development','reserved','fresh_raw')]!=[16384,16384,8197]):
        raise ValueError('Every archived ordered source equivalence and review must agree')
    evidence=review.get('evidence_sha256',{})
    required={**proof['evidence_sha256'],
        fast_check.OUTPUT.relative_to(ROOT).as_posix():sha(fast_check.OUTPUT),
        fast_check.OUTPUT.with_name(fast_check.OUTPUT.stem+'_attempt.json').relative_to(ROOT).as_posix():
            sha(fast_check.OUTPUT.with_name(fast_check.OUTPUT.stem+'_attempt.json'))}
    if any(evidence.get(p)!=h for p,h in required.items()):
        raise ValueError('Fast source review lacks complete equivalence evidence')
    for p,h in evidence.items():
        if sha(ROOT/p)!=h:raise ValueError('Fast source equivalence evidence changed')
        require_committed(p)
    require_committed(FAST_REVIEW)
    return dict(result_sha256=sha(fast_check.OUTPUT),review_sha256=sha(path))


def preflight(game):
    proof=require_reference(game);hashes=source_hashes()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    review=json.loads((ROOT/PREFLIGHT_REVIEW).read_text(encoding='utf-8'))
    native_hash=require_native_build()
    fast_proof=require_fast_equivalence()
    if (protocol.get('games')!=['ms_pacman','asteroids'] or protocol.get('benchmark')!=BENCHMARK
            or review.get('passed') is not True or review.get('source_sha256')!=hashes
            or review.get('runtime')!=runtime() or review.get('proof_sha256',{}).get(game)!=proof
            or review.get('native_build_manifest_sha256')!=native_hash
            or review.get('fast_source_equivalence')!=fast_proof):
        raise ValueError('Complete independent CPU preflight required before new seeds')
    for p in (*SOURCE_PATHS,PREFLIGHT_REVIEW):require_committed(p)
    return dict(source_sha256=hashes,proof_sha256=proof,runtime=runtime(),
        preflight_review_sha256=sha(ROOT/PREFLIGHT_REVIEW),native_build_manifest_sha256=native_hash,
        fast_source_equivalence=fast_proof)


def require_geometry(game,proof):
    path=base(game)/'cpu_geometry_v0.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    if (result.get('kind')!='visible_cpu_complete_geometry_equivalence' or result.get('passed') is not True
            or any(result.get(k)!=v for k,v in proof.items())):
        raise ValueError('Complete current geometry equivalence required')
    require_committed(path.relative_to(ROOT).as_posix())
    return sha(path)


def _numpy_pair(game,proof,geometry,wiring):
    path=base(game)/'cpu_chain_numpy_v0.json'
    previous=json.loads(path.read_text(encoding='utf-8'))
    if (previous.get('kind')!='visible_maxpooled_repeat4_cpu_chain_benchmark'
            or previous.get('game')!=game or previous.get('backend')!='numpy'
            or previous.get('benchmark')!=BENCHMARK or previous.get('geometry_sha256')!=geometry
            or previous.get('wiring_sha256')!=wiring or len(previous.get('worker_reports',[]))!=16
            or any(previous.get(k)!=v for k,v in proof.items())
            or any(r.get('failure') is not None or r.get('decisions')!=2048 for r in previous['worker_reports'])):
        raise ValueError('Native comparison requires the complete identical-source NumPy trajectory')
    require_committed(path.relative_to(ROOT).as_posix())
    return previous,sha(path)


def benchmark(game,backend):
    if backend not in BENCHMARK['backends']:raise ValueError('Unknown declared backend')
    folder=base(game);out=folder/f'cpu_chain_{backend}_v0.json'
    attempt=folder/f'cpu_chain_{backend}_v0_attempt.json'
    if out.exists() or attempt.exists():raise ValueError('Preserve every earlier CPU result/attempt; no retry')
    proof=preflight(game)
    geometry=require_geometry(game,proof)
    wiring_path=folder/'cpu_wiring_v0.json';wiring=json.loads(wiring_path.read_text(encoding='utf-8'))
    if (wiring.get('kind')!='visible_cpu_wiring_check' or wiring.get('passed') is not True
            or any(wiring.get(k)!=v for k,v in proof.items()) or wiring.get('geometry_sha256')!=geometry):
        raise ValueError('Complete current wiring proof required before benchmark')
    require_committed(wiring_path.relative_to(ROOT).as_posix())
    wiring_sha=sha(wiring_path)
    pair=_numpy_pair(game,proof,geometry,wiring_sha) if backend=='native' else None
    folder.mkdir(parents=True,exist_ok=True)
    record=dict(status='started',game=game,backend=backend,started_utc=utc(),**proof)
    write_new(attempt,record);pool=None;reports=None;known_reports={};report_errors={}
    started=time.perf_counter()
    try:
        configs=[EmulatorConfig(game=game,seed=18000+i,frame_skip=4,mode='lab',
            max_episode_frames=108000,max_charged_frames=100000,sticky_probability=.25,experiment_fraction=.2)
            for i in range(16)]
        pool=VisibleReferencePool(configs,branch_every=128,reference_backend=backend,
            native_build=NATIVE_BUILD if backend=='native' else None)
        startup=time.perf_counter()-started;tick=time.perf_counter()
        for step in range(2048):
            outputs=pool.step([(step//4+i)%18 for i in range(16)])
            if any(rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8 or boxes.ndim!=2 or boxes.shape[1]!=4
                    for rgb,boxes in outputs):raise ValueError('Malformed reference observation')
        work=time.perf_counter()-tick
        for i in range(16):
            try:known_reports[i]=pool.report(i)
            except BaseException as error:
                report_errors[i]=str(error)
                raise
        reports=[known_reports[i] for i in range(16)]
        if any(r.get('failure') is not None or r['decisions']!=2048 for r in reports):
            raise ValueError('Every worker must complete the fixed trajectory')
        pool.close();pool=None
        if preflight(game)!=proof:raise ValueError('CPU sources/proofs changed during benchmark')
        if sha(wiring_path)!=wiring_sha or require_geometry(game,proof)!=geometry:
            raise ValueError('CPU wiring/geometry evidence changed during benchmark')
        if pair is not None and _numpy_pair(game,proof,geometry,wiring_sha)[1]!=pair[1]:
            raise ValueError('Paired NumPy evidence changed during benchmark')
        result=dict(kind='visible_maxpooled_repeat4_cpu_chain_benchmark',game=game,backend=backend,
            created_utc=utc(),**proof,geometry_sha256=geometry,wiring_sha256=wiring_sha,
            benchmark=BENCHMARK,startup_seconds=startup,work_seconds=work,
            ordinary_agent_steps_per_second=32768/work,
            ordinary_raw_frames_per_second=sum(r['ordinary_raw_frames'] for r in reports)/work,
            emulated_frames_in_work=sum(r['accounting']['emulated_frames']-132 for r in reports),
            worker_reports=reports,charged_frames=sum(r['accounting']['charged_frames'] for r in reports),
            gpu_used=False,agent_training_allowed=False,gate_result=None,
            limitations=['Diagnostic includes RGB IPC, reference, every-real-frame judge, branches and restores.',
                'First cold0-to1 transition is outside movement eligibility; frame0 remains a pooled geometry source.',
                'ROI selected by box center; precision covers all visible categories including static pills.',
                'Render-slot center motion is not certified physical identity; corrected source judge lacks a new independent raw validation.',
                'ARC3 runs concurrently; sequential NumPy/native timings do not establish a controlled universal speedup.'])
        if pair is not None:
            result['paired_numpy_result_sha256']=pair[1]
            result['worker_reports_exactly_equal']=reports==pair[0]['worker_reports']
            result['engineering_equivalence_pass']=result['worker_reports_exactly_equal']
        write_new(out,result);record.update(status='complete',completed_utc=utc(),result_sha256=sha(out))
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error))
        if reports is not None:
            record.update(accounting_status='collected_worker_reports',worker_reports=reports,
                charged_frames=sum(r['accounting']['charged_frames'] for r in reports))
        elif pool is None:record['accounting_status']='pool_constructor_or_closed_transport_cost_unavailable_no_retry'
        else:
            for i in range(16):
                if i not in known_reports and i not in report_errors:
                    try:known_reports[i]=pool.report(i)
                    except BaseException as failed:report_errors[i]=str(failed)
            partial=[dict(worker=i,report=known_reports[i]) if i in known_reports else
                dict(worker=i,accounting_error=report_errors[i]) for i in range(16)]
            record.update(accounting_status='partial_reports_when_transport_available',partial_reports=partial,
                known_charged_frames=sum(r['accounting']['charged_frames'] for r in known_reports.values()),
                accounting_complete=len(known_reports)==16)
        raise
    finally:
        try:
            if pool is not None:pool.close()
        finally:attempt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',choices=tuple(reference.REGISTRY),required=True)
    parser.add_argument('--backend',choices=BENCHMARK['backends'],required=True)
    args=parser.parse_args();result=benchmark(args.game,args.backend)
    print(json.dumps({k:result[k] for k in ('game','backend','ordinary_agent_steps_per_second','charged_frames')},indent=2))
