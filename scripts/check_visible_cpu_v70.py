"""Exhaustive archived geometry, then bounded prospective paid wiring checks."""
import argparse
import json
import numpy as np
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from ragnarok.evaluation.visible_temporal_groups_v70 import group_visible_sources
from ragnarok.evaluation.visible_scene_v70 import mspacman_visible_scene,asteroids_visible_scene
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine,VisibleReferencePool
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import pooled_mspacman_ram_objects
from ragnarok.evaluation.pooled_asteroids_wrap_v70 import pooled_asteroids_ram_objects
from scripts.visible_temporal_prerequisite_v70 import RUNNERS
from scripts import bench_visible_reference_v70 as bench
from scripts.temporal_validation_seal_v70 import ROOT,sha,utc,write_new


def _same_window(row,game):
    scene,pooled=(mspacman_visible_scene,pooled_mspacman_ram_objects) if game=='ms_pacman' else (
        asteroids_visible_scene,pooled_asteroids_ram_objects)
    older,previous,current=(row[k] for k in ('ram_before_previous','ram_previous','ram_current'))
    candidate=[g.label for g in group_visible_sources(
        [*scene(previous,older),*scene(current,previous)],game=game)]
    return candidate==pooled(older,previous,current)


def geometry(game):
    folder=bench.base(game);out=folder/'cpu_geometry_v0.json'
    attempt=folder/'cpu_geometry_v0_attempt.json'
    if out.exists() or attempt.exists():raise ValueError('Preserve every geometry result/attempt; no retry')
    proof=bench.preflight(game)
    folder.mkdir(parents=True,exist_ok=True)
    write_new(attempt,record:=dict(status='started',game=game,started_utc=utc(),**proof))
    collections={}
    try:
        runner=RUNNERS[game]
        for name,directory in (('development',runner.DESTINATION.parent/'pooled_development_v0'),
                               ('reserved',runner.DESTINATION)):
            count=0
            for row in triple_pooled_fixture_frames(directory):
                if not _same_window(row,game):
                    raise ValueError(f'Source membership raster differs from sealed geometry: {name}/{row["index"]}')
                count+=1
            if count!=8192:raise ValueError('Every declared temporal window is required')
            collections[name]=dict(decisions=count,fixture_manifest_sha256=sha(directory/'manifest.json'))
        if bench.preflight(game)!=proof:raise ValueError('Sources/proofs changed during geometry check')
        result=dict(kind='visible_cpu_complete_geometry_equivalence',game=game,passed=True,
            created_utc=utc(),**proof,collections=collections,new_charged_frames=0,
            agent_training_allowed=False,gate_result=None)
        write_new(out,result);record.update(status='complete',completed_utc=utc(),result_sha256=sha(out))
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),
            completed_collections=collections,new_charged_frames=0)
        raise
    finally:attempt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


def _same_history(left,right):
    return (left.frame_numbers==right.frame_numbers and left.recorded_actions==right.recorded_actions
        and all(np.array_equal(getattr(left,k),getattr(right,k))
            for k in ('before_previous','previous','current')))


def _config(game,seed,limit):
    return EmulatorConfig(game=game,seed=seed,frame_skip=4,mode='lab',max_episode_frames=limit,
        max_charged_frames=20000,sticky_probability=.25,experiment_fraction=.2)


def wiring(game):
    folder=bench.base(game);out=folder/'cpu_wiring_v0.json';attempt=folder/'cpu_wiring_v0_attempt.json'
    if out.exists() or attempt.exists():raise ValueError('Preserve every wiring result/attempt; no retry')
    proof=bench.preflight(game);geometry_sha=bench.require_geometry(game,proof)
    record=dict(status='started',game=game,started_utc=utc(),geometry_sha256=geometry_sha,**proof)
    write_new(attempt,record)
    objects=[];pool=None;constructing=None;pool_report=None
    def create(name,factory):
        nonlocal constructing
        constructing=name
        obj=factory();objects.append((name,obj));constructing=None
        return obj
    def reports():
        rows=[]
        for name,obj in objects:
            try:
                report=obj.report() if hasattr(obj,'report') else dict(accounting=obj.accounting())
                rows.append(dict(name=name,report=report))
            except BaseException as error:rows.append(dict(name=name,accounting_error=str(error)))
        if pool_report is not None:rows.append(dict(name='private_worker',report=pool_report))
        elif pool is not None:
            try:rows.append(dict(name='private_worker',report=pool.report()))
            except BaseException as error:rows.append(dict(name='private_worker',accounting_error=str(error)))
        return rows
    try:
        config=_config(game,18176,300)
        left=create('oracle_present',lambda:VisibleReferenceEngine(config))
        right=create('oracle_removed',lambda:VisibleReferenceEngine(config))
        right._decoder=lambda now,previous:[]
        for step in range(128):
            a,b=left.step((step//4)%18),right.step((step//4)%18)
            if not all(np.array_equal(x,y) for x,y in zip(a,b,strict=True)):
                raise ValueError('Oracle removal changed RGB/reference output')
        a,b=left.report(),right.report()
        if (a['self_generated_label_sha256']!=b['self_generated_label_sha256']
                or a['ordered_reference_output_sha256']!=b['ordered_reference_output_sha256']
                or a['accounting']!=b['accounting'] or a['perception']['matched_visible']<=0
                or b['perception']['matched_visible']!=0 or left.reference.frames_seen!=128
                or right.reference.frames_seen!=128 or left._episode!=1 or right._episode!=1
                or not _same_history(left.engine._ale,right.engine._ale)):
            raise ValueError('Oracle invariance, cold episode or reference history mismatch')

        config=_config(game,18177,509)
        subject=create('terminal_subject',lambda:VisibleReferenceEngine(config))
        witness=create('terminal_cold_witness',lambda:_ColdPixelEngine(config))
        witness.set_native_wrapper(_TripleRAMTap)
        for step in range(129):
            action=(step//4)%18
            pixels,_=subject.step(action);expected=witness.step(action)
            if not np.array_equal(pixels,expected) or not _same_history(subject.engine._ale,witness._ale):
                raise ValueError('Ordinary RGB/three-RAM diverged across a branch or cold start')
        a=subject.report();wa=witness.accounting()
        if (a['branch_pairs']!=1 or a['decoded_branch_frames']!=0 or a['accounting']['restores']!=2
                or a['accounting']['cold_loads']!=2 or wa['cold_loads']!=2
                or a['ordinary_raw_frames']!=513 or subject.engine._ale.frame_numbers!=(2,3,4)
                or a['accounting']['charged_frames']!=wa['charged_frames']+120):
            raise ValueError('Terminal branch/restoration/next cold history accounting mismatch')

        constructing='private_worker'
        pool=VisibleReferencePool([_config(game,18178,108000)]);constructing=None
        for forbidden in ('getRAM','_capture','clone','restore','branch_pair','lives','reward','decode_visible_ram'):
            try:pool._one(0,forbidden)
            except ValueError as error:
                if 'Unknown' not in str(error):raise
            else:raise ValueError('Oracle/state RPC unexpectedly available')
        pixels,boxes=pool.step([3])[0]
        if pixels.dtype!=np.uint8 or pixels.shape!=(210,160,3) or boxes.ndim!=2 or boxes.shape[1]!=4:
            raise ValueError('Malformed RGB-only observation')
        pool_report=pool.report();processes=list(pool._processes)
        pool.close();pool=None
        if any(p.is_alive() for p in processes):raise ValueError('Worker cleanup failed')
        rows=reports()
        if len(rows)!=5 or any('accounting_error' in r for r in rows):raise ValueError('All five constructor costs required')
        if bench.preflight(game)!=proof:raise ValueError('Sources/proofs changed during wiring')
        if bench.require_geometry(game,proof)!=geometry_sha:
            raise ValueError('Geometry evidence changed during wiring')
        result=dict(kind='visible_cpu_wiring_check',game=game,passed=True,created_utc=utc(),**proof,
            geometry_sha256=geometry_sha,engine_reports=rows,
            new_charged_frames=sum(r['report']['accounting']['charged_frames'] for r in rows),
            checks=['Oracle removal preserves RGB, reference boxes, C, history and ledger',
                'Terminal branch pays both restores and next cold frame history agrees with independent engine',
                'Private RPC rejects oracle/state access; worker exits'],
            agent_training_allowed=False,gate_result=None)
        write_new(out,result);record.update(status='complete',completed_utc=utc(),result_sha256=sha(out),
            engine_reports=rows,new_charged_frames=result['new_charged_frames'])
    except BaseException as error:
        rows=reports()
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),
            partial_reports=rows,known_charged_frames=sum(r['report']['accounting']['charged_frames']
                for r in rows if 'report' in r),constructor_without_return=constructing,
            accounting_complete=constructing is None and all('report' in r for r in rows),no_retry=True)
        raise
    finally:
        try:
            if pool is not None:pool.close()
        finally:attempt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('geometry','wiring'))
    parser.add_argument('--game',choices=tuple(bench.reference.REGISTRY),required=True)
    args=parser.parse_args();result=(geometry if args.mode=='geometry' else wiring)(args.game)
    print(json.dumps({k:result[k] for k in ('kind','passed','new_charged_frames')},indent=2))
