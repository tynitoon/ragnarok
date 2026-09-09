"""Fresh, bounded, external raw reference diagnostics for the last two train games."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames,KIND,KEYS
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap
from ragnarok.evaluation.visible_scene_v70 import mspacman_visible_scene, asteroids_visible_scene
from ragnarok.evaluation.scene_perception_v70 import ScenePerceptionScorer
from scripts.ref_extract_v70 import ReferenceExtractor
from scripts.mspacman_raw_prerequisite_v70 import require_raw_calibration as require_mspacman, RAW_SEALED_SOURCES as MS_SOURCES
from scripts.asteroids_raw_prerequisite_v70 import require_raw_calibration as require_asteroids, RAW_SEALED_SOURCES as AST_SOURCES
from scripts.validate_asteroids_raw_v70 import ROOT, sha, git, runtime, utc, write_new
from scripts.visible_temporal_prerequisite_v70 import require_temporal_calibration,SEALED_SOURCES

PROTOCOL='docs/ARC4_VISIBLE_REFERENCE_PROTOCOL.json'
REVIEW='craft_v6_out/v70_visible_reference/preflight_review_v0.json'
REGISTRY={
    'ms_pacman':dict(seed=17471,roi_xyxy=[0,0,160,171],mobile_categories=['player','ghost','fruit']),
    'asteroids':dict(seed=17571,roi_xyxy=[0,16,160,194],mobile_categories=['player','asteroid','player_missile']),
}
KINDS={'ms_pacman':('player','ghost','fruit','power_pill','pill'),
       'asteroids':('player','asteroid','player_missile')}
SOURCE_PATHS=tuple(dict.fromkeys((*MS_SOURCES,*AST_SOURCES,*SEALED_SOURCES,PROTOCOL,
    'scripts/visible_temporal_prerequisite_v70.py','ragnarok/evaluation/raw_history_fixtures_v70.py',
    'tests/test_raw_history_fixtures_v70.py','tests/test_visible_temporal_prerequisite_v70.py',
    'scripts/mspacman_raw_prerequisite_v70.py','scripts/asteroids_raw_prerequisite_v70.py',
    'scripts/visible_reference_v70.py','ragnarok/evaluation/visible_scene_v70.py',
    'ragnarok/evaluation/scene_perception_v70.py','ragnarok/evaluation/perception_v70.py',
    'scripts/ref_extract_v70.py','tests/test_visible_scene_v70.py','tests/test_visible_reference_v70.py')))


def paths(game):
    if game not in REGISTRY:
        raise ValueError('Only the two prospectively declared training games are enabled')
    folder='mspacman' if game=='ms_pacman' else 'asteroids'
    base=ROOT/f'craft_v6_out/v70_{folder}_calibration'
    return base/'raw_reference_v0',base/'raw_reference_v0_capture_attempt.json',base/'raw_reference_v0_result.json'


def check(game):
    paths(game)
    proof=(require_mspacman if game=='ms_pacman' else require_asteroids)()
    temporal_proof=require_temporal_calibration(game)
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol.get('games')!=REGISTRY or protocol.get('frames_per_game')!=8192
            or protocol.get('mode')!=0 or protocol.get('episode_limit')!=108000
            or protocol.get('sticky_probability')!=.25 or protocol.get('max_charged_frames_per_game')!=1089668
            or protocol.get('agent_training_allowed') is not False
            or protocol.get('reference_background_update_allowed') is not True
            or protocol.get('agent_entry_states_allowed') is not False or protocol.get('gate_result') is not None):
        raise ValueError('Outside the declared fresh reference protocol')
    hashes={p:sha(ROOT/p) for p in SOURCE_PATHS}
    review=json.loads((ROOT/REVIEW).read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('source_sha256')!=hashes
            or review.get('runtime')!=runtime() or review.get('raw_proof_sha256',{}).get(game)!=proof
            or review.get('temporal_proof_sha256',{}).get(game)!=temporal_proof):
        raise ValueError('Complete independent precollection review required')
    for p in (*SOURCE_PATHS,REVIEW):
        if git('hash-object',f'--path={p}',str(ROOT/p))!=git('rev-parse',f'HEAD:{p}'):
            raise ValueError('Commit all reference sources and review before capture')
    return dict(source_sha256=hashes,runtime=runtime(),raw_proof_sha256=proof,
                review_sha256=sha(ROOT/REVIEW),protocol=protocol,temporal_proof_sha256=temporal_proof)


def capture(game):
    directory,receipt,_=paths(game)
    if directory.exists() or receipt.exists():
        raise ValueError('Preserve all earlier captures and attempts')
    proof=check(game)
    seed=REGISTRY[game]['seed']
    record=dict(status='started',seed=seed,game=game,started_utc=utc(),source_commit=git('rev-parse','HEAD'),**proof)
    write_new(receipt,record)
    engine=None
    completed_frames=0
    started=time.perf_counter()
    try:
        directory.mkdir()
        engine=_ColdPixelEngine(EmulatorConfig(game=game,seed=seed,frame_skip=1,mode='live',
            max_episode_frames=108000,sticky_probability=.25,max_charged_frames=1089668))
        engine.set_native_wrapper(_TripleRAMTap)
        rng=random.Random(seed+100000)
        keys=KEYS
        rows={k:[] for k in keys}
        shards=[]
        episode=0
        loads=engine.accounting()['cold_loads']
        for index in range(1,8193):
            if (index-1)%16==0:
                action=rng.randrange(18)
            engine.step(action)
            report=engine.accounting()
            if report['cold_loads']!=loads:
                episode+=1;loads=report['cold_loads']
            if (report['charged_frames']!=index+132*(episode+1) or loads!=episode+1
                    or report['experiment_frames'] or report['restores']):
                raise RuntimeError('Fresh raw frame/cold charge mismatch')
            tap=engine._ale;local=int(tap.getEpisodeFrameNumber())
            if tap.frame_numbers[-2:]!=(local-1,local) or local<1:
                raise RuntimeError('Actual adjacent RAM history is required at every raw frame')
            values=(engine.observe(raw=True),tap.current.copy(),tap.previous.copy(),episode,action,
                    report['charged_frames'],local,tap.frame_numbers[-2:])
            for k,v in zip(keys,values,strict=True):rows[k].append(v)
            completed_frames=index
            if len(rows['rgb'])==256:
                path=directory/f'frames_{index-255:05d}_{index:05d}.npz'
                np.savez_compressed(path,rgb=np.stack(rows['rgb']),ram=np.stack(rows['ram']),
                    episode=np.asarray(rows['episode'],dtype=np.int32),action=np.asarray(rows['action'],dtype=np.uint8),
                    charged_frame=np.asarray(rows['charged_frame'],dtype=np.int64),
                    ram_previous=np.stack(rows['ram_previous']),
                    frame_in_episode=np.asarray(rows['frame_in_episode'],dtype=np.int64),
                    ram_frame_numbers=np.asarray(rows['ram_frame_numbers'],dtype=np.int64))
                shards.append(dict(path=path.name,frames=256,sha256=sha(path)))
                rows={k:[] for k in keys}
        if check(game)!=proof:
            raise RuntimeError('Reference capture prerequisites changed')
        manifest=dict(kind=KIND,game=game,seed=seed,action_seed=seed+100000,
            frames=8192,mode=0,action_repeat=16,episode_limit=108000,raw_every_real_frame=True,
            sticky_probability=.25,sticky_implementation='worker-python-random-per-frame',reset_convention=CONVENTION,
            episode_count=episode+1,accounting=engine.accounting(),shards=shards,
            source_commit=record['source_commit'],created_utc=utc(),elapsed_seconds=time.perf_counter()-started,
            source_sha256=proof['source_sha256'],runtime=proof['runtime'],raw_proof_sha256=proof['raw_proof_sha256'],
            temporal_proof_sha256=proof['temporal_proof_sha256'],
            review_sha256=proof['review_sha256'],agent_training_allowed=False,
            reference_background_update_allowed=True,agent_entry_states_allowed=False,gate_result=None)
        write_new(directory/'manifest.json',manifest)
        record.update(status='complete',completed_utc=utc(),manifest_sha256=sha(directory/'manifest.json'),
                      completed_frames=completed_frames,accounting=engine.accounting())
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),
                      completed_frames=completed_frames,elapsed_seconds=time.perf_counter()-started)
        if engine is None:
            record['accounting_status']='constructor_did_not_return_cost_unknown_no_retry'
        else:
            try:record.update(accounting=engine.accounting(),accounting_status='partial_live_ledger')
            except BaseException as ledger_error:
                record.update(accounting_status='unavailable_after_failure',ledger_error=str(ledger_error))
        raise
    finally:
        receipt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return manifest


def measure_frames(game,frames):
    """The reference sees only RGB; RAM history is a separate external input."""
    reference=ReferenceExtractor()
    scorer=ScenePerceptionScorer(roi=tuple(REGISTRY[game]['roi_xyxy']))
    decode=mspacman_visible_scene if game=='ms_pacman' else asteroids_visible_scene
    timing=dict(decode_seconds=0.,reference_seconds=0.,score_seconds=0.)
    started=time.perf_counter()
    for i,row in enumerate(frames):
        rgb,ram,previous=(row[k] for k in ('rgb','ram','ram_previous'))
        episode=int(row['episode'])
        tick=time.perf_counter();truth=decode(ram,previous)
        timing['decode_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();boxes=reference.observe(rgb)
        timing['reference_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();scorer.add(truth,boxes,frame=i,episode=episode)
        timing['score_seconds']+=time.perf_counter()-tick
    timing['wall_seconds']=time.perf_counter()-started
    metrics=scorer.report()
    for kind in KINDS[game]:
        metrics['per_type'].setdefault(kind,dict(visible=0,eligible=0,recalled=0,recall_moving=None))
    denominator=sum(timing[k] for k in ('decode_seconds','reference_seconds','score_seconds'))
    timing['measured_raw_frames_per_second']=metrics['frames']/denominator if denominator else None
    return dict(metrics=metrics,**timing)


def score(game):
    directory,capture_receipt,out=paths(game)
    receipt=out.with_name(out.stem+'_score_attempt.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve earlier reference result/attempt')
    record=dict(status='started',game=game,started_utc=utc())
    write_new(receipt,record)
    try:
        proof=check(game)
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        attempt=json.loads(capture_receipt.read_text(encoding='utf-8'))
        seed=REGISTRY[game]['seed']
        expected=dict(game=game,seed=seed,action_seed=seed+100000,frames=8192,mode=0,action_repeat=16,
            episode_limit=108000,raw_every_real_frame=True,sticky_probability=.25,
            sticky_implementation='worker-python-random-per-frame',reset_convention=CONVENTION,
            source_sha256=proof['source_sha256'],runtime=proof['runtime'],raw_proof_sha256=proof['raw_proof_sha256'],
            temporal_proof_sha256=proof['temporal_proof_sha256'],
            review_sha256=proof['review_sha256'],agent_training_allowed=False,
            reference_background_update_allowed=True,agent_entry_states_allowed=False,gate_result=None)
        if (any(manifest.get(k)!=v for k,v in expected.items()) or attempt.get('status')!='complete'
                or attempt.get('seed')!=seed or attempt.get('game')!=game
                or any(attempt.get(k)!=v for k,v in proof.items())
                or attempt.get('manifest_sha256')!=sha(directory/'manifest.json')
                or attempt.get('accounting')!=manifest.get('accounting')
                or attempt.get('source_commit')!=manifest.get('source_commit')):
            raise ValueError('Reference needs its own completed fresh ordinary capture')
        accounting=manifest['accounting']
        if (accounting['charged_frames']!=8192+132*accounting['cold_loads']
                or accounting['experiment_frames'] or accounting['restores']):
            raise ValueError('Fresh reference capture budget differs')
        record['fixture_manifest_sha256']=sha(directory/'manifest.json')
        result=measure_frames(game,raw_history_fixture_frames(directory))
        if result['metrics']['frames']!=8192 or check(game)!=proof:
            raise ValueError('Full declared frame sequence and unchanged proof required')
        result.update(kind='fresh_visible_raw_reference_diagnostic',game=game,seed=seed,created_utc=utc(),
            source_sha256=proof['source_sha256'],runtime=proof['runtime'],raw_proof_sha256=proof['raw_proof_sha256'],
            temporal_proof_sha256=proof['temporal_proof_sha256'],
            fixture_manifest_sha256=record['fixture_manifest_sha256'],
            agent_training_allowed=False,agent_entry_states_allowed=False,gate_result=None,
            limitations=['Raw frame diagnostic, excludes emulator/IPC/branches/learning from timing.',
                'Corrected source-frame RAM judge has not received a new independent raw validation; reference metrics are conditional on this composed judge. Historical raw and final temporal validations remain distinct.',
                'Visible render-slot center motion is not independently tracked physical translation.',
                'Static pills are precision targets but never motion eligible.',
                'Reference is RGB only, continuous across deaths, never a learned-parser teacher.',
                'Every frame and category retained; empty denominators null.'])
        write_new(out,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(out))
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',choices=tuple(REGISTRY),required=True)
    parser.add_argument('--stage',choices=('check','capture','score'),required=True)
    args=parser.parse_args()
    result={'check':check,'capture':capture,'score':score}[args.stage](args.game)
    print(json.dumps(result if args.stage=='check' else {k:result[k] for k in
        (('seed','frames','accounting') if args.stage=='capture' else ('seed','metrics','measured_raw_frames_per_second'))},indent=2))
