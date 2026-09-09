"""Single committed reserved temporal validation for ms_pacman; no retry or seed substitution."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageDraw
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap
from ragnarok.evaluation.triple_pooled_fixtures_v70 import KIND, KEYS, RGB_KEYS, RAM_KEYS, INDEX_KEYS, triple_pooled_fixture_frames
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import KINDS, REVISION, pooled_mspacman_ram_objects, MsPacmanPooledPixelJudge
from scripts.mspacman_raw_prerequisite_v70 import require_raw_calibration
from scripts.capture_mspacman_pooled_v70 import SOURCE_PATHS as CAPTURE_SOURCES, SCHEDULE
from scripts.calibrate_mspacman_pooled_eyes_v70 import SOURCE_PATHS as DEVELOPMENT_SOURCES
from scripts.calibrate_breakout_v70 import compare_boxes
from scripts.temporal_validation_seal_v70 import ROOT, sha, git, utc, runtime, write_new, create_seal, verify_seal
from scripts.mspacman_coverage_controller_v70 import CoverageController

BASE=ROOT/'craft_v6_out/v70_mspacman_calibration'
PROTOCOL='docs/ARC4_MSPACMAN_POOLED_RESERVED_PROTOCOL.json'
DESTINATION=BASE/'pooled_reserved_v2'
ATTEMPT=BASE/'pooled_reserved_v2_capture_attempt.json'
RESULT=BASE/'pooled_reserved_geometry_v2.json'
SOURCE_PATHS=tuple(dict.fromkeys((*DEVELOPMENT_SOURCES,PROTOCOL,
    'scripts/temporal_validation_seal_v70.py','scripts/validate_mspacman_pooled_v70.py',
    'tests/test_temporal_validation_seal_v70.py','tests/test_mspacman_pooled_reserved_v70.py')))
SPEC=dict(game='ms_pacman',development_seed=17421,reserved_seed=17422,kinds=KINDS,revision=REVISION,
    protocol=PROTOCOL,source_paths=SOURCE_PATHS,development_sources=DEVELOPMENT_SOURCES,capture_sources=CAPTURE_SOURCES,
    development_fixture='craft_v6_out/v70_mspacman_calibration/pooled_development_v0',
    development_result='craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v2.json',
    development_capture_attempt='craft_v6_out/v70_mspacman_calibration/pooled_development_v0_attempt.json',
    development_score_attempt='craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v2_attempt.json',
    development_review='craft_v6_out/v70_mspacman_calibration/pooled_development_result_review_v2.json',
    preflight_review='craft_v6_out/v70_mspacman_calibration/pooled_reserved_preflight_review_v2.json',
    seal='craft_v6_out/v70_mspacman_calibration/pooled_reserved_seal_v2.json')
SPEC['metadata_paths']=(SPEC['development_fixture']+'/manifest.json',SPEC['development_result'],
    SPEC['development_capture_attempt'],SPEC['development_score_attempt'],SPEC['development_review'],
    SPEC['preflight_review'],'craft_v6_out/v70_temporal_instruments/pooled_v1_corrections_review.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v0_attempt.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v0.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_development_result_review_v0.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_ghost_top_row_inventory_v0.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v1.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_geometry_development_v1_attempt.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_ghost_count_inventory_v1.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_returning_eye_inventory_v1.json',
    'craft_v6_out/v70_mspacman_calibration/pooled_v2_candidate_review.json')


def check():
    return verify_seal(SPEC,require_raw_calibration)


def capture():
    if DESTINATION.exists() or ATTEMPT.exists():
        raise ValueError('Preserve every earlier attempt; no duplicate or substitute capture')
    preflight = check()  # Entire proof is read-only; no native emulator yet.
    started = time.perf_counter()
    record = dict(status='started',seed=17422,started_utc=utc(),source_commit=git('rev-parse','HEAD'),**preflight)
    write_new(ATTEMPT,record)  # A constructor failure must burn and record the attempt too.
    engine = None
    completed_decisions = 0
    try:
        DESTINATION.mkdir()
        engine = _ColdPixelEngine(EmulatorConfig(game='ms_pacman',seed=17422,frame_skip=4,
            mode='live',max_episode_frames=108000,max_charged_frames=1114244,sticky_probability=.25))
        engine.set_native_wrapper(_TripleRAMTap)
        controller = CoverageController()
        rows = {k:[] for k in KEYS}
        shards,previews = [],[]
        episode = ordinary = previous_local = 0
        resets = engine.accounting()['cold_loads']
        for index in range(1,8193):
            action = controller.action(engine.observe(raw=True))
            pooled = engine.step(action)
            before,after = (rgb.copy() for rgb in engine._frames)
            if not np.array_equal(pooled,np.maximum(before,after)):
                raise RuntimeError('Captured pool differs from actual wrapper observation')
            report = engine.accounting()
            reset = report['cold_loads'] != resets
            if reset:
                episode += 1
                resets = report['cold_loads']
            tap = engine._ale
            local = int(tap.getEpisodeFrameNumber())
            advanced = local if reset else local-previous_local
            ordinary += advanced
            if (not 1 <= advanced <= 4 or report['charged_frames'] != ordinary+132*(episode+1)
                    or report['cold_loads'] != episode+1 or report['experiment_frames'] or report['restores']
                    or tap.frame_numbers != (max(0,local-2),max(0,local-1),local)
                    or tap.cold_history_fallback != (local<2)):
                raise RuntimeError('Invalid ordinary/cold ledger or RAM history')
            previous_local = local
            values = (before,after,pooled,tap.before_previous.copy(),tap.previous.copy(),tap.current.copy(),
                      episode,local,report['charged_frames'],action,tap.frame_numbers,tap.cold_history_fallback)
            for key,value in zip(KEYS,values,strict=True):
                rows[key].append(value)
            completed_decisions = index
            if index in (1,4,16,32,64,128,256,512,1024,2048,4096,8192):
                previews.append((index,pooled.copy()))
            if len(rows['pooled'])==128 or index==8192:
                n = len(rows['pooled'])
                path = DESTINATION/f'decisions_{index-n+1:05d}_{index:05d}.npz'
                arrays = {k:np.stack(rows[k]) for k in (*RGB_KEYS,*RAM_KEYS)}
                arrays.update({k:np.asarray(rows[k],dtype=np.int64) for k in (*INDEX_KEYS,'ram_frame_numbers')})
                arrays['cold_history_fallback'] = np.asarray(rows['cold_history_fallback'],dtype=np.bool_)
                np.savez_compressed(path,**arrays)
                shards.append(dict(path=path.name,decisions=n,sha256=sha(path)))
                rows = {k:[] for k in KEYS}
            if index%2048==0:
                print(f"pooled capture {index}/8192; charged={report['charged_frames']}",flush=True)
        if check()!=preflight:
            raise RuntimeError('Prospective source/runtime/raw proof changed during capture')
        sheet = Image.new('RGB',(4*176,3*240),'#202020')
        draw = ImageDraw.Draw(sheet)
        for i,(index,rgb) in enumerate(previews):
            x,y = (i%4)*176,(i//4)*240
            draw.text((x+8,y+5),f'decision{index}',fill='white')
            sheet.paste(Image.fromarray(rgb),(x+8,y+24))
        sheet.save(DESTINATION/'fixed_pooled_frames.png')
        manifest = dict(kind=KIND,game='ms_pacman',partition='reserved',seed=17422,
            decisions=8192,action_repeat=4,mode=0,episode_limit=108000,episode_count=episode+1,
            action_schedule=SCHEDULE,sticky_probability=.25,sticky_implementation='worker-python-random-per-frame',
            reset_convention=CONVENTION,ordinary_real_frames=ordinary,accounting=engine.accounting(),
            source_commit=record['source_commit'],source_sha256=preflight['source_sha256'],runtime=preflight['runtime'],
            raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
            validation_seal_sha256=preflight['seal_sha256'],preflight_review_sha256=preflight['review_sha256'],shards=shards,created_utc=utc(),
            work_seconds=time.perf_counter()-started,wrapper_pool_verified=True,
            agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
        write_new(DESTINATION/'manifest.json',manifest)
        record.update(status='complete',manifest_sha256=sha(DESTINATION/'manifest.json'),completed_utc=utc(),
                      accounting=manifest['accounting'])
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),failed_utc=utc(),
                      completed_decisions=completed_decisions,elapsed_seconds=time.perf_counter()-started)
        if engine is not None:
            try:
                record.update(accounting=engine.accounting(),accounting_status='partial_live_ledger')
            except BaseException as ledger_error:
                record.update(accounting_status='unavailable_after_failure',ledger_error=str(ledger_error))
        else:
            record['accounting_status']='constructor_did_not_return_cost_unknown_no_retry'
        raise
    finally:
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return manifest


def new_stats():
    return dict(frames=0,presence_count_agreement=0,positive_frames=0,positive_count_agreement=0,
        ram_objects=0,pixel_objects=0,joint_matches=0,boxes_within1=0,boxes_exact=0)


def accumulate(s,a,b):
    n,within,exact = compare_boxes(a,b)
    s['frames'] += 1
    s['presence_count_agreement'] += len(a)==len(b)
    if a or b:
        s['positive_frames'] += 1
        s['positive_count_agreement'] += len(a)==len(b)
    s['ram_objects'] += len(a)
    s['pixel_objects'] += len(b)
    s['joint_matches'] += n
    s['boxes_within1'] += within
    s['boxes_exact'] += exact
    return n,within


def finish(s):
    s['presence_fraction'] = s['presence_count_agreement']/s['frames'] if s['frames'] else None
    s['positive_presence_fraction'] = s['positive_count_agreement']/s['positive_frames'] if s['positive_frames'] else None
    s['box_fraction'] = s['boxes_within1']/s['joint_matches'] if s['joint_matches'] else None
    s['passed'] = (s['joint_matches']>=50 and s['presence_fraction']>=.99
        and s['positive_presence_fraction'] is not None and s['positive_presence_fraction']>=.99
        and s['box_fraction']>=.99)


def calibrate():
    preflight = check()
    candidate_hashes = {p:sha(ROOT/p) for p in SOURCE_PATHS}
    manifest = json.loads((DESTINATION/'manifest.json').read_text(encoding='utf-8'))
    attempt = json.loads(ATTEMPT.read_text(encoding='utf-8'))
    expected = dict(game='ms_pacman',partition='reserved',seed=17422,decisions=8192,
        action_repeat=4,mode=0,episode_limit=108000,sticky_probability=.25,
        reset_convention='fixed_hardware_cold_start_v1',sticky_implementation='worker-python-random-per-frame',
        action_schedule=SCHEDULE,wrapper_pool_verified=True,source_sha256=preflight['source_sha256'],
        runtime=preflight['runtime'],validation_seal_sha256=preflight['seal_sha256'],preflight_review_sha256=preflight['review_sha256'],
        raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
        agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    if (any(manifest.get(k)!=v for k,v in expected.items()) or attempt.get('status')!='complete'
            or attempt.get('manifest_sha256')!=sha(DESTINATION/'manifest.json') or attempt.get('seed')!=17422
            or any(attempt.get(k)!=v for k,v in preflight.items())
            or attempt.get('source_commit')!=manifest.get('source_commit')
            or attempt.get('accounting')!=manifest.get('accounting')):
        raise ValueError('Outside the completed prospective development attempt')
    stats = {k:new_stats() for k in KINDS}
    examples = {k:[] for k in KINDS}
    judge = MsPacmanPooledPixelJudge()
    for row in triple_pooled_fixture_frames(DESTINATION):
        truth = pooled_mspacman_ram_objects(row['ram_before_previous'],row['ram_previous'],row['ram_current'])
        pixels = judge.step(row['raw_previous'],row['raw_current'])
        for kind,s in stats.items():
            a,b = [o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within = accumulate(s,a,b)
            if (len(a)!=len(b) or n!=within) and len(examples[kind])<12:
                examples[kind].append(dict(decision=int(row['index']),ram=[o.box for o in a],pixels=[o.box for o in b]))
        if row['index']%2048==0:
            print(f"pooled geometry {row['index']}/8192",flush=True)
    for s in stats.values():
        finish(s)
    if check()!=preflight or candidate_hashes!={p:sha(ROOT/p) for p in SOURCE_PATHS}:
        raise RuntimeError('Prospective sources or evidence changed during scoring')
    return dict(kind='mspacman_two_raw_frame_reserved_calibration',seed=17422,decoder_revision=REVISION,
        created_utc=utc(),calibration_pass=all(s['passed'] for s in stats.values()),per_type=stats,
        first_disagreements=examples,source_sha256=candidate_hashes,
        fixture_manifest_sha256=sha(DESTINATION/'manifest.json'),
        raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
        validation_seal_sha256=preflight['seal_sha256'],
        scope='Four-connected contributing source envelopes, separate five categories; empty denominators null/FAIL.',
        agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)


def run(out):
    out = Path(out)
    receipt = out.with_name(out.stem+'_attempt.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve the previous result and scoring attempt')
    record = dict(status='started',seed=17422,started_utc=utc(),output=str(out.resolve()))
    write_new(receipt,record)
    try:
        record['fixture_manifest_sha256'] = sha(DESTINATION/'manifest.json')
        result = calibrate()
        write_new(out,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(out),
                      calibration_pass=result['calibration_pass'])
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('seal','check','capture','score'),required=True)
    args=parser.parse_args()
    if args.stage=='seal':result=create_seal(SPEC,require_raw_calibration)
    elif args.stage=='check':result=check()
    elif args.stage=='capture':result=capture()
    else:result=run(RESULT)
    print(json.dumps(result if args.stage in ('seal','check') else {k:result[k] for k in
        (('seed','decisions','accounting') if args.stage=='capture' else ('calibration_pass','per_type'))},indent=2))
