"""One prospective repeat4 development capture, with a durable attempt marker."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap
from ragnarok.evaluation.triple_pooled_fixtures_v70 import KIND, KEYS, RGB_KEYS, RAM_KEYS, INDEX_KEYS
from scripts.mspacman_coverage_controller_v70 import CoverageController
from scripts.mspacman_raw_prerequisite_v70 import require_raw_calibration, RAW_SEALED_SOURCES
from scripts.validate_mspacman_raw_v70 import ROOT, BASE, sha, git, runtime

PROTOCOL = 'docs/ARC4_MSPACMAN_POOLED_PROTOCOL.json'
REVIEW = 'craft_v6_out/v70_mspacman_calibration/pooled_development_preflight_review_v0.json'
DESTINATION = BASE/'pooled_development_v0'
ATTEMPT = BASE/'pooled_development_v0_attempt.json'
SOURCE_PATHS = tuple(dict.fromkeys((*RAW_SEALED_SOURCES,
    'ragnarok/__init__.py','ragnarok/evaluation/__init__.py',
    'ragnarok/environments/__init__.py','scripts/__init__.py',
    'scripts/mspacman_raw_prerequisite_v70.py', PROTOCOL,
    'ragnarok/evaluation/triple_ram_tap_v70.py',
    'ragnarok/evaluation/triple_pooled_fixtures_v70.py',
    'ragnarok/evaluation/pooled_mspacman_v70.py',
    'scripts/capture_mspacman_pooled_v70.py',
    'scripts/calibrate_mspacman_pooled_v70.py',
    'tests/test_triple_ram_tap_v70.py',
    'tests/test_triple_pooled_fixtures_v70.py',
    'tests/test_mspacman_pooled_v70.py')))
SCHEDULE = 'instrumentation_rgb_maze_walker_v1_each_repeat4_decision'


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def check():
    proof = require_raw_calibration()
    protocol = json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    expected = dict(development_seed=17421,reserved_validation_seed=17422,decisions=8192,
        action_repeat=4,mode=0,episode_limit=108000,sticky_probability=.25,
        reset_convention=CONVENTION,max_charged_frames=1114244,
        agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('Outside the prospective pooled development protocol')
    hashes = {p:sha(ROOT/p) for p in SOURCE_PATHS}
    review = json.loads((ROOT/REVIEW).read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('source_sha256') != hashes
            or review.get('runtime') != runtime() or review.get('raw_calibration_proof_sha256') != proof):
        raise ValueError('Complete independent precollection review required')
    for p in (*SOURCE_PATHS,REVIEW):
        if git('hash-object',f'--path={p}',str(ROOT/p)) != git('rev-parse',f'HEAD:{p}'):
            raise ValueError(f'Commit all sources and review before emulator construction: {p}')
    return dict(protocol=protocol,source_sha256=hashes,runtime=runtime(),
                review_sha256=sha(ROOT/REVIEW),raw_calibration_proof_sha256=proof)


def capture():
    if DESTINATION.exists() or ATTEMPT.exists():
        raise ValueError('Preserve every earlier attempt; no duplicate or substitute capture')
    preflight = check()  # Entire proof is read-only; no native emulator yet.
    started = time.perf_counter()
    record = dict(status='started',seed=17421,started_utc=utc(),source_commit=git('rev-parse','HEAD'),**preflight)
    write_new(ATTEMPT,record)  # A constructor failure must burn and record the attempt too.
    engine = None
    completed_decisions = 0
    try:
        DESTINATION.mkdir()
        engine = _ColdPixelEngine(EmulatorConfig(game='ms_pacman',seed=17421,frame_skip=4,
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
        manifest = dict(kind=KIND,game='ms_pacman',partition='development',seed=17421,
            decisions=8192,action_repeat=4,mode=0,episode_limit=108000,episode_count=episode+1,
            action_schedule=SCHEDULE,sticky_probability=.25,sticky_implementation='worker-python-random-per-frame',
            reset_convention=CONVENTION,ordinary_real_frames=ordinary,accounting=engine.accounting(),
            source_commit=record['source_commit'],source_sha256=preflight['source_sha256'],runtime=preflight['runtime'],
            raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
            preflight_review_sha256=preflight['review_sha256'],shards=shards,created_utc=utc(),
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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    result = check() if args.check else capture()
    print(json.dumps(result if args.check else {k:result[k] for k in ('seed','decisions','work_seconds','accounting')},indent=2))
