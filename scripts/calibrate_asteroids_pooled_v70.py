"""Development-only geometry for the declared two-RGB/three-RAM sequence."""
import argparse
import json
from pathlib import Path

from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from ragnarok.evaluation.pooled_asteroids_v70 import (
    KINDS, REVISION, pooled_asteroids_ram_objects, pooled_asteroids_screen_objects,
)
from scripts.calibrate_breakout_v70 import compare_boxes
from scripts.capture_asteroids_pooled_v70 import (
    ROOT, SOURCE_PATHS, DESTINATION, ATTEMPT, SCHEDULE, check, sha, utc, write_new,
)


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
    manifest = json.loads((DESTINATION/'manifest.json').read_text(encoding='utf-8'))
    attempt = json.loads(ATTEMPT.read_text(encoding='utf-8'))
    expected = dict(game='asteroids',partition='development',seed=17521,decisions=8192,
        action_repeat=4,mode=0,episode_limit=108000,sticky_probability=.25,
        reset_convention='fixed_hardware_cold_start_v1',sticky_implementation='worker-python-random-per-frame',
        action_schedule=SCHEDULE,action_seed=117521,wrapper_pool_verified=True,source_sha256=preflight['source_sha256'],
        runtime=preflight['runtime'],preflight_review_sha256=preflight['review_sha256'],
        raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
        agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    if (any(manifest.get(k)!=v for k,v in expected.items()) or attempt.get('status')!='complete'
            or attempt.get('manifest_sha256')!=sha(DESTINATION/'manifest.json') or attempt.get('seed')!=17521
            or any(attempt.get(k)!=v for k,v in preflight.items())
            or attempt.get('source_commit')!=manifest.get('source_commit')
            or attempt.get('accounting')!=manifest.get('accounting')):
        raise ValueError('Outside the completed prospective development attempt')
    stats = {k:new_stats() for k in KINDS}
    examples = {k:[] for k in KINDS}
    for row in triple_pooled_fixture_frames(DESTINATION):
        truth = pooled_asteroids_ram_objects(row['ram_before_previous'],row['ram_previous'],row['ram_current'])
        pixels = pooled_asteroids_screen_objects(row['raw_previous'],row['raw_current'])
        for kind,s in stats.items():
            a,b = [o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within = accumulate(s,a,b)
            if (len(a)!=len(b) or n!=within) and len(examples[kind])<12:
                examples[kind].append(dict(decision=int(row['index']),ram=[o.box for o in a],pixels=[o.box for o in b]))
        if row['index']%2048==0:
            print(f"pooled geometry {row['index']}/8192",flush=True)
    for s in stats.values():
        finish(s)
    if check()!=preflight:
        raise RuntimeError('Prospective sources or evidence changed during scoring')
    return dict(kind='asteroids_two_raw_frame_development_calibration',seed=17521,decoder_revision=REVISION,
        created_utc=utc(),calibration_pass=all(s['passed'] for s in stats.values()),per_type=stats,
        first_disagreements=examples,source_sha256={p:sha(ROOT/p) for p in SOURCE_PATHS},
        fixture_manifest_sha256=sha(DESTINATION/'manifest.json'),
        raw_calibration_proof_sha256=preflight['raw_calibration_proof_sha256'],
        scope='Four-connected contributing source envelopes, separate three categories; empty denominators null/FAIL.',
        agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)


def run(out):
    out = Path(out)
    receipt = out.with_name(out.stem+'_attempt.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve the previous result and scoring attempt')
    record = dict(status='started',seed=17521,started_utc=utc(),output=str(out.resolve()))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({k:result[k] for k in ('calibration_pass','per_type')},indent=2))
