"""Bounded offline profile of external scorer/reference on existing17471 frames.

No native act, new ROM/seed, learner input or full-chain throughput measurement.
The four-frame consumption schedule is synthetic over the archived raw stream.
"""
from collections import deque
import cProfile
import hashlib
import json
import pstats
from time import perf_counter

import numpy as np

from ragnarok.evaluation.fast_visible_mspacman_v70 import mspacman_visible_scene
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.visible_temporal_groups_v70 import VisibleTemporalScorer,group_visible_sources
from ragnarok.evaluation.native_reference_v70 import NativeReferenceExtractor
from scripts import bench_visible_reference_v70 as bench
from scripts.temporal_validation_seal_v70 import ROOT,sha,utc,write_new,require_committed

PROTOCOL='docs/ARC4_MSPACMAN_PROFILE_PROTOCOL.json'
DIRECTORY=ROOT/'craft_v6_out/v70_mspacman_calibration/raw_reference_v0'
FOLDER=ROOT/'craft_v6_out/v70_mspacman_cpu'
OUTPUT=FOLDER/'offline_scoring_profile_v0.json'
ATTEMPT=FOLDER/'offline_scoring_profile_v0_attempt.json'


def profile_rows(rows,reference):
    rows=iter(rows)
    row=next(rows,None)
    scorer=VisibleTemporalScorer('ms_pacman')
    recent=deque(maxlen=2)
    images=deque(maxlen=2)
    profile=cProfile.Profile()
    timings=dict(decode=0.,movement_history=0.,reference=0.,grouping=0.,matching_score=0.)
    frames=decisions=episodes=mixed_actions=0
    actions=set()
    digest=hashlib.sha256()
    start=perf_counter()
    def measured(name,function,*args,**kwargs):
        begin=perf_counter()
        profile.enable()
        try:return function(*args,**kwargs)
        finally:
            profile.disable()
            timings[name]+=perf_counter()-begin
    while row is not None:
        following=next(rows,None)
        episode,local=int(row['episode']),int(row['frame_in_episode'])
        if local==1:
            episodes+=1
            recent.clear();images.clear();actions.clear()
        before=row['ram'].copy(),row['ram_previous'].copy()
        labels=measured('decode',mspacman_visible_scene,row['ram'],row['ram_previous'])
        recent.append(labels);images.append(row['rgb'])
        frames+=1
        measured('movement_history',scorer.observe_truth,labels,frame=frames,episode=episode)
        if not (np.array_equal(row['ram'],before[0]) and np.array_equal(row['ram_previous'],before[1])):
            raise ValueError('Diagnostic mutated archived RAM')
        actions.add(int(row['action']))
        boundary=following is None or int(following['episode'])!=episode or local%4==0
        if boundary:
            if len(images)!=2:
                raise ValueError('Actual two-frame RGB history required, no invented cold image')
            pooled=np.maximum(images[0],images[1])
            boxes=measured('reference',reference.observe,pooled)
            groups=measured('grouping',group_visible_sources,[*recent[0],*recent[1]],game='ms_pacman')
            measured('matching_score',scorer.score,boxes,frame_truths=groups)
            decisions+=1
            mixed_actions+=len(actions)>1
            actions.clear()
            digest.update(pooled.tobytes());digest.update(boxes.astype('<i4').tobytes())
        row=following
    elapsed=perf_counter()-start
    if (frames,episodes)!=(8192,5): raise ValueError('Complete predeclared17471 archive required')
    functions=[]
    for (filename,line,function),(primitive,total,self_seconds,cumulative,_) in pstats.Stats(profile).stats.items():
        functions.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,
                              total_calls=total,self_seconds=self_seconds,cumulative_seconds=cumulative))
    return dict(raw_frames=frames,archived_episodes=episodes,synthetic_decisions=decisions,
        mixed_action_groups=mixed_actions,instrumented_component_seconds=timings,
        loop_wall_seconds=elapsed,profile_functions=sorted(functions,key=lambda r:-r['self_seconds']),
        reference_frames_seen=reference.frames_seen,perception=scorer.report(),
        ordered_rgb_boxes_sha256=digest.hexdigest())


def run():
    if OUTPUT.exists() or ATTEMPT.exists():raise ValueError('Preserve previous diagnostic result/attempt')
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol['seed'],protocol['raw_frames'],protocol['episodes'],protocol['input'])!=(
            17471,8192,5,DIRECTORY.relative_to(ROOT).as_posix()):
        raise ValueError('Unexpected profile input population')
    proof=bench.preflight('ms_pacman')
    sources={p:sha(ROOT/p) for p in ('scripts/profile_mspacman_scoring_v70.py',PROTOCOL)}
    manifest=json.loads((DIRECTORY/'manifest.json').read_text(encoding='utf-8'))
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in (
        DIRECTORY/'manifest.json',*(DIRECTORY/s['path'] for s in manifest['shards']))}
    for path in (*sources,*inputs):require_committed(path)
    record=dict(status='started',started_utc=utc(),source_sha256=sources,input_sha256=inputs,
        reviewed_chain_provenance=proof,new_emulated_frames=0,agent_training_allowed=False,gate_result=None)
    write_new(ATTEMPT,record)
    try:
        reference=NativeReferenceExtractor(build_manifest=bench.NATIVE_BUILD)
        result=profile_rows(raw_history_fixture_frames(DIRECTORY),reference)
        if (bench.preflight('ms_pacman')!=proof or any(sha(ROOT/p)!=s for p,s in sources.items())
                or any(sha(ROOT/p)!=s for p,s in inputs.items())):
            raise ValueError('Sources or inputs changed during profile')
        result.update(kind='offline_external_mspacman_scoring_profile',created_utc=utc(),
            source_sha256=sources,input_sha256=inputs,reviewed_chain_provenance=proof,
            new_emulated_frames=0,agent_training_allowed=False,gate_result=None,
            limitations=protocol['limitations'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error))
        raise
    finally:ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    result=run()
    print(json.dumps({k:result[k] for k in ('raw_frames','synthetic_decisions','mixed_action_groups',
                                           'instrumented_component_seconds','loop_wall_seconds')},indent=2))
