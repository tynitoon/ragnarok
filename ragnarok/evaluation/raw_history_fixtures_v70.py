"""Scorer-only raw frames with the actual adjacent RAM even at a cold first act."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS,CONVENTION

KIND='scorer_only_cold_raw_history_fixture'
KEYS=('rgb','ram','ram_previous','episode','action','charged_frame','frame_in_episode','ram_frame_numbers')


def raw_history_fixture_frames(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('kind')!=KIND or manifest.get('agent_training_allowed') is not False
            or manifest.get('reset_convention')!=CONVENTION or manifest.get('game') not in ROM_SPECS
            or manifest.get('raw_every_real_frame') is not True):
        raise ValueError('Not a declared scorer-only raw history fixture')
    bootstrap=ROM_SPECS[manifest['game']][1]
    seen,previous_episode,previous_charge,previous_local=0,0,bootstrap,0
    previous_ram=None;paths=set()
    for shard in manifest['shards']:
        name=shard['path'];n=shard['frames']
        if Path(name).name!=name or name in paths or type(n) is not int or not 1<=n<=256:
            raise ValueError('Unique local shards with bounded frame counts required')
        paths.add(name);path=directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shard['sha256']:
            raise ValueError('Raw history shard hash mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(KEYS):raise ValueError('Unexpected raw history keys')
            data={k:arrays[k] for k in KEYS}
        expected={'rgb':((n,210,160,3),np.uint8),'ram':((n,128),np.uint8),
            'ram_previous':((n,128),np.uint8),'episode':((n,),np.int32),
            'action':((n,),np.uint8),'charged_frame':((n,),np.int64),
            'frame_in_episode':((n,),np.int64),'ram_frame_numbers':((n,2),np.int64)}
        if any(data[k].shape!=shape or data[k].dtype!=dtype for k,(shape,dtype) in expected.items()):
            raise ValueError('Malformed raw history arrays')
        if np.any(data['action']>=18):raise ValueError('Invalid raw history button')
        for i in range(n):
            row={k:data[k][i] for k in KEYS}
            episode,charge,local=(int(row[k]) for k in ('episode','charged_frame','frame_in_episode'))
            jump=episode-previous_episode
            if (jump not in (0,1) or (seen==0 and jump!=0)
                    or charge-previous_charge!=1+bootstrap*jump
                    or local!=(1 if jump else previous_local+1)
                    or tuple(row['ram_frame_numbers'])!=(local-1,local)):
                raise ValueError('Noncontiguous raw frame, RAM frame numbers or cold charge')
            if previous_ram is not None and not jump and not np.array_equal(row['ram_previous'],previous_ram):
                raise ValueError('Raw previous RAM differs from preceding current RAM')
            seen+=1;previous_episode,previous_charge,previous_local=episode,charge,local
            previous_ram=row['ram'].copy()
            row['index']=seen
            yield row
    ledger=manifest['accounting']
    if (seen!=manifest['frames'] or previous_charge!=ledger['charged_frames']
            or ledger['cold_loads']!=previous_episode+1 or manifest['episode_count']!=previous_episode+1
            or ledger['experiment_frames'] or ledger['restores']
            or ledger['format_detection_emulated_frames']!=60*(previous_episode+1)):
        raise ValueError('Raw history length or final ledger is inconsistent')
