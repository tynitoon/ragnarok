"""Bounded scorer-only 17511 diagnostic; no emulator/capture/reserved access."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_breakout_v70 import compare_boxes
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_player_v70 import (
    decode_player_missile_ram, screen_player_missile_objects, player_ram_pixels,
    BLUE, PLAYER_COLOUR,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'craft_v6_out/v70_asteroids_calibration'


def main():
    output = BASE / 'player_render_development_diagnostic_v1.json'
    if output.exists(): raise ValueError('Preserve prior diagnostic')
    source_paths = ('ragnarok/evaluation/asteroids_player_v70.py',
                    'ragnarok/evaluation/asteroids_v70.py',
                    'ragnarok/evaluation/cold_fixtures_v70.py',
                    'scripts/calibrate_breakout_v70.py',
                    'scripts/diagnose_asteroids_player_render_v70.py',
                    'docs/ARC4_ASTEROIDS_RAW_PROTOCOL.json')
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    stats = {k:dict(frames=0,presence_count_agreement=0,positive_frames=0,
                   positive_count_agreement=0,joint_matches=0,boxes_within1=0,
                   boxes_exact=0,ram_objects=0,pixel_objects=0)
             for k in ('player','player_missile')}
    disagreements = {k:[] for k in stats}
    player_extra_pixels = blue_missed_pixels = blue_extra_pixels = 0
    red_remaining = {}
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(BASE/'development')):
        truth = decode_player_missile_ram(ram)
        pixels = screen_player_missile_objects(rgb)
        for kind,s in stats.items():
            a,b = [o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within,exact = compare_boxes(a,b)
            s['frames'] += 1; s['presence_count_agreement'] += len(a)==len(b)
            s['positive_frames'] += bool(a or b)
            s['positive_count_agreement'] += bool(a or b) and len(a)==len(b)
            s['joint_matches'] += n; s['boxes_within1'] += within; s['boxes_exact'] += exact
            s['ram_objects'] += len(a); s['pixel_objects'] += len(b)
            if len(a)!=len(b) or within!=n:
                disagreements[kind].append(dict(zero_based_frame=index,episode=episode,
                                                ram=[o.box for o in a],rgb=[o.box for o in b]))
        mask = np.all(rgb == PLAYER_COLOUR,axis=2); mask[:16] = False
        y,x = np.nonzero(mask); red = set(zip(map(int,x),map(int,y)))
        ship = player_ram_pixels(ram)
        player_extra_pixels += len(ship-red)
        rest = red-ship
        if rest: red_remaining[str(index)] = sorted(rest)
        if not int(ram[57])&1:
            mask = np.all(rgb == BLUE,axis=2); mask[:16] = False
            y,x = np.nonzero(mask); blue = set(zip(map(int,x),map(int,y)))
            predicted = {(x+dx,y+dy) for o in truth if o.slot.startswith('missile:1:')
                         for x,y,w,h in [o.box] for dx in range(w) for dy in range(h)}
            blue_missed_pixels += len(blue-predicted); blue_extra_pixels += len(predicted-blue)
        if (index+1)%2048==0: print(f'player diagnostic {index+1}/8192',flush=True)
    for s in stats.values():
        s['presence_fraction'] = s['presence_count_agreement']/s['frames']
        s['positive_presence_fraction'] = s['positive_count_agreement']/s['positive_frames'] if s['positive_frames'] else None
        s['box_fraction'] = s['boxes_within1']/s['joint_matches'] if s['joint_matches'] else None
        s['passed'] = (s['joint_matches']>=50 and s['presence_fraction']>=.99 and
                       s['positive_presence_fraction'] is not None and s['positive_presence_fraction']>=.99 and
                       s['box_fraction'] is not None and s['box_fraction']>=.99)
    if hashes != {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}:
        raise RuntimeError('Sources changed during diagnostic; preserve output before retry')
    result = dict(kind='asteroids_player_render_development_diagnostic',seed=17511,
                  per_type=stats,all_disagreements=disagreements,
                  player_predicted_pixels_absent_from_red=player_extra_pixels,
                  blue_missile_pixel_missed=blue_missed_pixels,blue_missile_pixel_extra=blue_extra_pixels,
                  red_pixels_remaining_after_player_support=red_remaining,
                  methods=['All 8192 archived frames, strict shard SHA/accounting reader; no exclusions.',
                           'RAM renderer and stateless RGB-only glyph matcher invoked separately.',
                           'Unchanged cardinality-first matcher and category criteria; no pooling.',
                           'Ship explosion dots grouped by complete glyph; wrapped ship has one box per visible vertical band.',
                           'RGB asteroid phase identified only by non-blue asteroid colours or a blue component larger than2x3.'],
                  limitations=['Empirical development correction only; no held-out17512 result.',
                               'Asteroids category not reviewed or certified here.',
                               'Blue-only asteroid frames containing only tiny fragments could remain ambiguous on unseen states.',
                               'Two red missiles have an unexplained1-pixel vertical difference, retained within frozen tolerance.',
                               'Finite observed player y(-5..83), missile y(0..88), bank/pointer guards; unknown states fail.'],
                  source_sha256=hashes,
                  fixture_manifest_sha256=hashlib.sha256((BASE/'development/manifest.json').read_bytes()).hexdigest(),
                  palette_inventory_sha256=hashlib.sha256((BASE/'development_palette_inventory_v0.npz').read_bytes()).hexdigest(),
                  pixel_inventory_sha256=hashlib.sha256((BASE/'development_player_pixels_v0.npz').read_bytes()).hexdigest(),
                  agent_training_allowed=False,reference_training_allowed=False,gate_result=None)
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__': main()
