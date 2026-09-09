"""RGB exact-cover witnesses and conservative ambiguity handling."""
from pathlib import Path
import json
import numpy as np

from ragnarok.evaluation.asteroids_deblend_v70 import TEMPLATES,deblend_component,screen_deblended_rocks
from ragnarok.evaluation.asteroids_occlusion_v70 import decode_occluded_rocks
from scripts.calibrate_breakout_v70 import compare_boxes

ROOT=Path(__file__).resolve().parents[1]


def frame(number):
    directory=ROOT/'craft_v6_out/v70_asteroids_calibration/development'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    with np.load(directory/manifest['shards'][(number-1)//256]['path']) as z:
        return z['rgb'][(number-1)%256],z['ram'][(number-1)%256]


def test_exact_pair_preserves_all_pixels_and_recovers_priority_from_full_origin():
    front=np.zeros((210,160),bool);back=np.zeros_like(front)
    front[60:88,40:56]=TEMPLATES[0]
    back[69:83,50:58]=TEMPLATES[2]
    pieces,reason=deblend_component(front|back)
    assert reason=='unique_pair' and len(pieces)==2
    assert np.array_equal(pieces[0],front)
    assert np.array_equal(pieces[1],back&~front)
    assert np.array_equal(pieces[0]|pieces[1],front|back)
    assert not np.any(pieces[0]&pieces[1])


def test_boundary_pixels_are_retained_without_speculative_deblend():
    mask=np.zeros((210,160),bool);mask[193,10:13]=True
    pieces,reason=deblend_component(mask)
    assert reason=='boundary_component' and len(pieces)==1
    assert np.array_equal(pieces[0],mask)


def test_observed_same_colour_contact_is_separated_without_ram_input():
    rgb,ram=frame(2113)
    pixels=screen_deblended_rocks(rgb)
    truth=decode_occluded_rocks(ram)
    assert len(pixels)==len(truth)==5
    assert compare_boxes(truth,pixels)==(5,5,5)


def test_occlusion_keeps_the_isolated_fragments():
    rgb,ram=frame(467)
    pixels=screen_deblended_rocks(rgb);truth=decode_occluded_rocks(ram)
    assert len(pixels)==len(truth)==11
    assert compare_boxes(truth,pixels)==(11,11,11)
    assert {(11,93,1,1),(3,110,1,1),(5,112,2,1)} <= {o.box for o in pixels}
