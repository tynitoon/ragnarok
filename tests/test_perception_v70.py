"""Independent adversarial checks of measurement and reference-state semantics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ragnarok.evaluation.pong_v70 import ObjectLabel, decode_pong_ram, screen_calibration_boxes
from ragnarok.evaluation.fixtures_v70 import fixture_frames
from ragnarok.evaluation.perception_v70 import PerceptionScorer, match_centers
from scripts.ref_extract_v70 import ReferenceExtractor


def test_gated_assignment_maximizes_number_of_matches_before_distance():
    assert set(match_centers([[0, 0], [3, 0]], [[0, 0], [0, 3.5]], tolerance=4)) == {(0, 1), (1, 0)}
    assert match_centers([[0, 0]], [[4, 0]], tolerance=4) == [(0, 0)]
    assert match_centers([[0, 0]], [[4.00001, 0]], tolerance=4) == []


def test_duplicate_detections_cannot_create_extra_true_positives():
    scorer = PerceptionScorer()
    obj = ObjectLabel("ball", (50, 60, 2, 4))
    scorer.add([obj], [obj.box, obj.box], frame=0, episode=0)
    report = scorer.report()
    assert report["precision_visible"] == .5
    assert report["recall_moving"] is None
    assert report["eligible_moving"] == 0


def test_motion_uses_history_not_net_displacement_and_clears_at_reset():
    scorer = PerceptionScorer()
    for frame, x in enumerate((50, 51, 50)):
        obj = ObjectLabel("ball", (x, 60, 2, 4))
        scorer.add([obj], [obj.box], frame=frame, episode=0)
    assert scorer.report()["eligible_moving"] == 2
    scorer.add([ObjectLabel("ball", (90, 60, 2, 4))], [[90, 60, 2, 4]], frame=3, episode=1)
    assert scorer.report()["eligible_moving"] == 2
    with pytest.raises(ValueError, match="consecutive"):
        scorer.add([], np.empty((0, 4)), frame=5, episode=1)


def test_stationary_objects_match_for_precision_and_hud_is_excluded():
    scorer = PerceptionScorer()
    scorer.add([ObjectLabel("player", (140, 60, 4, 16))], [[140, 60, 4, 16], [40, 5, 10, 15]], frame=0, episode=0)
    assert scorer.report()["precision_visible"] == 1
    assert scorer.report()["predictions"] == 1
    empty = PerceptionScorer().report()
    assert empty["recall_moving"] is empty["precision_visible"] is None


@pytest.mark.parametrize("boxes", [[[0, 0, -1, 3]], [[0, 0, 0, 3]], [[float('nan'), 0, 1, 1]]])
def test_bad_predictions_do_not_change_score(boxes):
    scorer = PerceptionScorer()
    before = scorer.report()
    with pytest.raises(ValueError):
        scorer.add([], boxes, frame=0, episode=0)
    assert before == scorer.report()


def test_ram_clipping_quantisation_and_unsigned_subtraction():
    ram = np.zeros(128, dtype=np.uint8)
    ram[51], ram[50], ram[49], ram[54] = 43, 188, 139, 46
    objects = {o.kind: o.box for o in decode_pong_ram(ram)}
    assert objects == {"player": (140, 34, 4, 12), "enemy": (16, 173, 4, 16), "ball": (90, 34, 2, 2)}
    ram[54] = 44
    assert "ball" not in {o.kind for o in decode_pong_ram(ram)}
    ram[:] = 0
    assert decode_pong_ram(ram) == []
    with pytest.raises(ValueError):
        decode_pong_ram(ram.astype(np.int64))


def test_calibration_extent_survives_ball_occlusion_without_using_ram():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[60:76, 16:20] = (213, 130, 74)
    rgb[64:68, 18:20] = (236, 236, 236)
    rgb[5:15, 30:40] = (213, 130, 74)  # HUD has no effect on sprite extent
    objects = {o.kind: o.box for o in screen_calibration_boxes(rgb)}
    assert objects == {"enemy": (16, 60, 4, 16), "ball": (18, 64, 2, 4)}


def test_reference_queries_are_read_only_and_do_not_learn_counterfactual_pixels():
    ref = ReferenceExtractor()
    black = np.zeros((10, 12, 3), dtype=np.uint8)
    ref.update(black)
    intervention = black.copy()
    intervention[4:6, 5:7] = (12, 34, 56)
    assert ref.extract(intervention).tolist() == [[5, 4, 2, 2]]
    for _ in range(8):
        ref.extract(intervention)
    assert ref.frames_seen == 1
    assert ref.extract(black).shape == (0, 4)
    assert ref.extract(intervention).tolist() == [[5, 4, 2, 2]]


def test_reference_mode_is_causal_with_stable_ties_and_explicit_reset():
    ref = ReferenceExtractor()
    first = np.full((5, 5, 3), 200, dtype=np.uint8)
    second = np.full((5, 5, 3), 10, dtype=np.uint8)
    assert ref.observe(first).shape == (0, 4)
    assert ref.observe(second).tolist() == [[0, 0, 5, 5]]  # tie retains first
    assert ref.observe(second).shape == (0, 4)
    ref.reset()
    assert ref.frames_seen == 0
    with pytest.raises(ValueError, match="background"):
        ref.extract(first)


def test_reference_label_invariance_under_injective_palette_permutation():
    a, b = ReferenceExtractor(), ReferenceExtractor()
    for t in range(16):
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        frame[3:5, t % 8:t % 8 + 2] = (255, 255, 255)
        transformed = 255 - frame
        assert np.array_equal(a.observe(frame), b.observe(transformed))


def test_fixture_reader_checks_tampering_and_real_frame_gaps(tmp_path):
    shard = tmp_path / "part.npz"
    def write(counts):
        np.savez_compressed(shard, rgb=np.zeros((2,210,160,3), dtype=np.uint8),
                            ram=np.zeros((2,128), dtype=np.uint8), episode=np.zeros(2,dtype=np.int32),
                            action=np.zeros(2,dtype=np.uint8), charged_frame=np.array(counts,dtype=np.int64))
        manifest = dict(kind="scorer_only_development_fixture", agent_training_allowed=False,
                        game="pong", mode=0, raw_every_real_frame=True, frames=2,
                        shards=[dict(path=shard.name,frames=2,sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write([73,74])
    assert len(list(fixture_frames(tmp_path))) == 2
    write([73,75])
    with pytest.raises(ValueError, match="contiguous"):
        list(fixture_frames(tmp_path))
    with shard.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        list(fixture_frames(tmp_path))
