"""Offline protocol tests on synthetic rows only, no archived arrays opened."""
import json
from time import perf_counter

import numpy as np
import pytest

from scripts import check_rectangle_groups_v70 as run
from ragnarok.evaluation.scene_perception_v70 import SceneObject


def rows():
    result = []
    for episode, length in enumerate((3, 2)):
        for local in range(1, length + 1):
            ram = np.zeros(128, dtype=np.uint8)
            ram[0] = len(result) + 1
            result.append(dict(episode=episode, frame_in_episode=local,
                               ram=ram, ram_previous=np.zeros_like(ram)))
    return result


def decode(current, previous):
    number = int(current[0])
    return [SceneObject(str(number), 'pill', (number, 1, 1, 1), False)]


def test_complete_pair_population_boundary_omission_and_alternating_order():
    calls = []
    def original(objects, *, game):
        calls.append('raster')
        return run.group_visible_sources(objects, game=game)
    def candidate(objects, *, game):
        calls.append('rectangles')
        return run.rectangle_visible_groups(objects, game=game)
    progress = {}
    result = run.compare_rows(rows(), game='ms_pacman', decoder=decode, progress=progress,
                              deadline=perf_counter() + 30, expected_frames=5, expected_episodes=2,
                              first_pair=1, original=original, candidate=candidate)
    assert result['paired_groups'] == 3
    assert result['consumed_source_objects'] == 6
    assert result['returned_groups'] == 3
    assert result['first_call_counts'] == dict(raster=1, rectangles=2)
    assert calls == ['rectangles', 'raster', 'raster', 'rectangles', 'rectangles', 'raster']
    assert progress['completed_raw_frames'] == 5 and progress['completed_pairs'] == 3


def test_first_mismatch_stops_and_retains_exact_prefix_location():
    progress = {}
    with pytest.raises(ValueError, match='group or membership mismatch'):
        run.compare_rows(rows(), game='ms_pacman', decoder=decode, progress=progress,
                         deadline=perf_counter() + 30, expected_frames=5, expected_episodes=2,
                         candidate=lambda objects, **kwargs: [])
    assert progress['completed_raw_frames'] == 2
    assert progress['completed_pairs'] == 0
    assert progress['first_mismatch'] == dict(game='ms_pacman', frame=2, episode=0, local=2)


def test_incomplete_population_and_expired_budget_are_not_passes():
    with pytest.raises(ValueError, match='Every declared frame'):
        run.compare_rows(rows()[:-1], game='ms_pacman', decoder=decode, progress={},
                         deadline=perf_counter() + 30, expected_frames=5, expected_episodes=2)
    with pytest.raises(ValueError, match='work ceiling'):
        run.compare_rows(rows(), game='ms_pacman', decoder=decode, progress={}, deadline=0)


def test_source_mutation_is_rejected():
    def mutate(objects, **kwargs):
        objects.clear()
        return []
    with pytest.raises(ValueError, match='mutated source objects'):
        run.compare_rows(rows(), game='ms_pacman', decoder=decode, progress={},
                         deadline=perf_counter() + 30, candidate=mutate)


def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ATTEMPT', tmp_path / 'attempt.json')
    monkeypatch.setattr(run, 'OUTPUT', tmp_path / 'output.json')


def test_missing_preflight_never_reads_arrays_or_starts_attempt(tmp_path, monkeypatch):
    paths(tmp_path, monkeypatch)
    def missing():
        raise ValueError('missing preflight')
    def forbidden(*args):
        raise AssertionError('No archive should be opened')
    monkeypatch.setattr(run, 'preflight', missing)
    monkeypatch.setattr(run, 'raw_history_fixture_frames', forbidden)
    with pytest.raises(ValueError, match='missing preflight'):
        run.run()
    assert not run.ATTEMPT.exists() and not run.OUTPUT.exists()


def test_failure_keeps_prefix_and_blocks_retry_without_archive(tmp_path, monkeypatch):
    paths(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'preflight', lambda: {})
    monkeypatch.setattr(run, 'COLLECTIONS', [('ms_pacman', 'synthetic_unused', 0, 'unused')])
    monkeypatch.setattr(run, 'raw_history_fixture_frames', lambda path: rows())
    def fail(data, **kwargs):
        kwargs['progress'].update(completed_raw_frames=2, completed_pairs=1)
        raise MemoryError('synthetic post-pair failure')
    monkeypatch.setattr(run, 'compare_rows', fail)
    with pytest.raises(MemoryError):
        run.run()
    record = json.loads(run.ATTEMPT.read_text())
    assert record['status'] == 'failed' and record['no_retry'] is True
    assert record['completed_raw_frames'] == 2 and record['completed_pairs'] == 1
    assert not run.OUTPUT.exists()
    with pytest.raises(ValueError, match='Preserve every'):
        run.run()
