"""Synthetic intervention observations, with labels confined to this judge."""
import numpy as np
import pytest

from ragnarok.environments.stationary_intervention_v70 import (
    POSITIVE_COUNTS, collect_interventions, generate_screens, step_rgb, new_ledger)


def test_collects_both_owned_branches_and_labels_only_their_observed_difference():
    screen = np.zeros((1, 3, 5, 3), dtype=np.uint8)
    screen[0, 0, 0] = [255, 137, 0]
    screen[0, 2, 4] = [255, 89, 0]
    screen[0, 1, 1] = [0, 127, 63]
    before = screen.copy()
    arrays, counts = collect_interventions(screen, horizon=3)
    assert counts == dict(initial_screens_generated=0, synthetic_step_entries=8,
                          synthetic_steps=8, synthetic_rgb_frames=8,
                          new_emulated_frames=0, optimizer_updates=0)
    assert np.array_equal(screen, before)
    assert np.array_equal(arrays['past_pairs'][0, 0], arrays['past_pairs'][0, 1])
    observed = np.zeros((3, 5), dtype=bool)
    for neutral, acted in zip(arrays['noop_futures'][0], arrays['action_futures'][0], strict=True):
        observed |= np.any(neutral != acted, axis=-1)
        assert np.array_equal(neutral, before[0])
        assert acted[0, 0, 2] == acted[2, 4, 2] == 255
    assert np.array_equal(arrays['observed_response_envelopes'][0], observed)
    assert observed.sum() == 2 and observed[0, 0] and observed[2, 4]


def test_empty_red_support_keeps_every_owned_future_equal():
    rng = np.random.default_rng(21)
    screens = rng.integers(0, 256, size=(2, 3, 4, 3), dtype=np.uint8)
    screens[..., 0] = 0
    arrays, counts = collect_interventions(screens, horizon=8)
    assert not arrays['observed_response_envelopes'].any()
    assert np.array_equal(arrays['noop_futures'], arrays['action_futures'])
    assert counts['synthetic_steps'] == counts['synthetic_rgb_frames'] == 36


def test_target_depends_on_the_world_response_not_a_generator_mask(monkeypatch):
    from ragnarok.environments import stationary_intervention_v70 as world
    screen = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    screen[0, 0, 0, 0] = 255
    # Removing the intervention must remove supervision despite the red cue.
    monkeypatch.setattr(world, 'step_rgb', lambda rgb, action: rgb.copy())
    arrays, _ = world.collect_interventions(screen)
    assert not arrays['observed_response_envelopes'].any()


def test_synthetic_draw_is_reproducible_sparse_and_preserves_empty_cases():
    counts = (1, 4, 0, 3, 2)
    first = generate_screens(seed=17, height=5, width=7, positive_counts=counts)
    second = generate_screens(seed=17, height=5, width=7, positive_counts=counts)
    assert np.array_equal(first, second)
    assert [(rgb[..., 0] == 255).sum() for rgb in first] == list(counts)
    assert np.all(first[..., 2][first[..., 0] == 255] == 0)
    assert np.isin(first[..., 0], [0, 255]).all()
    assert POSITIVE_COUNTS == (1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2)


def test_commands_are_stationary_and_do_not_mutate_the_supplied_state():
    initial = np.array([[[255, 99, 0], [0, 201, 77]]], dtype=np.uint8)
    changed = step_rgb(initial, 1)
    assert np.array_equal(changed, step_rgb(changed, 1))
    assert np.array_equal(changed, step_rgb(changed, 0))
    assert initial[0, 0, 2] == 0 and changed[0, 0, 2] == 255
    assert np.array_equal(initial[0, 1], changed[0, 1])


def test_unadmitted_past_cannot_silently_become_a_failed_positive_control():
    red_already_blue = np.array([[[[255, 0, 255]]]], dtype=np.uint8)
    with pytest.raises(ValueError, match='initial RGB family'):
        collect_interventions(red_already_blue)
    with pytest.raises(ValueError, match='initial RGB family'):
        collect_interventions(np.array([[[[17, 0, 0]]]], dtype=np.uint8))
    with pytest.raises(ValueError, match='two fixed commands'):
        step_rgb(np.zeros((1, 1, 3), dtype=np.uint8), 2)


def test_first_return_is_preserved_if_the_second_noop_raises(monkeypatch):
    from ragnarok.environments import stationary_intervention_v70 as world
    original, calls = world.step_rgb, []
    def interrupted(rgb, action):
        calls.append(action)
        if len(calls) == 2:
            raise RuntimeError('second NOOP interrupted')
        return original(rgb, action)
    monkeypatch.setattr(world, 'step_rgb', interrupted)
    ledger = world.new_ledger()
    with pytest.raises(RuntimeError, match='second NOOP'):
        world.collect_interventions(np.zeros((1, 2, 2, 3), dtype=np.uint8), ledger=ledger)
    assert ledger['synthetic_step_entries'] == 2
    assert ledger['synthetic_steps'] == ledger['synthetic_rgb_frames'] == 1


def test_journal_failure_after_step_return_keeps_that_return_count():
    ledger = new_ledger()
    def broken_log(_tally):
        raise OSError('journal write failed')
    with pytest.raises(OSError, match='journal write'):
        collect_interventions(np.zeros((1, 2, 2, 3), dtype=np.uint8), ledger=ledger,
                              on_step_return=broken_log)
    assert ledger['synthetic_step_entries'] == ledger['synthetic_steps'] == ledger['synthetic_rgb_frames'] == 1


def test_initial_screens_are_separate_from_returned_step_frames():
    ledger = new_ledger()
    screens = generate_screens(seed=18, height=3, width=4, positive_counts=(1, 0), ledger=ledger)
    assert ledger['initial_screens_generated'] == 2
    assert ledger['synthetic_step_entries'] == ledger['synthetic_rgb_frames'] == 0
    _, returned = collect_interventions(screens, horizon=2, ledger=ledger)
    assert returned is ledger
    assert ledger['synthetic_steps'] == ledger['synthetic_rgb_frames'] == 12
    assert ledger['initial_screens_generated'] == 2
