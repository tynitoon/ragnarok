"""Earlier-anchor experiments: native/RGB/RNG witnesses, costs and failure stops.

Native implementation is entirely synthetic; no ALE instance or ROM is opened.
"""
from collections import deque
import pickle

import numpy as np
import pytest

from ragnarok.environments.frame_budget_v70 import BudgetExceeded
from ragnarok.environments.rgb_archive_v70 import _RGBArchiveEngine
from ragnarok.environments import rgb_archive_v70 as archive
from ragnarok.environments import rgb_branch_trace_v70 as trace
from tests.test_rgb_branch_trace_v70 import engine, PixelNative


class Saved:
    def __init__(self, value): self.value = value
    def serialize(self): return pickle.dumps(self.value, protocol=5)


class ArchiveNative(PixelNative):
    def cloneState(self, include_rng): return Saved(super().cloneState(include_rng))
    def restoreState(self, saved): super().restoreState(saved.value)


class SyntheticArchiveEngine(_RGBArchiveEngine):
    def _decode_native(self, data): return Saved(pickle.loads(data))


def create(*, credit=1200, terminal=999, entries=20_000, byte_limit=2_000_000_000):
    obj = engine(SyntheticArchiveEngine, terminal=terminal, credit=credit)
    obj._ale = ArchiveNative(terminal)
    obj._init_archive(entries, byte_limit)
    return obj


def state(obj):
    return obj._ale.cloneState(True).serialize(), obj._sticky_rng.getstate(), obj._last_action


def equal_trace(a, b):
    assert all(np.array_equal(getattr(a, key), getattr(b, key)) for key in ('raw', 'pooled', 'endpoint'))


@pytest.mark.parametrize('terminal,count', [(999, 8), (5, 1), (4, 0)])
def test_archive_pair_uses_earlier_state_and_resumes_current_pixels_rng_and_history(terminal, count):
    subject, anchor_witness, current_witness = (create(terminal=terminal) for _ in range(3))
    for obj in (subject, anchor_witness, current_witness): obj.step(1)
    token = subject.archive_save()
    saved_bytes = subject._archive[token]
    if terminal == 999:
        for obj in (subject, current_witness):
            for action in (0, 1, 0): obj.step(action)
    before_state, before_rgb = state(subject), subject.observe()
    before_ledger = subject._budget.report()
    observed = subject.archive_pair_trace(token, 1, control=0, frames=8)
    expected = anchor_witness.branch_pair_trace(1, control=0, frames=8)
    for actual, witness in zip(observed, expected, strict=True):
        equal_trace(actual, witness)
        assert len(actual.raw) == count
    assert state(subject) == before_state
    assert np.array_equal(subject.observe(), before_rgb)
    after = subject._budget.report()
    assert after['ordinary_frames'] == before_ledger['ordinary_frames']
    assert after['ordinary_agent_steps'] == before_ledger['ordinary_agent_steps']
    assert after['experiment_frames'] - before_ledger['experiment_frames'] == 2*count
    assert after['restores'] - before_ledger['restores'] == 3
    assert after['charged_frames'] - before_ledger['charged_frames'] == 180+2*count
    assert after['experiment_agent_steps'] - before_ledger['experiment_agent_steps'] == 2
    assert subject._archive[token] is saved_bytes and not subject._saves
    if terminal == 999:
        for action in (1, 0, 1):
            assert np.array_equal(subject.step(action), current_witness.step(action))
            assert state(subject) == state(current_witness)
        assert subject._budget.report()['ordinary_frames'] == current_witness._budget.report()['ordinary_frames']


def test_storing_opaque_token_does_not_open_lab_or_alias_current_pixels():
    subject = create()
    before = subject._budget.report()
    token = subject.archive_save()
    assert isinstance(token, str) and len(token) == 32
    assert subject._budget.report() == before and subject._saves == {}
    expected = subject._decode_saved(subject._archive[token])[1]
    subject._frames[-1][:] = 99
    assert not np.array_equal(subject._frames[-1], subject._decode_saved(subject._archive[token])[1][-1])
    assert np.array_equal(expected[-1], subject._decode_saved(subject._archive[token])[1][-1])
    subject.step(1)
    after = subject._budget.report()
    assert after['ordinary_frames'] == before['ordinary_frames']+4 and after['experiment_frames'] == 0


def test_three_restore_admission_refuses_before_native_load_without_mutation():
    subject = create(credit=600)  # Current-anchor136 fits, earlier-anchor196 does not.
    token = subject.archive_save()
    before_state, before = state(subject), subject._budget.report()
    with pytest.raises(BudgetExceeded): subject.archive_pair_trace(token, 1, frames=8)
    assert state(subject) == before_state and subject._budget.report() == before
    assert subject._trace_failure is None and subject._recording is None
    subject.step(1)


def test_first_trace_failure_pays_load_act_and_return_then_stops(monkeypatch):
    subject = create()
    token = subject.archive_save()
    subject.step(1)
    before_state, before = state(subject), subject._budget.charged_frames
    def broken(self, frame): raise RuntimeError('synthetic archive recorder failure')
    monkeypatch.setattr(trace._RecordingDeque, 'append', broken)
    with pytest.raises(RuntimeError, match='recorder'): subject.archive_pair_trace(token, 1)
    assert subject._budget.charged_frames == before+121
    assert subject._budget.report()['restores'] == 2 and state(subject) == before_state
    assert subject._trace_failure is not None and subject._recording is None
    for call in (subject.archive_save, lambda: subject.archive_pair_trace(token, 1), lambda: subject.step(1)):
        with pytest.raises(ValueError, match='stopped'): call()
    subject.archive_drop(token)
    assert subject.archive_report()['entries'] == 0


def test_native_restore_failure_stays_charged_and_poisoned_even_if_return_succeeds():
    subject = create()
    token = subject.archive_save()
    subject.step(1)
    before_state, before = state(subject), subject._budget.charged_frames
    native_restore = subject._ale.restoreState
    calls = []
    def fail_first(saved):
        calls.append(saved)
        if len(calls) == 1: raise RuntimeError('native archive load failure')
        native_restore(saved)
    subject._ale.restoreState = fail_first
    with pytest.raises(RuntimeError, match='archive load'): subject.archive_pair_trace(token, 1)
    assert len(calls) == 2 and subject._budget.charged_frames == before+120
    assert state(subject) == before_state and subject._trace_failure['accounting_complete'] is False


def test_archive_survives_native_recreation_with_current_state_restored():
    subject, anchor_witness = create(), create()
    subject.step(1); anchor_witness.step(1)
    token = subject.archive_save()
    # Synthetic analogue of a later cold episode; counters are never archived.
    subject._ale = ArchiveNative()
    subject._frames = deque([subject._ale.getScreenRGB(), subject._ale.getScreenRGB()], maxlen=2)
    subject._last_action = 0
    subject._budget.record_play(132, agent_steps=0)
    subject.step(0)
    before_state, before_rgb = state(subject), subject.observe()
    actual = subject.archive_pair_trace(token, 1)
    for a, b in zip(actual, anchor_witness.branch_pair_trace(1, frames=8), strict=True): equal_trace(a, b)
    assert state(subject) == before_state and np.array_equal(subject.observe(), before_rgb)
    assert subject._budget.report()['ordinary_frames'] == 1340


def test_entry_and_byte_caps_preserve_existing_tokens_and_drop_releases_bytes():
    subject = create(entries=1)
    token = subject.archive_save()
    report = subject.archive_report()
    assert report['serialized_bytes'] == len(subject._archive[token])
    with pytest.raises(ValueError, match='entry cap'): subject.archive_save()
    assert subject.archive_report() == report and subject._trace_failure is None
    subject.archive_drop(token)
    assert subject.archive_report()['serialized_bytes'] == 0
    subject._archive_limits = 2, report['serialized_bytes']-1
    with pytest.raises(ValueError, match='byte cap'): subject.archive_save()
    assert subject.archive_report()['entries'] == 0 and subject._trace_failure is None


def test_foreign_tokens_live_mode_and_legacy_api_cannot_bypass_archive_contract():
    subject, foreign = create(), create()
    token = foreign.archive_save()
    before = subject._budget.report()
    for invalid in (token, b'native state', None, 1):
        with pytest.raises(ValueError, match='token'): subject.archive_pair_trace(invalid, 1)
    for call in (subject.clone, lambda: subject.restore(token), lambda: subject.release(token)):
        with pytest.raises(ValueError): call()
    subject.config.mode = 'live'
    with pytest.raises(ValueError, match='live'): subject.archive_save()
    assert subject._budget.report() == before


@pytest.mark.parametrize('entries,byte_limit', [(0,1000), (True,1000), (20001,1000), (10,2000000001)])
def test_archive_limits_cannot_exceed_declared_caps(entries,byte_limit):
    with pytest.raises(ValueError): create(entries=entries,byte_limit=byte_limit)


def test_token_allocation_failure_keeps_archive_counts_and_stops(monkeypatch):
    subject = create()
    subject.archive_save()
    before, ledger = subject.archive_report(), subject._budget.report()
    def fail(size): raise MemoryError('synthetic token allocation failure')
    monkeypatch.setattr(archive.secrets, 'token_hex', fail)
    with pytest.raises(MemoryError): subject.archive_save()
    assert subject.archive_report() == before and subject._budget.report() == ledger
    with pytest.raises(ValueError, match='stopped'): subject.step(1)
