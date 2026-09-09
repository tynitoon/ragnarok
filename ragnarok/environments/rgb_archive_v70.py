"""Bounded private archive for paid RGB experiments at earlier anchors.

Saving keeps ordinary play ordinary. An archived pair costs three restores:
load its anchor, restore it for the control, restore the current play state.
Native bytes remain inside this worker; tokens contain no game information.
This draft has no native validation or training/collection authorization yet.
"""
import pickle
import secrets

import numpy as np

from ragnarok.environments.frame_budget_v70 import BudgetExceeded, nonnegative_int
from ragnarok.environments.rgb_branch_trace_v70 import _RGBTraceColdEngine


class _RGBArchiveEngine(_RGBTraceColdEngine):
    def __init__(self, config, *, max_archive_entries=20_000, max_archive_bytes=2_000_000_000):
        self._init_archive(max_archive_entries, max_archive_bytes)
        super().__init__(config)

    def _init_archive(self, max_entries, max_bytes):
        for value, name in ((max_entries, 'max_archive_entries'), (max_bytes, 'max_archive_bytes')):
            if nonnegative_int(value, name) == 0:
                raise ValueError(f'{name} must be positive')
        if max_entries > 20_000 or max_bytes > 2_000_000_000:
            raise ValueError('Archive caps cannot exceed20000entries or2billion serialized bytes')
        self._archive = {}
        self._archive_bytes = 0
        self._archive_limits = max_entries, max_bytes

    def _archive_ready(self):
        self._guard()
        self._lab_only()
        if self._saves or self._recording is not None:
            raise ValueError('Archive calls require ordinary play outside a laboratory session')

    def clone(self):
        raise ValueError('Use archive_save; legacy laboratory snapshot tokens are disabled')

    def restore(self, token):
        raise ValueError('Only an archived branch pair can load an archive token')

    def release(self, token):
        raise ValueError('Use archive_drop; legacy laboratory snapshot tokens are disabled')

    def _encode_saved(self, saved):
        native, frames, rng, last_action = saved
        # Pickle is private encoding of state generated here, never an import API.
        return pickle.dumps((native.serialize(), frames, rng, last_action), protocol=5)

    def _decode_native(self, data):
        import ale_py
        return ale_py.ALEState(data)

    def _decode_saved(self, data):
        native, frames, rng, last_action = pickle.loads(data)
        return self._decode_native(native), frames, rng, last_action

    def archive_save(self):
        self._archive_ready()
        if len(self._archive) >= self._archive_limits[0]:
            raise ValueError('Archive entry cap reached; explicitly drop a token first')
        before = self._budget.charged_frames
        try:
            data = self._encode_saved(self._capture())
        except BaseException as error:
            self._record_failure(error, before)
            raise
        if self._archive_bytes + len(data) > self._archive_limits[1]:
            raise ValueError('Archive serialized-byte cap reached; no entry was replaced')
        try:
            new_size = self._archive_bytes + len(data)
            token = secrets.token_hex(16)
            while token in self._archive:
                token = secrets.token_hex(16)
            self._archive[token] = data
            self._archive_bytes = new_size
            return token
        except BaseException as error:
            self._record_failure(error, before)
            raise

    def archive_drop(self, token):
        if type(token) is not str or token not in self._archive:
            raise ValueError('Unknown worker-local archive token')
        self._archive_bytes -= len(self._archive.pop(token))

    def archive_report(self):
        """Experimenter metadata: payload bound, not a claim about process RSS."""
        return dict(entries=len(self._archive), serialized_bytes=self._archive_bytes,
                    max_entries=self._archive_limits[0], max_serialized_bytes=self._archive_limits[1])

    def archive_pair_trace(self, token, action, control=0, frames=8):
        self._archive_ready()
        if type(token) is not str or token not in self._archive:
            raise ValueError('Unknown worker-local archive token')
        self._action(action)
        self._action(control)
        if not 1 <= nonnegative_int(frames, 'frames') <= 64:
            raise ValueError('Branch length must be in 1..64 emulated frames')
        if not self._budget.can_experiment(2*frames, restores=3):
            raise BudgetExceeded('Complete archived pair, including three restores, does not fit')
        before = self._budget.charged_frames
        try:
            anchor = self._decode_saved(self._archive[token])
            current = self._capture()
            self._recording = []
            try:
                self._restore(anchor)
                intervention = self._advance(action, frames, experimental=True)
                self._restore(anchor)
                control_pixels = self._advance(control, frames, experimental=True)
                traces = tuple(self._recording)
                if len(traces) != 2 or any(not np.array_equal(t.endpoint, endpoint)
                                          for t, endpoint in zip(traces, (intervention, control_pixels), strict=True)):
                    raise RuntimeError('Archived trace endpoints differ from the paid branch outputs')
                return traces
            finally:
                # Even a failed branch returns to current play if native restore
                # succeeds. The monotone budget and failure latch never rewind.
                self._restore(current)
        except BaseException as error:
            self._record_failure(error, before)
            raise
        finally:
            self._recording = None
