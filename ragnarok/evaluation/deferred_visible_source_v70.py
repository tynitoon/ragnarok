"""Prospective lazy cold-frame geometry, preserving the failed CPU v0 runner.

A four-frame ordinary repeat scores only its last two sources. Its cold frame0
must not be decoded eagerly and then discarded. A one-frame repeat still needs
that actual source: it is decoded on access, with every original guard intact.
This prototype is not connected to a new native benchmark or collection.
"""
import numpy as np
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine


class _DeferredColdSource:
    def __init__(self, decoder, current, previous):
        self._decoder = decoder
        self._current, self._previous = current.copy(), previous.copy()
        for ram in (self._current, self._previous):
            if ram.dtype != np.uint8 or ram.shape != (128,):
                raise ValueError('Actual cold RAM history must be uint8[128]')
            ram.flags.writeable = False
        self._scene = None

    def __iter__(self):
        if self._scene is None:
            self._scene = tuple(self._decoder(self._current, self._previous))
        return iter(self._scene)


class DeferredVisibleSourceEngine(VisibleReferenceEngine):
    def _on_frame(self, tap):
        labels = self._decoder(tap.current, tap.previous)
        if self._phase == 'lab':
            self._lab_decoded += 1
            return
        loads = self.engine.accounting()['cold_loads']
        if loads != self._loads:
            if loads != self._loads+1:
                raise ValueError('Unexpected cold episode jump')
            self._loads = loads
            self._episode += 1
            self._source_labels.clear()
        if not self._source_labels:
            if tap.frame_numbers[-1] != 1:
                raise ValueError('Source history must start at the cold first act')
            self._source_labels.append(_DeferredColdSource(self._decoder, tap.previous, tap.before_previous))
        self._source_labels.append(labels)
        self._ordinary_frames += 1
        self.scorer.observe_truth(labels, frame=self._ordinary_frames, episode=self._episode)
