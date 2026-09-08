"""Experimenter-only distributed reference instrument, with no RAM observation RPC.

This raw-frame diagnostic uses action repeat4. It is not the max-pooled gate
pipeline or an agent: pixels/unsupervised boxes are the only step payloads;
aggregate evaluation/accounting use a separate report command.
"""
import hashlib
import multiprocessing as mp

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig, EmulatorPool, _PixelEngine
from ragnarok.evaluation.pong_v70 import decode_pong_ram
from ragnarok.evaluation.perception_v70 import PerceptionScorer
from scripts.ref_extract_v70 import ReferenceExtractor


class _ALEScorerTap:
    """The scorer owns this ALE proxy; the pixel engine and its outputs stay unchanged."""
    def __init__(self, native, on_frame):
        self._native, self._on_frame = native, on_frame

    def __getattr__(self, name):
        return getattr(self._native, name)

    def act(self, action):
        value = self._native.act(action)
        self._on_frame(self._native.getRAM())
        return value


class _ReferenceEngine:
    def __init__(self, config, *, branch_every=128):
        if type(branch_every) is not int or branch_every < 128:
            raise ValueError("A branch cadence below128 decisions exceeds the proposed 20% allocation")
        if config.frame_skip != 4 or config.mode != "lab":
            raise ValueError("This development instrument requires repeat4 and laboratory mode")
        self.engine = _PixelEngine(config)
        self.reference, self.scorer = ReferenceExtractor(), PerceptionScorer()
        self._decoder = decode_pong_ram
        self._ordinary_frames, self._episode, self._steps = 0, 0, 0
        self._phase = "ordinary"
        self._branch_every = branch_every
        self._lab_decoded = self._pairs = self._changed_pixels = 0
        self._label_digest = hashlib.sha256()
        self._reset_spent = self.engine.accounting()["reset_emulated_frames"]
        native = self.engine._ale
        self.engine._ale = _ALEScorerTap(native, self._on_frame)

    def _on_frame(self, ram):
        labels = self._decoder(ram)
        if self._phase == "lab":
            self._lab_decoded += 1
            return
        reset_spent = self.engine.accounting()["reset_emulated_frames"]
        if reset_spent != self._reset_spent:
            self._episode += 1
            self.reference.reset()  # disclosed experimenter-side fixture/reset boundary
            self._reset_spent = reset_spent
        self._ordinary_frames += 1
        self.scorer.observe_truth(labels, frame=self._ordinary_frames, episode=self._episode)

    def step(self, action):
        self.engine.step(action)
        raw = self.engine.observe(raw=True)
        boxes = self.reference.observe(raw)
        self.scorer.score(boxes)
        self._steps += 1
        if self._steps % self._branch_every == 0:
            self._phase = "lab"
            try:
                intervention, control = self.engine.branch_pair(action, control=0, frames=4)
                # Self-generated C target; no RAM label is used to construct it.
                changed = np.any(intervention != control, axis=2)
                self._label_digest.update(changed.tobytes())
                self._changed_pixels += int(changed.sum())
                self.reference.extract(intervention)
                self.reference.extract(control)
                self._pairs += 1
            finally:
                self._phase = "ordinary"
        return raw, boxes

    def report(self):
        return dict(accounting=self.engine.accounting(), perception=self.scorer.report(),
                    decisions=self._steps, ordinary_raw_frames=self._ordinary_frames,
                    branch_pairs=self._pairs, decoded_branch_frames=self._lab_decoded,
                    changed_pixels=self._changed_pixels, self_generated_label_sha256=self._label_digest.hexdigest())


def _serve_reference(connection, config, branch_every):
    try:
        engine = _ReferenceEngine(config, branch_every=branch_every)
        connection.send((True, None))
        while True:
            command, args = connection.recv()
            if command == "close":
                break
            try:
                if command == "step":
                    value = engine.step(*args)
                elif command == "report":
                    value = engine.report()
                else:
                    raise ValueError("Unknown reference instrument command")
                connection.send((True, value))
            except Exception as exc:
                connection.send((False, (type(exc).__name__, str(exc))))
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send((False, (type(exc).__name__, str(exc))))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class ReferencePool(EmulatorPool):
    """Reuse the tested closed transport/cleanup, with a dedicated evaluator worker."""
    def __init__(self, configs, *, branch_every=128):
        configs = list(configs)
        if not 1 <= len(configs) <= 16 or not all(isinstance(c, EmulatorConfig) for c in configs):
            raise ValueError("Supply1..16 EmulatorConfig instances")
        self._timeout, self._closed = 60.0, False
        self._connections, self._processes = [], []
        ctx = mp.get_context("spawn")
        try:
            for config in configs:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve_reference, args=(child, config, branch_every), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            for connection in self._connections:
                self._receive(connection)
        except BaseException:
            self.close()
            raise

    def report(self, index=0):
        return self._one(index, "report")
