"""Experimenter-only distributed reference instrument, with no RAM observation RPC.

This diagnostic selects a raw or calibrated max-pooled convention at repeat4.
It is not a complete gate or an agent: pixels/unsupervised boxes are the only step payloads;
aggregate evaluation/accounting use a separate report command.
"""
import hashlib
import multiprocessing as mp

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig, EmulatorPool, _PixelEngine
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.evaluation.pong_v70 import decode_pong_ram
from ragnarok.evaluation.pooled_pong_v70 import pooled_ram_objects
from ragnarok.evaluation.perception_v70 import PerceptionScorer
from scripts.ref_extract_v70 import ReferenceExtractor


class _ALEScorerTap:
    """The scorer owns this ALE proxy; the pixel engine and its outputs stay unchanged."""
    def __init__(self, native, on_frame):
        self._native, self._on_frame = native, on_frame
        self.current_ram = native.getRAM().copy()
        self.previous_ram = self.current_ram.copy()

    def __getattr__(self, name):
        return getattr(self._native, name)

    def act(self, action):
        self.previous_ram = self.current_ram
        value = self._native.act(action)
        self.current_ram = self._native.getRAM().copy()
        self._on_frame(self.current_ram)
        return value

    def reset_game(self):
        value = self._native.reset_game()
        self.current_ram = self._native.getRAM().copy()
        self.previous_ram = self.current_ram.copy()
        return value

    def restoreState(self, state):
        value = self._native.restoreState(state)
        self.current_ram = self._native.getRAM().copy()
        self.previous_ram = self.current_ram.copy()
        return value


class _RawBranchEngine(_PixelEngine):
    """This diagnostic selects raw observations consistently, including branch endpoints."""
    def observe(self, raw=True):
        return super().observe(raw=raw)


class _RawColdBranchEngine(_ColdPixelEngine):
    def observe(self, raw=True):
        return super().observe(raw=raw)


class _ReferenceEngine:
    def __init__(self, config, *, branch_every=128, observation_mode="raw", reset_convention="natural_legacy"):
        if type(branch_every) is not int or branch_every < 128:
            raise ValueError("A branch cadence below128 decisions exceeds the proposed 20% allocation")
        if config.frame_skip != 4 or config.mode != "lab":
            raise ValueError("This development instrument requires repeat4 and laboratory mode")
        if observation_mode not in ("raw", "maxpooled"):
            raise ValueError("Observation mode must be raw or maxpooled")
        if reset_convention not in ("natural_legacy", "cold") or config.game != "pong":
            raise ValueError("Reference scorer supports Pong with an explicit known reset convention")
        self.observation_mode = observation_mode
        self.reset_convention = reset_convention
        if reset_convention == "cold":
            self.engine = (_RawColdBranchEngine if observation_mode == "raw" else _ColdPixelEngine)(config)
        else:
            self.engine = (_RawBranchEngine if observation_mode == "raw" else _PixelEngine)(config)
        self.reference, self.scorer = ReferenceExtractor(), PerceptionScorer()
        self._decoder = decode_pong_ram
        self._pooled_decoder = pooled_ram_objects
        self._ordinary_frames, self._episode, self._steps = 0, 0, 0
        self._phase = "ordinary"
        self._branch_every = branch_every
        self._lab_decoded = self._pairs = self._changed_pixels = 0
        self._label_digest = hashlib.sha256()
        self._reset_spent = self.engine.accounting()["reset_emulated_frames"]
        if reset_convention == "cold":
            self.engine.set_native_wrapper(lambda native: _ALEScorerTap(native, self._on_frame))
        else:
            self.engine._ale = _ALEScorerTap(self.engine._ale, self._on_frame)

    def _on_frame(self, ram):
        labels = self._decoder(ram)
        if self._phase == "lab":
            self._lab_decoded += 1
            return
        reset_spent = self.engine.accounting()["reset_emulated_frames"]
        if reset_spent != self._reset_spent:
            self._episode += 1
            self._reset_spent = reset_spent
        self._ordinary_frames += 1
        self.scorer.observe_truth(labels, frame=self._ordinary_frames, episode=self._episode)

    def step(self, action):
        pixels = self.engine.step(action)
        boxes = self.reference.observe(pixels)
        if self.observation_mode == "maxpooled":
            tap = self.engine._ale
            self.scorer.score(boxes, frame_truths=self._pooled_decoder(tap.previous_ram, tap.current_ram))
        else:
            self.scorer.score(boxes)
        self._steps += 1
        if self._steps % self._branch_every == 0:
            tap = self.engine._ale
            anchor_ram_pair = tap.previous_ram.copy(), tap.current_ram.copy()
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
                # Native restore only knows the anchor's current RAM. Preserve its preceding
                # RAM as well so an immediately inspected pooled observation stays aligned.
                tap.previous_ram, tap.current_ram = anchor_ram_pair
                self._phase = "ordinary"
        return pixels, boxes

    def report(self):
        return dict(accounting=self.engine.accounting(), perception=self.scorer.report(), reset_convention=self.reset_convention,
                    decisions=self._steps, ordinary_raw_frames=self._ordinary_frames,
                    branch_pairs=self._pairs, decoded_branch_frames=self._lab_decoded,
                    changed_pixels=self._changed_pixels, self_generated_label_sha256=self._label_digest.hexdigest())


def _serve_reference(connection, config, branch_every, observation_mode, reset_convention):
    try:
        engine = _ReferenceEngine(config, branch_every=branch_every, observation_mode=observation_mode,
                                  reset_convention=reset_convention)
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
    def __init__(self, configs, *, branch_every=128, observation_mode="raw", reset_convention="natural_legacy"):
        configs = list(configs)
        if not 1 <= len(configs) <= 16 or not all(isinstance(c, EmulatorConfig) for c in configs):
            raise ValueError("Supply1..16 EmulatorConfig instances")
        self._timeout, self._closed = 60.0, False
        self._connections, self._processes = [], []
        ctx = mp.get_context("spawn")
        try:
            for config in configs:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve_reference,
                                     args=(child, config, branch_every, observation_mode, reset_convention), daemon=True)
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
