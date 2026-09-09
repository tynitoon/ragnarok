"""Separate MsPacman judge chain with exact, stateless rectangle grouping.

The reviewed cached-chain step/stop/report path is reused unchanged. Its input
cache is replaced before the first acted scene, not combined with this candidate.
"""
import multiprocessing as mp

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferencePool
from ragnarok.evaluation.cached_visible_pool_v70 import CachedVisibleEngine
from ragnarok.evaluation.rectangle_visible_groups_v70 import rectangle_visible_groups


class _RectangleGroups:
    def group(self, objects):
        return rectangle_visible_groups(objects, game='ms_pacman')


class RectangleVisibleEngine(CachedVisibleEngine):
    def __init__(self, config, *, branch_every=128, reference_backend='numpy', native_build=None):
        if config.game != 'ms_pacman':
            raise ValueError('This prospective rectangle chain is MsPacman only')
        super().__init__(config, branch_every=branch_every,
                         reference_backend=reference_backend, native_build=native_build)
        if self._ordinary_frames or self._source_labels or self._steps:
            raise ValueError('Grouping selection must precede the first acted source scene')
        self._group_cache = _RectangleGroups()  # No cached inputs or outputs.


def _serve_rectangles(connection, config, branch_every, reference_backend, native_build):
    try:
        engine = RectangleVisibleEngine(config, branch_every=branch_every,
            reference_backend=reference_backend, native_build=native_build)
        connection.send((True, None))
        while True:
            command, args = connection.recv()
            if command == 'close':
                break
            try:
                if command == 'step':
                    value = engine.step(*args)
                elif command == 'report':
                    value = engine.report()
                else:
                    raise ValueError('Unknown visible reference command')
                connection.send((True, value))
            except Exception as exc:
                kind = 'ValueError' if command == 'step' else type(exc).__name__
                connection.send((False, (kind, f'{type(exc).__name__}: {exc}')))
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send((False, (type(exc).__name__, str(exc))))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class RectangleVisiblePool(VisibleReferencePool):
    def __init__(self, configs, *, branch_every=128, reference_backend='numpy', native_build=None):
        configs = list(configs)
        if (not 1 <= len(configs) <= 16 or not all(isinstance(c, EmulatorConfig) for c in configs)
                or any(c.game != 'ms_pacman' for c in configs)):
            raise ValueError('Supply1..16 MsPacman EmulatorConfig instances')
        self._timeout, self._closed = 60., False
        self._step_failed = False
        self._connections, self._processes = [], []
        ctx = mp.get_context('spawn')
        try:
            for config in configs:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve_rectangles,
                    args=(child, config, branch_every, reference_backend, native_build), daemon=True)
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
