"""Prospective closed RGB/paid-trace transport; no scorer or native-state RPC.

The reference benchmark does not use this pool. Its real-emulator wiring and
entry protocol must be reviewed before any collection or training launch.
"""
from ragnarok.environments.emu_pool import EmulatorPool
from ragnarok.environments.frame_budget_v70 import BudgetExceeded
from ragnarok.environments.rgb_branch_trace_v70 import _RGBTraceColdEngine


def _serve_rgb_trace(connection, config):
    try:
        engine = _RGBTraceColdEngine(config)
        connection.send((True, engine.observe()))
        methods = {name: getattr(engine, name) for name in
                   ('observe', 'step', 'branch_pair', 'branch_pair_trace', 'accounting')}
        methods['report'] = lambda: dict(accounting=engine.accounting(), failure=engine._trace_failure)
        while True:
            command, args = connection.recv()
            if command == 'close':
                connection.send((True, None))
                break
            try:
                if command not in methods:
                    raise ValueError('Unknown RGB trace command')
                connection.send((True, methods[command](*args)))
            except Exception as error:
                # A paid engine failure stays inspectable. Budget refusal keeps
                # its own class; ordinary Python failures cannot hide the ledger
                # by making the parent interpret them as a transport failure.
                name = 'BudgetExceeded' if isinstance(error, BudgetExceeded) else 'ValueError'
                connection.send((False, (name, f'{type(error).__name__}: {error}')))
    except EOFError:
        pass
    except Exception as error:
        try:
            connection.send((False, (type(error).__name__, str(error))))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class RGBTracePool(EmulatorPool):
    _worker_target = staticmethod(_serve_rgb_trace)

    def __init__(self, configs, *, timeout=30.0):
        self._stopped = False
        super().__init__(configs, timeout=timeout)

    def _guard(self):
        if self._stopped:
            raise ValueError('RGB trace pool stopped after an earlier action failure')

    def step(self, actions):
        self._guard()
        try:
            return super().step(actions)
        except BaseException:
            self._stopped = True
            raise

    def _branch(self, command, action, control, frames, index):
        self._guard()
        try:
            return self._one(index, command, action, control, frames)
        except BaseException:
            self._stopped = True
            raise

    def branch_pair_trace(self, action, *, control=0, frames=4, index=0):
        return self._branch('branch_pair_trace', action, control, frames, index)

    def branch_pair(self, action, *, control=0, frames=4, index=0):
        return self._branch('branch_pair', action, control, frames, index)

    def report(self, index=0):
        return self._one(index, 'report')

    def clone(self, index=0):
        raise ValueError('Persistent snapshots are not part of the RGB trace transport')

    def restore(self, token, index=0):
        raise ValueError('Persistent snapshots are not part of the RGB trace transport')

    def release(self, token, index=0):
        raise ValueError('Persistent snapshots are not part of the RGB trace transport')
