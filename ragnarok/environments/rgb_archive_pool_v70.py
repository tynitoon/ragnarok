"""Closed archive RPC with a pool-wide serialized-payload capacity.

No native state can enter or leave the transport. Per-worker capacity is the
floor of the pool total divided by its size; unused remainders are not lent.
This draft is not connected to a native collection or learning runner.
"""
from functools import partial

from ragnarok.environments.frame_budget_v70 import BudgetExceeded, nonnegative_int
from ragnarok.environments.rgb_archive_v70 import _RGBArchiveEngine
from ragnarok.environments.rgb_trace_pool_v70 import RGBTracePool


def _serve_rgb_archive(connection, config, *, archive_limits):
    try:
        engine = _RGBArchiveEngine(config, max_archive_entries=archive_limits[0],
                                   max_archive_bytes=archive_limits[1])
        connection.send((True, engine.observe()))
        methods = {name: getattr(engine, name) for name in (
            'observe', 'step', 'branch_pair', 'branch_pair_trace', 'accounting',
            'archive_save', 'archive_drop', 'archive_pair_trace')}
        methods['report'] = lambda: dict(accounting=engine.accounting(),failure=engine._trace_failure,
                                        archive=engine.archive_report())
        while True:
            command, args = connection.recv()
            if command == 'close':
                connection.send((True,None))
                break
            try:
                if command not in methods: raise ValueError('Unknown RGB archive command')
                connection.send((True,methods[command](*args)))
            except Exception as error:
                name = 'BudgetExceeded' if isinstance(error,BudgetExceeded) else 'ValueError'
                connection.send((False,(name,f'{type(error).__name__}: {error}')))
    except EOFError:
        pass
    except Exception as error:
        try: connection.send((False,(type(error).__name__,str(error))))
        except (BrokenPipeError,EOFError,OSError): pass
    finally:
        connection.close()


class RGBArchivePool(RGBTracePool):
    _worker_target = staticmethod(_serve_rgb_archive)

    def __init__(self, configs, *, max_archive_entries=20_000, max_archive_bytes=2_000_000_000, timeout=30.0):
        configs = list(configs)
        if not 1 <= len(configs) <= 16: raise ValueError('Supply1..16 worker configs')
        entries = nonnegative_int(max_archive_entries,'max_archive_entries')
        byte_limit = nonnegative_int(max_archive_bytes,'max_archive_bytes')
        if not len(configs) <= entries <= 20_000 or not len(configs) <= byte_limit <= 2_000_000_000:
            raise ValueError('Pool caps must allow each worker and fit20000entries/2billion bytes')
        limits = entries//len(configs),byte_limit//len(configs)
        self._worker_target = partial(self._worker_target,archive_limits=limits)
        super().__init__(configs,timeout=timeout)

    def _archive_action(self, index, command, *args):
        self._guard()
        try: return self._one(index,command,*args)
        except BaseException:
            self._stopped = True
            raise

    def archive_save(self, index=0):
        return self._archive_action(index,'archive_save')

    def archive_pair_trace(self, token, action, *, control=0, frames=8, index=0):
        return self._archive_action(index,'archive_pair_trace',token,action,control,frames)

    def archive_drop(self, token, index=0):
        # Resource cleanup and reports remain available after any action failure.
        return self._one(index,'archive_drop',token)
