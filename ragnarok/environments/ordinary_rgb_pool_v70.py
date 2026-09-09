"""Closed prospective transport for ordinary RGB traces and existing paid branches."""
from ragnarok.environments.frame_budget_v70 import BudgetExceeded
from ragnarok.environments.ordinary_rgb_trace_v70 import _OrdinaryRGBTraceEngine,validate_trace
from ragnarok.environments.rgb_trace_pool_v70 import RGBTracePool


def _serve_ordinary_rgb(connection,config):
    try:
        engine=_OrdinaryRGBTraceEngine(config)
        connection.send((True,engine.observe()))
        methods={name:getattr(engine,name) for name in (
            'observe','observe_pair','branch_pair','branch_pair_trace','accounting')}
        methods['step']=engine.step_trace
        methods['report']=lambda:dict(accounting=engine.accounting(),failure=engine._trace_failure)
        while True:
            command,args=connection.recv()
            if command=='close':
                connection.send((True,None))
                break
            try:
                if command not in methods:raise ValueError('Unknown ordinary RGB command')
                connection.send((True,methods[command](*args)))
            except Exception as error:
                kind='BudgetExceeded' if isinstance(error,BudgetExceeded) else 'ValueError'
                connection.send((False,(kind,f'{type(error).__name__}: {error}')))
    except EOFError:pass
    except Exception as error:
        try:connection.send((False,(type(error).__name__,str(error))))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:connection.close()


class OrdinaryRGBPool(RGBTracePool):
    _worker_target=staticmethod(_serve_ordinary_rgb)

    def observe_pair(self,index=0):
        return self._one(index,'observe_pair')

    def step(self,actions):
        self._guard()
        try:
            traces=super().step(actions)
            for trace in traces:validate_trace(trace)
            return traces
        except BaseException:
            # Paid replies may have been drained even if one worker failed.
            # Never resend an ambiguous action; reports/cleanup stay available.
            self._stopped=True
            raise
