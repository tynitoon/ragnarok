"""Prospective external chain replacing only the MsPacman pill decoder.

Original grouping and the original reference/judge/branch path stay unchanged.
No model, new native run or throughput claim is introduced by this adapter.
"""
import multiprocessing as mp

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine,VisibleReferencePool
from ragnarok.evaluation.gather_visible_mspacman_v70 import mspacman_visible_scene


class GatherVisibleEngine(VisibleReferenceEngine):
    def __init__(self,config,*,branch_every=128,reference_backend='numpy',native_build=None):
        if config.game!='ms_pacman':raise ValueError('This prospective gather chain is MsPacman only')
        self._gather_chain_stopped=False
        super().__init__(config,branch_every=branch_every,reference_backend=reference_backend,native_build=native_build)
        if self._ordinary_frames or self._source_labels or self._steps:
            raise ValueError('Decoder selection must precede the first acted source scene')
        self._decoder=mspacman_visible_scene

    def step(self,action):
        if self._gather_chain_stopped:raise ValueError('Trajectory stopped after an earlier gather-chain failure')
        try:return super().step(action)
        except BaseException:
            # The frozen parent's diagnostic allocation may itself fail.
            self._gather_chain_stopped=True
            raise

    def report(self):
        result=super().report()
        if self._gather_chain_stopped and result['failure'] is None:
            result['failure']=dict(error_type='UnreportedStepFailure',
                error='Step failed before its diagnostic could be allocated; trajectory stopped')
        return result


def _serve_gather(connection,config,branch_every,reference_backend,native_build):
    try:
        engine=GatherVisibleEngine(config,branch_every=branch_every,
            reference_backend=reference_backend,native_build=native_build)
        connection.send((True,None))
        while True:
            command,args=connection.recv()
            if command=='close':break
            try:
                if command=='step':value=engine.step(*args)
                elif command=='report':value=engine.report()
                else:raise ValueError('Unknown visible reference command')
                connection.send((True,value))
            except Exception as exc:
                kind='ValueError' if command=='step' else type(exc).__name__
                connection.send((False,(kind,f'{type(exc).__name__}: {exc}')))
    except EOFError:pass
    except Exception as exc:
        try:connection.send((False,(type(exc).__name__,str(exc))))
        except (EOFError,OSError):pass
    finally:connection.close()


class GatherVisiblePool(VisibleReferencePool):
    def __init__(self,configs,*,branch_every=128,reference_backend='numpy',native_build=None):
        configs=list(configs)
        if (not 1<=len(configs)<=16 or not all(isinstance(c,EmulatorConfig) for c in configs)
                or any(c.game!='ms_pacman' for c in configs)):
            raise ValueError('Supply1..16 MsPacman EmulatorConfig instances')
        self._timeout,self._closed=60.,False
        self._step_failed=False
        self._connections,self._processes=[],[]
        ctx=mp.get_context('spawn')
        try:
            for config in configs:
                parent,child=ctx.Pipe()
                process=ctx.Process(target=_serve_gather,
                    args=(child,config,branch_every,reference_backend,native_build),daemon=True)
                try:process.start()
                except BaseException:
                    parent.close();child.close();raise
                child.close()
                self._connections.append(parent);self._processes.append(process)
            for connection in self._connections:self._receive(connection)
        except BaseException:
            self.close();raise
