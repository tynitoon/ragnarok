"""Prospective MsPacman external chain changing only exact visible grouping.

The frozen decoder, reference, per-frame judge and paid branch/restore engine
remain inherited. This module is never a learner observation/target provider.
"""
import multiprocessing as mp
import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine,VisibleReferencePool
from ragnarok.evaluation.cached_visible_groups_v70 import CachedVisibleGroups


class CachedVisibleEngine(VisibleReferenceEngine):
    def __init__(self,config,*,branch_every=128,reference_backend='numpy',native_build=None):
        if config.game!='ms_pacman':raise ValueError('This prospective cached chain is MsPacman only')
        self._cache_chain_stopped=False
        self._group_cache=CachedVisibleGroups(config.game)
        super().__init__(config,branch_every=branch_every,reference_backend=reference_backend,native_build=native_build)

    def step(self,action):
        if self._cache_chain_stopped:raise ValueError('Trajectory stopped after an earlier cached-chain failure')
        try:return super().step(action)
        except BaseException:
            # The frozen parent's diagnostic dict allocation may itself fail.
            # This allocation-free latch still prevents the next paid command.
            self._cache_chain_stopped=True
            raise

    def report(self):
        result=super().report()
        if self._cache_chain_stopped and result['failure'] is None:
            result['failure']=dict(error_type='UnreportedStepFailure',
                                   error='Step failed before its diagnostic could be allocated; trajectory stopped')
        return result

    def _step(self,action):
        pixels=self.engine.step(action)
        boxes=self.reference.observe(pixels)
        self._record_reference(b'ordinary',pixels,boxes)
        if len(self._source_labels)!=2:raise ValueError('Both pooled source scenes are required')
        groups=self._group_cache.group([*self._source_labels[0],*self._source_labels[1]])
        self.scorer.score(boxes,frame_truths=groups)
        self._steps+=1
        if self._steps%self._branch_every==0:
            self._phase='lab'
            try:
                intervention,control=self.engine.branch_pair(action,control=0,frames=4)
                changed=np.any(intervention!=control,axis=2)
                self._label_digest.update(changed.tobytes());self._changed_pixels+=int(changed.sum())
                self._record_reference(b'intervention',intervention,self.reference.extract(intervention))
                self._record_reference(b'control',control,self.reference.extract(control))
                self._pairs+=1
            finally:self._phase='ordinary'
        return pixels,boxes


def _serve_cached(connection,config,branch_every,reference_backend,native_build):
    try:
        engine=CachedVisibleEngine(config,branch_every=branch_every,
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


class CachedVisiblePool(VisibleReferencePool):
    """Private reference-only workers; same closed/drained failure transport."""
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
                process=ctx.Process(target=_serve_cached,
                    args=(child,config,branch_every,reference_backend,native_build),daemon=True)
                try:process.start()
                except BaseException:
                    parent.close();child.close();raise
                child.close();self._connections.append(parent);self._processes.append(process)
            for connection in self._connections:self._receive(connection)
        except BaseException:self.close();raise
