"""Private Ms-Pac-Man/Asteroids CPU reference chain; no agent or oracle RPC.

Each real frame updates external movement history. Source labels are cached
only along ordinary play; laboratory frames are decoded for diagnostic cost,
then their native/RGB/sticky/three-RAM state is restored in the charged engine.
"""
from collections import deque
import hashlib
import multiprocessing as mp
import numpy as np
from ragnarok.environments.emu_pool import EmulatorConfig,EmulatorPool
from ragnarok.evaluation.restorable_ram_history_v70 import _ScorerHistoryColdEngine
from ragnarok.evaluation.visible_scene_v70 import asteroids_visible_scene
from ragnarok.evaluation.fast_visible_mspacman_v70 import mspacman_visible_scene
from ragnarok.evaluation.visible_temporal_groups_v70 import VisibleTemporalScorer,group_visible_sources,REGISTRY
from scripts.ref_extract_v70 import ReferenceExtractor


class VisibleReferenceEngine:
    def __init__(self,config,*,branch_every=128,reference_backend='numpy',native_build=None):
        if (config.game not in REGISTRY or config.frame_skip!=4 or config.mode!='lab' or branch_every!=128):
            raise ValueError('The visible CPU chain requires a declared repeat4/lab game and cadence128')
        if reference_backend not in ('numpy','native') or (reference_backend=='native' and native_build is None):
            raise ValueError('Choose an explicit reference backend and native build')
        self.game=config.game;self._branch_every=branch_every
        self._decoder=mspacman_visible_scene if self.game=='ms_pacman' else asteroids_visible_scene
        self.scorer=VisibleTemporalScorer(self.game)
        if reference_backend=='native':
            from ragnarok.evaluation.native_reference_v70 import NativeReferenceExtractor
            self.reference=NativeReferenceExtractor(build_manifest=native_build)
        else:self.reference=ReferenceExtractor()
        self._phase='ordinary';self._source_labels=deque(maxlen=2)
        self._failure=None
        self._ordinary_frames=self._episode=self._steps=0
        self._lab_decoded=self._pairs=self._changed_pixels=0
        self._label_digest=hashlib.sha256()
        self._reference_digest=hashlib.sha256()
        self.engine=_ScorerHistoryColdEngine(config,self._on_frame)
        self._loads=self.engine.accounting()['cold_loads']

    def _on_frame(self,tap):
        labels=self._decoder(tap.current,tap.previous)
        if self._phase=='lab':
            self._lab_decoded+=1
            return
        loads=self.engine.accounting()['cold_loads']
        if loads!=self._loads:
            if loads!=self._loads+1:raise ValueError('Unexpected cold episode jump')
            self._loads=loads;self._episode+=1;self._source_labels.clear()
        if not self._source_labels:
            if tap.frame_numbers[-1]!=1:raise ValueError('Source history must start at the cold first act')
            self._source_labels.append(self._decoder(tap.previous,tap.before_previous))
            # Frame0 supplies pooled geometry only. As in the other reference
            # diagnostics, movement history begins with ordinary acted frames;
            # the cold0->1 transition is not a movement-eligible observation.
        self._source_labels.append(labels)
        self._ordinary_frames+=1
        self.scorer.observe_truth(labels,frame=self._ordinary_frames,episode=self._episode)

    def step(self,action):
        if self._failure is not None:
            raise ValueError('Trajectory stopped after an earlier failure; only report/close remain')
        try:return self._step(action)
        except BaseException as error:
            self._failure=dict(error_type=type(error).__name__,error=str(error))
            raise

    def _step(self,action):
        pixels=self.engine.step(action)
        boxes=self.reference.observe(pixels)
        self._record_reference(b'ordinary',pixels,boxes)
        if len(self._source_labels)!=2:raise ValueError('Both pooled source scenes are required')
        groups=group_visible_sources([*self._source_labels[0],*self._source_labels[1]],game=self.game)
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

    def _record_reference(self,tag,pixels,boxes):
        self._reference_digest.update(tag)
        for array in (pixels,boxes):
            self._reference_digest.update(array.dtype.str.encode('ascii'))
            self._reference_digest.update(np.asarray(array.shape,dtype='<i8').tobytes())
            self._reference_digest.update(np.ascontiguousarray(array).tobytes())

    def report(self):
        return dict(accounting=self.engine.accounting(),perception=self.scorer.report(),
            reset_convention='fixed_hardware_cold_start_v1',decisions=self._steps,
            ordinary_raw_frames=self._ordinary_frames,branch_pairs=self._pairs,
            decoded_branch_frames=self._lab_decoded,changed_pixels=self._changed_pixels,
            self_generated_label_sha256=self._label_digest.hexdigest(),
            ordered_reference_output_sha256=self._reference_digest.hexdigest(),failure=self._failure)


def _serve_visible(connection,config,branch_every,reference_backend,native_build):
    try:
        engine=VisibleReferenceEngine(config,branch_every=branch_every,
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
                # The trajectory is poisoned by step(); keep the transport open
                # for its partial accounting report, without allowing another act.
                kind='ValueError' if command=='step' else type(exc).__name__
                connection.send((False,(kind,f'{type(exc).__name__}: {exc}')))
    except EOFError:pass
    except Exception as exc:
        try:connection.send((False,(type(exc).__name__,str(exc))))
        except (EOFError,OSError):pass
    finally:connection.close()


class VisibleReferencePool(EmulatorPool):
    """Same closed transport/cleanup; the parent receives only RGB and RGB boxes."""
    def __init__(self,configs,*,branch_every=128,reference_backend='numpy',native_build=None):
        configs=list(configs)
        if not 1<=len(configs)<=16 or not all(isinstance(c,EmulatorConfig) for c in configs):
            raise ValueError('Supply1..16 EmulatorConfig instances')
        self._timeout,self._closed=60.,False
        self._step_failed=False
        self._connections,self._processes=[],[]
        ctx=mp.get_context('spawn')
        try:
            for config in configs:
                parent,child=ctx.Pipe()
                process=ctx.Process(target=_serve_visible,
                    args=(child,config,branch_every,reference_backend,native_build),daemon=True)
                try:process.start()
                except BaseException:
                    parent.close();child.close();raise
                child.close();self._connections.append(parent);self._processes.append(process)
            for connection in self._connections:self._receive(connection)
        except BaseException:self.close();raise

    def report(self,index=0):return self._one(index,'report')

    def step(self,actions):
        if self._step_failed:
            raise ValueError('Visible pool trajectory stopped; only report/close remain')
        try:return super().step(actions)
        except BaseException:
            self._step_failed=True
            raise
