"""Seaquest-only external reference worker, reusing the tested step/branch path."""
import hashlib
import multiprocessing as mp

from ragnarok.environments.emu_pool import EmulatorConfig,EmulatorPool
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.evaluation.reference_pool_v70 import _ReferenceEngine,_ALEScorerTap
from ragnarok.evaluation.seaquest_scene_v70 import SeaquestSceneScorer,seaquest_scene_objects
from ragnarok.evaluation.fast_pooled_seaquest_v70 import pooled_seaquest_groups
from scripts.ref_extract_v70 import ReferenceExtractor


class SeaquestReferenceEngine(_ReferenceEngine):
    def __init__(self,config,*,branch_every=128,observation_mode="maxpooled",reset_convention="cold",
                 reference_backend="numpy",native_build=None):
        if (config.game!="seaquest" or config.frame_skip!=4 or config.mode!="lab" or
                observation_mode!="maxpooled" or reset_convention!="cold" or branch_every!=128):
            raise ValueError("Seaquest instrument requires the fixed cold/maxpooled/repeat4/lab convention")
        if reference_backend not in ("numpy","native") or (reference_backend=="native" and native_build is None):
            raise ValueError("Select an explicit reference backend/build")
        self.observation_mode,self.reset_convention=observation_mode,reset_convention
        self.engine=_ColdPixelEngine(config)
        if reference_backend=="native":
            from ragnarok.evaluation.native_reference_v70 import NativeReferenceExtractor
            self.reference=NativeReferenceExtractor(build_manifest=native_build)
        else:
            self.reference=ReferenceExtractor()
        self.scorer=SeaquestSceneScorer()
        self._decoder,self._pooled_decoder=seaquest_scene_objects,pooled_seaquest_groups
        self._ordinary_frames,self._episode,self._steps=0,0,0
        self._phase="ordinary"
        self._branch_every=branch_every
        self._lab_decoded=self._pairs=self._changed_pixels=0
        self._label_digest=hashlib.sha256()
        self._reset_spent=self.engine.accounting()["reset_emulated_frames"]
        self.engine.set_native_wrapper(lambda native:_ALEScorerTap(native,self._on_frame))


def _serve_reference(connection, config, branch_every, observation_mode, reset_convention, reference_backend, native_build):
    try:
        engine = SeaquestReferenceEngine(config, branch_every=branch_every, observation_mode=observation_mode,
                                  reset_convention=reset_convention,reference_backend=reference_backend,native_build=native_build)
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


class SeaquestReferencePool(EmulatorPool):
    """Reuse the tested closed transport/cleanup, with a dedicated evaluator worker."""
    def __init__(self, configs, *, branch_every=128, observation_mode="maxpooled", reset_convention="cold",
                 reference_backend="numpy", native_build=None):
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
                                     args=(child, config, branch_every, observation_mode, reset_convention,
                                           reference_backend, native_build), daemon=True)
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
