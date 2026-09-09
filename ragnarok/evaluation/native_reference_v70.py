"""Optional exact native arithmetic for the generic causal RGB reference.

The NumPy implementation owns palette growth, limits and component extraction.
The native loop only fuses existing-colour counter increments and RGB packing.
"""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.ref_extract_v70 import ReferenceExtractor


class NativeReferenceExtractor(ReferenceExtractor):
    def __init__(self, *, build_manifest, max_colours=256):
        manifest_path=Path(build_manifest).resolve()
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        root=Path(__file__).resolve().parents[2]
        source=root/"ragnarok/native/reference_update_v70.c"
        dll=manifest_path.parent/"reference_update_v70.dll"
        if (ctypes.sizeof(ctypes.c_void_p)!=8 or
                hashlib.sha256(source.read_bytes()).hexdigest()!=manifest["source_sha256"] or
                hashlib.sha256(dll.read_bytes()).hexdigest()!=manifest["dll_sha256"]):
            raise ValueError("Native reference source/binary differs from the explicit x64 build")
        self.native_provenance=dict(build_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                    source_sha256=manifest["source_sha256"],dll_sha256=manifest["dll_sha256"])
        self._library=ctypes.CDLL(str(dll))
        u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags="C_CONTIGUOUS")
        u16=np.ctypeslib.ndpointer(dtype=np.uint16,flags="C_CONTIGUOUS")
        u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags="C_CONTIGUOUS")
        self._library.reference_update.argtypes=[u32,ctypes.c_uint64,u16,ctypes.c_uint32,u32,u32,u32]
        self._library.reference_update.restype=ctypes.c_int
        self._library.reference_pack.argtypes=[u8,ctypes.c_uint64,u32]
        self._library.reference_pack.restype=None
        super().__init__(max_colours=max_colours)

    def _packed(self,rgb):
        if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or
                rgb.shape[2]!=3 or not rgb.shape[0] or not rgb.shape[1]):
            raise ValueError("Expected a nonempty uint8 RGB image")
        packed=np.empty(rgb.shape[:2],dtype=np.uint32)
        contiguous=np.ascontiguousarray(rgb)
        self._library.reference_pack(contiguous,packed.size,packed)
        return packed

    def _update_packed(self,packed):
        if self._mode is None:
            return super()._update_packed(packed)
        if self._mode.shape!=packed.shape:
            raise ValueError("Resolution changed; start a new background explicitly")
        if self.frames_seen>=1_000_000:
            raise ValueError("Background observation limit reached; counters cannot overflow")
        result=self._library.reference_update(packed,packed.size,self._colour_ids,self._counts.shape[0],
                                               self._counts,self._mode,self._best)
        if result==1:
            # The C preflight made no mutation. Python may grow the palette or fail
            # atomically at its original capacity limit, then update this frame once.
            return super()._update_packed(packed)
        if result!=0:
            raise ValueError("Invalid packed colour or palette ID before native mutation")
        self.frames_seen+=1
