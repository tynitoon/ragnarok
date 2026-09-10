"""One detached dispatch of the already reviewed direct-path fit."""

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "craft_v6_out/v70_stationary_direct_fit"
EXPECTED_PREFLIGHT = "891f582ead06ef741bfbc72541cefdb0d151e8817055415853ad02cbc2eb4e83"

os.chdir(ROOT)
os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.stdout = (FOLDER / "launch_v0/console.log").open("x", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.IsProcessInJob.argtypes = (
    wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)
)
inside = wintypes.BOOL()
if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)):
    raise ctypes.WinError(ctypes.get_last_error())
if inside.value:
    raise RuntimeError("The fit requires a process outside the tool's Job")
if hashlib.sha256((FOLDER / "preflight_v0.json").read_bytes()).hexdigest() != EXPECTED_PREFLIGHT:
    raise RuntimeError("Independent preflight changed after dispatch preparation")
if list(FOLDER.glob("fit_v0*")):
    raise RuntimeError("An attempt or result already exists; do not retry")

print(json.dumps(dict(pid=os.getpid(), detached_from_job=True,
                      mode="fixed_stationary_direct_path_fit")))
sys.stdout.flush()
os.fsync(sys.stdout.fileno())
sys.argv = ["scripts.fit_stationary_direct_v70", "--run"]
runpy.run_module("scripts.fit_stationary_direct_v70", run_name="__main__")
