import ctypes,json,os,time
from pathlib import Path
from ctypes import wintypes
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=(wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL))
b=wintypes.BOOL()
if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(b)):raise ctypes.WinError(ctypes.get_last_error())
p=Path("F:/dev/ragnarok/craft_v6_out/v62_confirm_interruption_20260910/process_lifetime_wmi_probe.json")
with p.open("x") as f:json.dump(dict(pid=os.getpid(),in_job=bool(b.value),started=time.time()),f)
time.sleep(20)
with p.with_name("process_lifetime_wmi_probe_completed.json").open("x") as f:json.dump(dict(pid=os.getpid(),completed=time.time()),f)
