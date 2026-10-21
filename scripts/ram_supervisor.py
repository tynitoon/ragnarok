"""RAM supervisor: keep >= 12 GB free for the other project (Tavern) sharing this machine.

Every few seconds: if available RAM < LOW (12 GB), SUSPEND the largest Ragnarok process and trim
its working set (its pages go to the pagefile, physical RAM is returned to the system); repeat
until available RAM >= LOW or nothing is left to suspend. When available RAM > HIGH (14 GB),
RESUME suspended processes one at a time, oldest first, waiting between resumes. Work is paused,
never lost. Nothing is installed: Windows APIs through ctypes, process listing through PowerShell.

A process is "Ragnarok" iff it is python.exe, never mentions "tavern", and EITHER descends from
this project's Claude Code session process (the claude.exe ancestor of the supervisor, or --root)
OR its command line names this repository / this session's scratch area (or its parent is such a
process: multiprocessing workers). Other sessions (Tavern) have their own claude.exe root and are
never touched.

Usage (detached): C:/Python314/python.exe scripts/ram_supervisor.py [--low 12] [--high 14]
Log: craft_v6_out/ram_supervisor.log ; status: craft_v6_out/ram_supervisor_status.json
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import time

GB = 1024 ** 3
MARKERS = ("ragnarok", "826ba484", "agent_v71", "-m scripts.", "scratchpad\\diag", "scratchpad/diag")
EXCLUDE = ("tavern",)
SELF = os.getpid()

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
psapi = ctypes.WinDLL("psapi")
PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_INFORMATION = 0x0400


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_gb():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GB


def processes():
    """[{pid, ppid, name, rss_gb, cmd}] for ALL processes (ancestry needs non-python parents)."""
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,WorkingSetSize,CommandLine"
          " | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=90).stdout.strip()
    except Exception:  # noqa: BLE001
        return []
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, dict):
        data = [data]
    return [dict(pid=int(d["ProcessId"]), ppid=int(d["ParentProcessId"] or 0), name=(d["Name"] or "").lower(),
                 rss_gb=(d["WorkingSetSize"] or 0) / GB, cmd=(d["CommandLine"] or "")) for d in data]


def find_root(procs):
    """The claude.exe (Claude Code CLI) ancestor of this supervisor: this session's process tree."""
    by = {p["pid"]: p for p in procs}
    pid, seen = SELF, set()
    while pid in by and pid not in seen:
        seen.add(pid)
        p = by[pid]
        if p["name"] == "claude.exe" and "claude-code" in p["cmd"].lower():
            return pid
        pid = p["ppid"]
    return None


def ragnarok_set(procs, root):
    by = {p["pid"]: p for p in procs}

    def descends(pid):
        seen = set()
        while pid in by and pid not in seen:
            if pid == root:
                return True
            seen.add(pid)
            pid = by[pid]["ppid"]
        return False

    py = [p for p in procs if p["name"] == "python.exe" and p["pid"] != SELF
          and not any(x in p["cmd"].lower() for x in EXCLUDE)]
    def by_cwd(p):
        c = process_cwd(p["pid"]).lower()
        return bool(c) and not any(x in c for x in EXCLUDE) and any(m in c for m in MARKERS)
    mine = {p["pid"] for p in py if (root and descends(p["pid"])) or any(m in p["cmd"].lower() for m in MARKERS)
            or by_cwd(p)}
    changed = True
    while changed:                       # multiprocessing children of a Ragnarok process
        changed = False
        for p in py:
            if p["pid"] not in mine and p["ppid"] in mine:
                mine.add(p["pid"]); changed = True
    return [by[i] for i in mine]


PROCESS_VM_READ = 0x0010


class _PBI(ctypes.Structure):
    _fields_ = [("ExitStatus", ctypes.c_void_p), ("PebBaseAddress", ctypes.c_void_p),
                ("AffinityMask", ctypes.c_void_p), ("BasePriority", ctypes.c_void_p),
                ("UniqueProcessId", ctypes.c_void_p), ("InheritedFromUniqueProcessId", ctypes.c_void_p)]


def process_cwd(pid):
    """Current directory of a 64-bit process, read-only from its PEB (ProcessParameters at +0x20,
    CurrentDirectory.DosPath UNICODE_STRING at +0x38). Returns '' on any failure."""
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return ""
    try:
        pbi = _PBI()
        if ntdll.NtQueryInformationProcess(h, 0, ctypes.byref(pbi), ctypes.sizeof(pbi), None) != 0:
            return ""

        def read(addr, size):
            buf = ctypes.create_string_buffer(size)
            n = ctypes.c_size_t()
            if not k32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)):
                return None
            return buf.raw[:n.value]
        raw = read(pbi.PebBaseAddress + 0x20, 8)
        if not raw:
            return ""
        params = int.from_bytes(raw, "little")
        us = read(params + 0x38, 16)
        if not us:
            return ""
        length = int.from_bytes(us[0:2], "little")
        bufp = int.from_bytes(us[8:16], "little")
        s = read(bufp, length)
        return s.decode("utf-16-le", "replace") if s else ""
    except Exception:  # noqa: BLE001
        return ""
    finally:
        k32.CloseHandle(h)


def _call(pid, fn, access):
    h = k32.OpenProcess(access, False, pid)
    if not h:
        return False
    try:
        return fn(h)
    finally:
        k32.CloseHandle(h)


def suspend(pid):
    def f(h):
        ok = ntdll.NtSuspendProcess(h) == 0
        psapi.EmptyWorkingSet(h)          # hand its physical pages back to the system
        return ok
    return _call(pid, f, PROCESS_SUSPEND_RESUME | PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION)


def resume(pid):
    return _call(pid, lambda h: ntdll.NtResumeProcess(h) == 0, PROCESS_SUSPEND_RESUME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", type=float, default=12.0)
    ap.add_argument("--high", type=float, default=14.0)
    ap.add_argument("--period", type=float, default=5.0)
    ap.add_argument("--resume-gap", type=float, default=45.0)
    ap.add_argument("--root", type=int, default=0, help="Claude Code session PID (default: auto, own ancestor)")
    ap.add_argument("--dry", action="store_true", help="print the classification once and exit")
    a = ap.parse_args()
    procs = processes()
    root = a.root or find_root(procs)
    if a.dry:
        mine = {p["pid"] for p in ragnarok_set(procs, root)}
        print(f"root {root} | available {available_gb():.1f} GB")
        for p in procs:
            if p["name"] == "python.exe":
                print("RAGNAROK" if p["pid"] in mine else "other   ", p["pid"], f"{p['rss_gb']:.2f}", p["cmd"][:100])
        return
    log = open("craft_v6_out/ram_supervisor.log", "a", buffering=1, encoding="utf-8")
    suspended = []                       # [(pid, t_suspended, rss_gb, cmd)] oldest first
    last_resume = 0.0
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start low={a.low} high={a.high}\n")
    while True:
        avail = available_gb()
        if avail < a.low:
            procs = ragnarok_set(processes(), root)
            alive = {p["pid"] for p in procs}
            suspended = [s for s in suspended if s[0] in alive]
            cand = sorted([p for p in procs if p["pid"] not in {s[0] for s in suspended}],
                          key=lambda p: -p["rss_gb"])
            if cand and cand[0]["rss_gb"] > 0.05:
                p = cand[0]
                if suspend(p["pid"]):
                    suspended.append((p["pid"], time.time(), p["rss_gb"], p["cmd"][:120]))
                    log.write(f"{time.strftime('%H:%M:%S')} avail {avail:.1f} GB < {a.low}: SUSPEND {p['pid']} "
                              f"({p['rss_gb']:.2f} GB) {p['cmd'][:100]}\n")
                    time.sleep(2)
                    continue
        elif suspended and avail > a.high and time.time() - last_resume > a.resume_gap:
            pid = suspended[0][0]
            resume(pid)
            suspended.pop(0)
            last_resume = time.time()
            log.write(f"{time.strftime('%H:%M:%S')} avail {avail:.1f} GB > {a.high}: RESUME {pid}\n")
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), available_gb=round(avail, 2),
                       suspended=[dict(pid=s[0], since=round(time.time() - s[1]), rss_gb=round(s[2], 2), cmd=s[3])
                                  for s in suspended]),
                  open("craft_v6_out/ram_supervisor_status.json", "w"), indent=1)
        time.sleep(a.period)


if __name__ == "__main__":
    main()
