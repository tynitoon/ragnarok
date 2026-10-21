---
name: ragnarok-run
description: Use before launching, monitoring, pausing or stopping any Ragnarok compute (training, evaluation, data collection, offline benchmarks) on this Windows machine shared with the Tavern project, and whenever another session or Jérémie asks for memory back.
---

# Running Ragnarok compute on the shared machine

The Tavern project runs on the same 32 GB Windows workstation and has priority. Two limits matter:
**physical RAM** and, binding in practice, **Windows commit** (commit limit 35.2 GB, small pagefile).
Both must keep >= 12 GB available for Tavern at all times.

## Before any launch
1. Measure both (a suspended process still holds its commit; any python importing torch holds ~1.1 GB):
   ```bash
   C:/Python314/python.exe -c "import ctypes,ctypes.wintypes as w
   class M(ctypes.Structure): _fields_=[('l',w.DWORD),('d',w.DWORD),('tp',ctypes.c_ulonglong),('ap',ctypes.c_ulonglong),('tf',ctypes.c_ulonglong),('af',ctypes.c_ulonglong),('x',ctypes.c_ulonglong*3)]
   m=M(); m.l=ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); print(f'RAM {m.ap/2**30:.1f} GB, commit {m.af/2**30:.1f} GB')"
   ```
   Launch only if both stay >= 12 GB after the job (a job takes ~1.5-2.5 GB). Usually 1-2 jobs at a time.
2. The RAM supervisor must be running (status `craft_v6_out/ram_supervisor_status.json` updated within
   seconds). Otherwise start it detached (Bash `run_in_background`), from the repo root:
   `C:/Python314/python.exe scripts/ram_supervisor.py --low 12 --high 14 --clow 12 --chigh 14`.
   Below 12 GB RAM it suspends Ragnarok processes; below 12 GB commit it terminates worker jobs.
3. Do not edit code a running queue imports: every queue job is a fresh subprocess that re-imports
   `ragnarok/` and `scripts/`. Develop in a separate git worktree (`git worktree add ../ragnarok_vNN -b ...`).

## Launching
- Use the resumable queue scripts (`python -m scripts.slice_vNN all [k/n]`): one subprocess per job,
  skip-if-output-exists, logs `craft_v6_out/vNN/<job>.log`, launch gates that WAIT (never skip) for
  14 GB of RAM and commit. Any new queue script must copy that gate and import torch lazily inside jobs.
- Windows: multiprocessing from stdin fails; run scripts as files or modules. Git Bash heredocs with
  non-ASCII text (±, é) can break Python patches: write patch scripts with the Write tool instead.
- Run from the right worktree (`cd F:/dev/ragnarok_vNN && ...`); outputs land in that worktree.

## Monitoring
- One Monitor per set of queue logs: `tail -n 0 -F <queue logs> | grep --line-buffered -E "end |Traceback|Error"`
  (timeout 30 min, re-arm on expiry). `exit 1` with "RAM gate" in the job log means a skipped job: rerun `all`.
- Check `craft_v6_out/ram_supervisor.log` when everything is quiet: SUSPEND lines mean Tavern needs memory.

## When Tavern (or Jérémie) asks for memory
1. `TaskStop` each queue's background task (kills the process tree), then list python processes:
   `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` (PowerShell) and confirm only the
   supervisor and processes that are not ours remain. Never touch anything mentioning `tavern`.
2. Measure RAM and commit again; report the numbers, what was freed and what is held by other apps.
3. Reply to the requesting session (`ListAgents`, `SendMessage`) and wait for its or Jérémie's
   go-ahead before relaunching. Queues resume from their outputs; only the job in progress is redone.
4. A larger pagefile would ease the clash, but it is a Windows setting: suggest it, never change it.
