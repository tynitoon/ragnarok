"""v85 queue (docs/ARC4_V85_PROTOCOL_EN.md §13.3; Part B item 11). Evaluator side; in no run's closure.

  python scripts/queue_v85.py STAGE --list [--U N]          the jobs in claiming order (done or not), no run
  python scripts/queue_v85.py STAGE --launch [N] [--U N]    N hidden detached workers (default 3, at most 4)
  python scripts/queue_v85.py STAGE --worker [--U N]        one worker loop (what --launch starts)

Stages (one job = one scripts/v85_job.py run; outputs under craft_v6_out/v85/<stage>/):
  smoke   G-build smoke runs (§12): breakout and pong, start A, seed 9800, cap 500k, each arm of SMOKE_ARMS whose
          module exists; start built by v85_job --smoke in smoke/<game>/.
  istage  the first U valid units in draw order (U from s0/natural_spikeins.json "U", else --U): V2s 9806-9808 at
          A, 9809-9811 at B; D 1 per start with agent seed E_s (A.10.2).
  g1      F1 units and n (f1_thresholds.json "units", "n", "n_O4", "K_gs"): O_4 (n_O4/2, §13.1 budget rule),
          K*map, K*mac (n/2) runs per start (A 9812.., B 9815..; K*mac only when the other start is valid);
          (g, s) dropped when absent from K_gs or |K_{g,s}| < 2 (A.10.4; never K_g after F1). Caption
          rows are queued after every core row: O_4-dice on 9818 (start A for units at even 0-based positions of
          the A.2 draw order, B at odd ones; skipped if that start is invalid), K*mac-exact on 9818 at every kept start (it uses the start's own lineage).
  g1p     G1-P (§7.2) after a ROOM verdict (g1/g1_verdict.json "G1" == "ROOM"), for its ROOM-K rows ("rows":
          {"o4"|"kmap"|"kmac": {"label": ..}}, as v85_verdict writes them): GF4, GF, K~mac on that row's units,
          starts and seeds.
Worker (§13.3): longest-first dynamic claiming (an atomic claim file per job; a dead worker's claim is taken
over); before each start, under a shared lock: >= 14.5 GB free on min(RAM, commit), >= 2 min since the last
start by any worker, >= 25 min since the last TERMINATE (a job exit code 3, the supervisor's
TerminateProcess code, or a new TERMINATE line in a supervisor log). Jobs run as hidden children
(CREATE_NO_WINDOW, OMP_NUM_THREADS=2, PYTHONHASHSEED=0) and resume from their own checkpoints. Unlimited
passes: a failed job is retried in the next pass; a pass that only meets failed jobs waits 10 min. The
workers are started through WMI Win32_Process.Create with ShowWindow=0, so they survive the app and show
no window. The supervisor (scripts/ram_supervisor.py WORKER tuple) treats v85_job processes as workers.
"""
import argparse
import ctypes
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
V85 = ROOT / "craft_v6_out" / "v85"
QDIR = V85 / "queue"
SUP_LOGS = (ROOT / "craft_v6_out" / "ram_supervisor.log", pathlib.Path(r"F:\dev\ragnarok\craft_v6_out\ram_supervisor.log"))
MIN_FREE_GB, STAGGER_S, BACKOFF_S, IDLE_S, POLL_S = 14.5, 120, 25 * 60, 600, 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SMOKE_GAMES, SMOKE_SEED, SMOKE_CAP = ("breakout", "pong"), 9800, 500_000
SMOKE_ARMS = ("v2s", "d", "o4", "o4_dice", "kmap", "kmac", "kmac_exact", "gf4", "gf", "kmac_tilde")
V2S_SEEDS = {"A": (9806, 9807, 9808), "B": (9809, 9810, 9811)}
G1_SEED0 = {"A": 9812, "B": 9815}
CAPTION_SEED = 9818
PLACEBO = {"o4": "gf4", "kmap": "gf", "kmac": "kmac_tilde"}
ARM_COST = {"o4": 5.0, "o4_dice": 5.0, "o1": 2.0, "d": 0.2}     # emulated frames per billed frame, roughly


@dataclass
class Job:
    name: str
    out: pathlib.Path
    args: list
    est: float
    tier: int = 0                 # 0 core rows, 1 caption rows (queued last, never delay a verdict)


def _load(p):
    return json.load(open(p, encoding="utf-8")) if pathlib.Path(p).exists() else None


def _arm_exists(arm):
    return (ROOT / "ragnarok" / "evaluation" / "arms_v85" / f"{arm}.py").exists()


def _job(stage_dir, g, s, arm, seed, est, tier=0, extra=()):
    name = f"{g}_{s}_{arm}_{seed}"
    out = stage_dir / f"{name}.json"
    return Job(name, out, ["scripts/v85_job.py", g, s, arm, str(seed), "--out", str(out), *extra],
               ARM_COST.get(arm, 1.0) * est, tier)


def _unit_cost(starts, g):
    """Longest-first key: the reference gap if S0 wrote one, else the mean billed of finished V2s runs, else 1."""
    u = starts["units"][g]
    if u.get("ref_gap"):
        return float(u["ref_gap"])
    b = [o["summary"]["billed"] for o in (_load(p) for p in (V85 / "istage").glob(f"{g}_*_v2s_*.json")) if o]
    return float(sum(b) / len(b)) if b else 1.0


def _valid(starts, g, s):
    st = starts["units"][g].get(s)
    return bool(st and st.get("valid"))


def jobs(stage, U=None):
    if stage == "smoke":
        out = []
        for g in SMOKE_GAMES:
            sd = V85 / "smoke" / g
            out += [_job(V85 / "smoke", g, "A", a, SMOKE_SEED, 1.0,
                         extra=("--smoke", str(sd), "--cap", str(SMOKE_CAP))) for a in SMOKE_ARMS if _arm_exists(a)]
        return _order(out)
    starts = _load(V85 / "s0" / "starts.json")
    if starts is None:
        raise SystemExit("no s0/starts.json: run v85_stage0 first")
    if stage == "istage":
        U = U or (_load(V85 / "s0" / "natural_spikeins.json") or {}).get("U")
        if not U:
            raise SystemExit("U unknown: s0/natural_spikeins.json has no 'U'; pass --U")
        ok = [g for g in starts["order"] if len(starts["units"][g]["K_g"]) >= 2
              and any(_valid(starts, g, s) for s in "AB")]
        out = []
        for g in ok[:int(U)]:
            c = _unit_cost(starts, g)
            for s in "AB":
                if _valid(starts, g, s):
                    out += [_job(V85 / "istage", g, s, "v2s", sd, c) for sd in V2S_SEEDS[s]]
                    out.append(_job(V85 / "istage", g, s, "d", starts["units"][g][s]["E"], c))
        return _order(out)
    f1 = _load(V85 / "f1_thresholds.json")
    if f1 is None:
        raise SystemExit("no f1_thresholds.json: G1 launches only after F1 (§6)")
    kgs, n = f1.get("K_gs", {}), int(f1["n"])
    n_o4 = int(f1.get("n_O4") or n)
    keep = lambda g, s: _valid(starts, g, s) and len(kgs.get(g, {}).get(s, [])) >= 2       # noqa: E731  A.10.4
    if stage == "g1":
        out = []
        for g in f1["units"]:
            pos = starts["order"].index(g)                     # 0-based position in the A.2 draw order
            c = _unit_cost(starts, g)
            for s in "AB":
                if not keep(g, s):
                    continue
                for arm in ("o4", "kmap", "kmac"):
                    if arm == "kmac" and not _valid(starts, g, "B" if s == "A" else "A"):
                        continue
                    m = n_o4 if arm == "o4" else n
                    out += [_job(V85 / "g1", g, s, arm, G1_SEED0[s] + i, c) for i in range(m // 2)]
                if s == ("B" if pos % 2 else "A"):
                    out.append(_job(V85 / "g1", g, s, "o4_dice", CAPTION_SEED, c, tier=1))
                out.append(_job(V85 / "g1", g, s, "kmac_exact", CAPTION_SEED, c, tier=1))   # own lineage only
        return _order(out)
    if stage == "g1p":
        v = _load(V85 / "g1" / "g1_verdict.json")
        if v is None:
            raise SystemExit("no g1/g1_verdict.json")
        if v.get("G1") != "ROOM":
            raise SystemExit(f"G1 is {v.get('G1')!r}: G1-P runs only after a ROOM verdict (§7.2)")
        room = [r for r, x in v["rows"].items() if isinstance(x, dict) and x.get("label") == "ROOM-K"]
        out = []
        for j in jobs("g1"):
            g, s, arm, seed = _parse(j.name)
            if j.tier == 0 and arm in room:
                out.append(_job(V85 / "g1p", g, s, PLACEBO[arm], seed, j.est / ARM_COST.get(arm, 1.0)))
        return _order(out)
    raise SystemExit(f"unknown stage {stage}")


def _parse(name):
    """(game, start, arm, seed) of a job name game_start_arm_seed (game and arm may contain '_')."""
    parts = name.split("_")
    seed = int(parts[-1])
    i = next(k for k in range(len(parts) - 2, 0, -1) if parts[k] in ("A", "B"))
    return "_".join(parts[:i]), parts[i], "_".join(parts[i + 1:-1]), seed


def _order(js):
    return sorted(js, key=lambda j: (j.tier, -j.est, j.name))      # longest first, caption rows last


# ---------------------------------------------------------------- machine
def free_gb():
    """min(available physical RAM, available commit) in GB (Windows)."""
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                    ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong * 3)]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.a, m.ap) / 2 ** 30


def alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259                                 # STILL_ACTIVE


class Lock:
    """Cross-worker lock: an O_EXCL file; a lock older than 2 min is stale (its holder died)."""

    def __init__(self, path):
        self.path = pathlib.Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 120:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.5)

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _state(qdir):
    return _load(qdir / "state.json") or {}


def _set_state(qdir, **kw):
    s = dict(_state(qdir), **kw)
    tmp = qdir / "state.json.tmp"
    tmp.write_text(json.dumps(s), encoding="utf-8")
    os.replace(tmp, qdir / "state.json")


def mark_terminate(qdir=QDIR):
    with Lock(qdir / "gate.lock"):
        _set_state(qdir, last_terminate=time.time())


class SupWatch:
    """New TERMINATE lines in the supervisor logs since this worker started."""

    def __init__(self, logs=SUP_LOGS):
        self.logs = {pathlib.Path(p).resolve() for p in logs}
        self.off = {p: (p.stat().st_size if p.exists() else 0) for p in self.logs}

    def poll(self):
        hit = False
        for p in self.logs:
            if not p.exists():
                continue
            size = p.stat().st_size
            if size < self.off[p]:
                self.off[p] = 0                            # rotated or truncated
            if size > self.off[p]:
                with open(p, "rb") as f:
                    f.seek(self.off[p])
                    hit |= b"TERMINATE" in f.read(size - self.off[p])
                self.off[p] = size
        return hit


def gate_ok(state, now, free):
    return (free >= MIN_FREE_GB and now - state.get("last_start", 0) >= STAGGER_S
            and now - state.get("last_terminate", 0) >= BACKOFF_S)


def wait_gate(sw, say, qdir=QDIR):
    while True:
        if sw.poll():
            say("supervisor TERMINATE seen: 25-min back-off")
            mark_terminate(qdir)
        with Lock(qdir / "gate.lock"):
            s, now = _state(qdir), time.time()
            if gate_ok(s, now, free_gb()):
                _set_state(qdir, last_start=now)
                return
        time.sleep(POLL_S)


def _claim_path(qdir, stage, j):
    return qdir / "claims" / f"{stage}_{j.name}.claim"


def claimed(qdir, stage, j):
    p = _claim_path(qdir, stage, j)
    try:
        return alive(int(p.read_text()))
    except (FileNotFoundError, ValueError):
        return False


def claim(qdir, stage, j):
    p = _claim_path(qdir, stage, j)
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if claimed(qdir, stage, j):
                return False
            try:
                p.unlink()                                 # a dead worker's claim
            except FileNotFoundError:
                pass
    return False


def release(qdir, stage, j):
    try:
        _claim_path(qdir, stage, j).unlink()
    except FileNotFoundError:
        pass


def run_job(j, sw, say, qdir=QDIR):
    (V85 / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONHASHSEED="0")
    with open(V85 / "logs" / f"{j.name}.log", "a") as f:
        p = subprocess.Popen([sys.executable] + j.args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env,
                             stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        while True:
            try:
                p.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                if sw.poll():
                    say("supervisor TERMINATE seen: 25-min back-off")
                    mark_terminate(qdir)
    return p.returncode


def worker(stage, U=None, qdir=QDIR):
    qdir.mkdir(parents=True, exist_ok=True)
    log = open(qdir / f"worker_{stage}_{os.getpid()}.log", "a", buffering=1, encoding="utf-8")

    def say(*a):
        log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(map(str, a)) + "\n")
    sw, failed = SupWatch(), set()
    say("worker start", stage, "pid", os.getpid())
    while True:
        todo = [j for j in jobs(stage, U) if not j.out.exists()]
        if not todo:
            say("stage done")
            return
        free = [j for j in todo if j.name not in failed and not claimed(qdir, stage, j)]
        if not free:
            if all(claimed(qdir, stage, j) for j in todo):
                say("nothing left to claim")
                return
            say("pass without a free job; new pass in", IDLE_S, "s")
            failed.clear()
            time.sleep(IDLE_S)
            continue
        j = free[0]
        if not claim(qdir, stage, j):
            continue
        try:
            wait_gate(sw, say, qdir)
            say("start", j.name)
            rc = run_job(j, sw, say, qdir)
            say("end", j.name, "exit", rc)
        finally:
            release(qdir, stage, j)
        if rc == 3:
            say("job terminated (exit 3): 25-min back-off")
            mark_terminate(qdir)
        if rc != 0 or not j.out.exists():
            failed.add(j.name)


def launch(stage, n, extra=()):
    """n hidden detached workers through WMI (ShowWindow=0): they outlive this process and the app."""
    assert 1 <= n <= 4, "3 workers by default, a 4th only with ample margin (§13.2)"
    q = lambda s: "'" + str(s).replace("'", "''") + "'"           # noqa: E731
    for _ in range(n):
        cmd = subprocess.list2cmdline([sys.executable, str(ROOT / "scripts" / "queue_v85.py"), stage, "--worker", *extra])
        ps = ("$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
              f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine={q(cmd)}; "
              f"CurrentDirectory={q(ROOT)}; ProcessStartupInformation=$s}}; \"$($r.ReturnValue) $($r.ProcessId)\"")
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, creationflags=NO_WINDOW)
        print("worker", r.stdout.strip() or r.stderr.strip(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["smoke", "istage", "g1", "g1p"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--launch", type=int, nargs="?", const=3)
    g.add_argument("--worker", action="store_true")
    ap.add_argument("--U", type=int)
    ap.add_argument("--min-free", type=float, default=MIN_FREE_GB,
                    help="launch gate in GB (default 14.5, §13.3); lower only under a dated build note")
    a = ap.parse_args()
    MIN_FREE_GB = a.min_free
    extra = (("--U", str(a.U)) if a.U else ()) + (("--min-free", str(a.min_free)) if a.min_free != 14.5 else ())
    if a.list:
        for j in jobs(a.stage, a.U):
            print("done" if j.out.exists() else "todo", j.tier, round(j.est), j.name)
    elif a.launch:
        launch(a.stage, a.launch, extra)
    else:
        worker(a.stage, a.U)
