"""One v85 run (unit, start, arm, seed) (docs/ARC4_V85_PROTOCOL_EN.md A.5, §13.3-13.4, A.15; Part B item 7).
Evaluator side. Never edited after the I-stage launch: new arms are new files in ragnarok/evaluation/arms_v85/.

  python scripts/v85_job.py UNIT START ARM SEED [--out OUT] [--v85 DIR] [--ckpt-s S] [--keep-ckpt]
  python scripts/v85_job.py breakout A v2s 9800 --smoke DIR [--cap N] [--out OUT] [--ckpt-s S]

Run (A.5): SenseEnv85(g, seed=E_s, trace=Trace85(stop_at=M_{k_ref})); start_from(s1): reset, one NOOP, restore
s1 (billed 60), cum = S(s1), clock start; cap = clock start + 4M billed; the start's cached Body (SHA-256
checked); the arm's hooks from arms_v85/<ARM>.py; Explorer2s with a ckpt_v85.Checkpointer at its expansion
boundaries (every <= 5 min of wall). A checkpoint `<OUT>.ckpt` of the same run (same key) is resumed
automatically (§13.3); an existing OUT means the run is done. The checkpoint is removed once OUT is written
(one rolling checkpoint per job, §13.2), unless --keep-ckpt.

Start data (written by scripts/v85_stage0.py; `build_smoke` below writes the same layout):
  <s0>/starts.json = {"order": [units in A.2 draw order],
                      "units": {game: {"k_star": int, "ladder": [M_0..M_10], "K_g": [counted rungs, A.4],
                                       "A": START or null, "B": START or null}}}
  START = {"valid": bool, "E": int, "s1": file (relative to s0/) of the pickled s1 dict of start_v85.top_lineage,
           "s1_sha256": SHA-256 of that file, "entry": {"path": Body pickle (relative to s0/), "sha256", "entry_frames"}}
  <v85>/f1_thresholds.json, if present: {"units": [...], "K_gs": {game: {"A": [...], "B": [...]}}} gives the run's
  rungs K_{g,s} (A.10.4); otherwise rungs = K_g. k_first = min(rungs); k_ref = max(K_g), the evaluator stop (A.4).

Run JSON (shared schema): game, start, arm, seed, cap, k_ref, ladder, S1, rungs, records [[billed since clock, cum,
true, depth, oracle frames since clock]], T and T_total for every ladder rung k >= 1 (null if missed), replay
[{k, ok, replayed, recorded}] for the first records reaching M_{k_first} and the highest reached counted rung (of
the run's rungs and, if higher, of K_g), replayed by Trace85.replay85 on a fresh emulator E_s (reset, NOOP,
restore s1, the lineage's actions); ok = cum equal exactly, else interruption = true. Then summary (explorer),
instrument (arm captions), entry, oracle_frames, wall_s, peak_ws_mb, peak_commit_mb, claim_header, senses,
provenance (commit, dirty, closure hash §13.4, input SHA-256s, one segment per process with its launch stamp and
whether its closure agrees with the final one, checkpoint stats); extras E, k_first, rungs_source, billed,
stopped, s1_sha256, aborted (null, or the message of an exception listed in the arm's ABORTS: the run then
ends there and is written, never retried), and exp = {n, meta_sha256, feat_sha256}: the explorer's exp_meta (N x 8, float64) and
exp_feat (N x 44, float32), kept for the comparator certificate of §13.4, are written to the sidecar `<OUT>.npz`
(before OUT) and hashed on their array bytes. VOLATILE fields (wall, memory, provenance) are the only ones that
may differ between two runs of the same (unit, start, arm, seed), interrupted or not (G-build tests 4-5).

Smoke (seed 9800 only, §12; never a measurement): `--smoke DIR` builds once, in DIR, a start of the game by the
S0 procedure on the v84 G1b reference logs (read as data only): both starts' s1 by start_v85.top_lineage and one
entry cache on SenseEnv85(g, 9800), shared by both starts (E = 9800). Smoke rungs: K_g = every k in 2..k* with
M_k above both S(s1) (A.4's 50k gap rule is not applied so that small smoke caps cross rungs).
"""
import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")      # BLAS thread buffers: ~0.4 GB of commit otherwise (§13.2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v84.explorer2 import N_EXP_FEAT                        # noqa: E402
from ragnarok.agent_v85.explorer2s import Explorer2s                       # noqa: E402
from ragnarok.evaluation import arms_v85                                   # noqa: E402
from ragnarok.evaluation import ckpt_v85 as ck                             # noqa: E402
from ragnarok.evaluation import start_v85 as s0lib                         # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                       # noqa: E402
from ragnarok.evaluation.senses_v85 import SenseEnv85, Spent, Trace85      # noqa: E402

V85 = ROOT / "craft_v6_out" / "v85"
CAP = 4_000_000
SMOKE_SEED = 9800
PATHS = ("ragnarok", "scripts", "tests")
VOLATILE = ("wall_s", "peak_ws_mb", "peak_commit_mb", "provenance")
JOB = pathlib.Path(__file__).resolve()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class Starts:
    """starts.json and its hash-checked files."""

    def __init__(self, s0):
        self.s0 = pathlib.Path(s0)
        self.path = self.s0 / "starts.json"
        self.data = json.load(open(self.path, encoding="utf-8"))

    def unit(self, game):
        return self.data["units"][game]

    def start(self, game, st):
        s = self.unit(game).get(st)
        return s if s and s.get("valid") else None

    def file(self, rel):
        return self.s0 / rel

    def load_s1(self, game, st):
        s = self.start(game, st)
        if s is None:
            return None
        blob = self.file(s["s1"]).read_bytes()
        assert _sha(blob) == s["s1_sha256"], f"s1 hash mismatch: {game} {st}"
        return pickle.loads(blob)

    def load_body(self, game, st):
        e = self.start(game, st)["entry"]
        return s0lib.load_body(self.file(e["path"]), e["sha256"])


def build_smoke(d, game="breakout"):
    """The S0 layout of one smoke game in `d`, built once (module docstring)."""
    d = pathlib.Path(d)
    if (d / "starts.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(s0lib.V84 / "room_verdict_v84.json", encoding="utf-8"))["rows"]
    k_star = int(next(r["k_star"] for r in rows if r["game"] == game))
    lad = s0lib.ladder(game)
    entry = s0lib.entry_cache(game, SMOKE_SEED, d / f"{game}_{SMOKE_SEED}_body.pkl")
    unit, S1s = dict(k_star=k_star, ladder=lad, K_g=[], A=None, B=None), []
    for st in ("A", "B"):
        s1 = s0lib.top_lineage(game, s0lib.REF_SEEDS[st], lad, k_star)
        if s1 is None:
            continue
        blob = pickle.dumps(s1)
        (d / f"{game}_{st}_s1.pkl").write_bytes(blob)
        unit[st] = dict(valid=True, E=SMOKE_SEED, s1=f"{game}_{st}_s1.pkl", s1_sha256=_sha(blob),
                        entry=dict(path=pathlib.Path(entry["path"]).name, sha256=entry["sha256"],
                                   entry_frames=int(entry["entry_frames"])))
        S1s.append(float(s1["S"]))
    unit["K_g"] = [k for k in range(2, k_star + 1) if S1s and all(lad[k] > s for s in S1s)]
    ck.write_json(d / "starts.json", dict(order=[game], units={game: unit}, smoke=True, seed=SMOKE_SEED))
    return d


def _first(recs, m):
    return next((r for r in recs if r[1] >= m), None)


def run(game, start, arm, seed, out=None, v85=V85, s0=None, cap=CAP, ckpt_s=300.0, smoke=False, keep_ckpt=False,
        observe=None):
    """One run; returns the run JSON dict. `observe(d)` (tests, diagnostics) sees the built run before it starts
    and must not change it."""
    t0 = time.time()
    v85 = pathlib.Path(v85)
    s0 = pathlib.Path(s0) if s0 else v85 / "s0"
    seed = int(seed)
    if smoke:
        assert seed == SMOKE_SEED, "smoke runs use seed 9800 only (§12)"
    else:
        assert cap == CAP, "cap = 4M billed frames from s1 (§5); smaller caps are for smoke runs only"
    out = pathlib.Path(out) if out else v85 / "runs" / f"{game}_{start}_{arm}_{seed}.json"
    if out.exists():
        return json.load(open(out, encoding="utf-8"))
    S = Starts(s0)
    u, me = S.unit(game), S.start(game, start)
    if me is None:
        raise SystemExit(f"{game} {start}: invalid start (A.3)")
    other = "B" if start == "A" else "A"
    lad, K_g = u["ladder"], sorted(int(k) for k in u["K_g"])
    if len(K_g) < 2:
        raise SystemExit(f"{game}: fewer than 2 counted rungs (A.4)")
    f1p = v85 / "f1_thresholds.json"
    f1 = json.load(open(f1p, encoding="utf-8")) if f1p.exists() else None
    if f1 is not None and game in f1.get("K_gs", {}):
        rungs, src = sorted(int(k) for k in f1["K_gs"][game][start]), "f1"
    else:
        rungs, src = K_g, "s0"
    k_first, k_ref, E = min(rungs), max(K_g), int(me["E"])
    s1, other_s1 = S.load_s1(game, start), S.load_s1(game, other)
    inputs = [S.path, S.file(me["s1"]), S.file(me["entry"]["path"])]
    inputs += [S.file(u[other]["s1"])] if other_s1 is not None else []
    inputs += [f1p] if f1 is not None else []
    key = dict(game=game, start=start, arm=arm, seed=seed, E=E, cap=int(cap), s1_sha256=me["s1_sha256"],
               body_sha256=me["entry"]["sha256"], rungs=rungs, k_ref=k_ref)
    ck_path = out.with_suffix(".ckpt")
    st = ck.load(ck_path, key)
    prior = st["meta"] if st is not None else dict(segments=[], wall_s=0.0, peak_ws_mb=0.0, peak_commit_mb=0.0)
    seg = dict(stamp=stamp((), cwd=str(ROOT), paths=PATHS), resumed_at_exp=None if st is None else st["exp"],
               closure_pairs=None)

    tr = Trace85(stop_at=lad[k_ref])
    env = SenseEnv85(game, seed=E, trace=tr)
    obs = lives = None
    if st is None:
        obs, lives = env.start_from(s1["blob"], s1["S"], cap=cap)
    body = S.load_body(game, start)
    units = list(f1["units"]) if f1 is not None and "units" in f1 else list(S.data["order"])
    ctx = arms_v85.Ctx(game=game, start=start, seed=seed, E=E, s1=s1, other_s1=other_s1, body=body, ladder=lad,
                       rungs=rungs, k_first=k_first, k_ref=k_ref, cap=int(cap), units=units,
                       paths=dict(v85=str(v85), s0=str(s0), starts=str(S.path), out=str(out)), starts=S.data,
                       load_s1=S.load_s1, load_body=S.load_body, smoke=smoke)
    mod, hooks = arms_v85.build(arm, ctx)
    cp = ck.Checkpointer(ck_path, every_s=ckpt_s, key=key)
    make = hooks["explorer"] or Explorer2s
    ex = make(body, budget=cap, seed=seed, steer=hooks["steer"], q5=hooks["q5"], chooser=hooks["chooser"],
              boundary=cp, **hooks["kwargs"])
    objs = arms_v85.stateful(hooks)

    def meta():
        seg["closure_pairs"] = ck.closure_hash(ROOT, [JOB])[1]
        ws, cm = ck.peak_mem_mb()
        return dict(segments=prior["segments"] + [seg], wall_s=prior["wall_s"] + time.time() - t0,
                    peak_ws_mb=max(prior["peak_ws_mb"], ws), peak_commit_mb=max(prior["peak_commit_mb"], cm))
    cp.attach(env, tr, objs, meta)
    if st is not None:
        ck.restore(st, env, tr, ex, objs)
    if observe is not None:
        observe(dict(env=env, trace=tr, ex=ex, hooks=hooks, ckpt=cp, ctx=ctx))
    aborted = None
    try:
        if st is None:
            summ = ex.run(env, obs, lives, s1["S"], spent=Spent)
        else:
            summ = ex.run(env, spent=Spent, resume=True)
    except tuple(getattr(mod, "ABORTS", ())) as err:          # e.g. O_4-dice's DiceMismatch (A.7): counted
        aborted = f"{type(err).__name__}: {err}"
        summ = dict(expansions=ex.exp_n, cells=len(ex.arch), best_S=ex.best_S, evicted=ex.arch.evicted,
                    billed=env.billed - ex.b0, aborted=True)

    # ---------------------------------------------------------------- times and replay confirmation (A.5)
    recs = tr.records
    T, T_total = {}, {}
    for k in range(1, len(lad)):
        r = _first(recs, lad[k])
        T[str(k)] = None if r is None else int(r[0])
        T_total[str(k)] = None if r is None else int(r[0] + r[5])
    hit = lambda ks: [k for k in ks if T[str(k)] is not None]          # noqa: E731
    conf = sorted({k for k in [k_first] + hit(rungs)[-1:] + hit(K_g)[-1:] if T[str(k)] is not None})
    replay = []
    for k in conf:
        r = _first(recs, lad[k])
        got = tr.replay85(r[2])
        replay.append(dict(k=k, ok=bool(got[0] == r[1]), replayed=[float(got[0]), float(got[1]), int(got[2])],
                           recorded=[float(r[1]), float(r[3]), int(r[4])]))
    inst = arms_v85.captions(mod, hooks, dict(ex=ex, env=env, trace=tr, ctx=ctx))
    em = np.array(ex.exp_meta, np.float64).reshape(-1, 8)
    ef = np.array(ex.exp_rows, np.float32).reshape(-1, N_EXP_FEAT)
    npz = out.with_suffix(".npz")
    with open(npz.with_name(npz.name + ".tmp"), "wb") as f:
        np.savez_compressed(f, exp_meta=em, exp_feat=ef)
    os.replace(npz.with_name(npz.name + ".tmp"), npz)
    exp = dict(n=len(em), meta_sha256=_sha(em.tobytes()), feat_sha256=_sha(ef.tobytes()))

    # ---------------------------------------------------------------- provenance (§13.4, A.15)
    h, pairs, foreign = ck.closure_hash(ROOT, [JOB])
    final = {tuple(p) for p in pairs}
    end = stamp(inputs, cwd=str(ROOT), paths=PATHS, full=True)
    segs = prior["segments"] + [seg]
    seg_out = [dict(code_sha=s["stamp"]["code_sha"], dirty=bool(s["stamp"]["dirty"]), launched=s["stamp"]["launched"],
                    resumed_at_exp=s["resumed_at_exp"],
                    closure_agrees=s["closure_pairs"] is None or {tuple(p) for p in s["closure_pairs"]} <= final)
               for s in segs]
    seg_out[-1]["closure_agrees"] = True
    prov = dict(commit=end["code_sha"], dirty=bool(end["dirty"] or any(s["dirty"] for s in seg_out)),
                closure_hash=h, closure_files=pairs, closure_foreign=foreign,
                closure_consistent=bool(all(s["closure_agrees"] for s in seg_out)
                                        and all(s["code_sha"] == end["code_sha"] for s in seg_out) and not foreign),
                code_tree=end["code_tree"], inputs=end["inputs"], segments=seg_out, ckpt=cp.stats(),
                launched=segs[0]["stamp"]["launched"], ended=end["launched"], python=sys.version.split()[0],
                hashseed=os.environ.get("PYTHONHASHSEED"))
    ws, cm = ck.peak_mem_mb()
    o = dict(game=game, start=start, arm=arm, seed=seed, cap=int(cap), k_ref=k_ref, ladder=lad, S1=float(s1["S"]),
             rungs=rungs, records=[[int(b), float(c), float(t), int(d), int(of)] for b, c, _, t, d, of in recs],
             T=T, T_total=T_total, replay=replay, interruption=not all(x["ok"] for x in replay), summary=summ,
             instrument=inst, entry=dict(sha256=me["entry"]["sha256"], entry_frames=int(me["entry"]["entry_frames"])),
             oracle_frames=int(env.oracle_frames), wall_s=prior["wall_s"] + time.time() - t0,
             peak_ws_mb=max(prior["peak_ws_mb"], ws), peak_commit_mb=max(prior["peak_commit_mb"], cm),
             claim_header=ck.CLAIM_HEADER, senses=ck.SENSES, provenance=prov, E=E, k_first=k_first,
             rungs_source=src, billed=int(env.billed_since_start), stopped=bool(tr.stopped), s1_sha256=me["s1_sha256"],
             exp=exp, aborted=aborted)
    ck.write_json(out, o)
    if not keep_ckpt:
        cp.remove()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("unit")
    ap.add_argument("start", choices=["A", "B"])
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--v85", default=str(V85))
    ap.add_argument("--ckpt-s", type=float, default=300.0, help="checkpoint period in s of wall (<= 300)")
    ap.add_argument("--keep-ckpt", action="store_true")
    ap.add_argument("--smoke", metavar="DIR", help="smoke run on seed 9800 from a start built in DIR")
    ap.add_argument("--cap", type=int, default=CAP, help="smoke runs only")
    a = ap.parse_args(argv)
    if a.smoke:
        d = build_smoke(a.smoke, a.unit)
        out = a.out or str(d / f"{a.unit}_{a.start}_{a.arm}_{a.seed}.json")
        o = run(a.unit, a.start, a.arm, a.seed, out=out, v85=d, s0=d, cap=a.cap, ckpt_s=a.ckpt_s, smoke=True,
                keep_ckpt=a.keep_ckpt)
    else:
        if a.cap != CAP:
            ap.error("--cap is for smoke runs only")
        o = run(a.unit, a.start, a.arm, a.seed, out=a.out, v85=a.v85, ckpt_s=a.ckpt_s, keep_ckpt=a.keep_ckpt)
    print(json.dumps(dict(T=o["T"], interruption=o["interruption"], summary=o["summary"], wall_s=round(o["wall_s"]),
                          peak_commit_mb=round(o["peak_commit_mb"]))))


if __name__ == "__main__":
    main()
