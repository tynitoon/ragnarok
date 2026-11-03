"""v85 S0 (docs/ARC4_V85_PROTOCOL_EN.md A.10.1 (a)-(f), A.2-A.4, A.15, §12-§13; Part B item 8). Evaluator side,
0 billed frames of any arm; the v84 G1b reference runs 9712 (start A) / 9713 (start B) are read as data only.

  python scripts/v85_stage0.py run [--units alien,zaxxon] [--out DIR] [--sims 1000] [--B 1000]
  python scripts/v85_stage0.py grep [--json P]          # (f) alone, e.g. repeated at F1 / F2

(a) A.3-A.4 per unit, in the A.2 draw order, with start_v85: top lineage (after-step context, else post-reset),
    s1 pickled to s1/<g>_<s>_s1.pkl (+ SHA-256), start identity on a fresh SenseEnv85(g, E), the entry cache
    Body.enter(SenseEnv85(g, E), rng [E, 0], E) pickled to entry/<g>_<E>_body.pkl (E = 9819 at A, 9820 at B; the
    cached Body is loaded back hash-checked), counted rungs K_g (start_v85.counted_rungs), drops, KC-start.
    Resumable per unit: units/<g>.json is written last; a unit whose JSON and files check out is skipped.
(b) floor.json: per valid start of a kept unit, F = 4 Ddepth + 60 ceil(Ddepth / 400) with Ddepth = the lineage
    steps from s1 to its first step with cum >= M_{k_ref}, against the start's natural gap T_r(M_{k_ref}) - T_r(M1);
    KC-floor if the median of gap / F is below 8.
(c) natural_spikeins.json: the counted rungs of the 32 G1b runs timed from their own M1; P(ROOM | x2) by
    scripts/v85_calib.p_room_x2 (n = 4, tau = 0.2, 1.75 / 1.25, [9802, b]) for U in {10, 12, 16} capped at the number
    of kept units; U = the cheapest with P >= 0.8.
(d) dice_g4.json: per segment p = share of expansions between M1 and M_{k_ref} that raised archive-best S,
    G_4 = mean over segments of (1 - (1 - p)^4) / p.
(e) stalls.json (report only, §11 v86): plateaus of >= 1M billed frames without a new best record in the G1b runs.
(f) seed_grep.json: the block 9800-9899 in the project .py files of every git worktree (uncommitted files
    included) and in the committed .py files of every local git branch (finished worktrees are removed, their
    branches remain).
Also starts.json (the layout scripts/v85_job.py loads), rungs.json and s0.json (labels). Every JSON carries the
commit, dirty flag, closure hash, input hashes, claim_header and senses (A.15).

Choices where the text leaves room (each listed in the build result):
- A.2 order: default_rng([9801, 1]).permutation of the 16 eligible units (start_v85.unit_order), drops removed
  afterwards with the order kept. starts.json "order" is that draw order (of the units processed), dropped units
  included, so positions are draw positions (O_4-dice parity); "kept" lists the kept units, the first U are used.
- The entry cache is built only for starts that pass the top-lineage replay and start identity (an invalid start
  is dropped, no arm loads its Body); start identity runs before the entry (fresh emulators, same result).
- K_g is [] for a unit with no valid start. A unit is kept iff it has a valid start and |K_g| >= 2. KC-start
  (fewer than 10 kept units), KC-floor and U are only labelled once all 16 units are done (partial values are
  reported beside them: U_partial, median_ratio).
- (b) the median ratio is over every valid start of the kept units; the median over the first U units is a caption.
- (c) natural cells: a unit's two G1b runs form one 2-run base, placed at each of its valid starts (one run per
  (g, s) cannot be resampled); T(M_k) - T(M1) is floored at 5,000 billed frames (the yardstick's own floor) so
  that a 0 gap (M_k reached by the M1 record) stays finite in log T. All candidate U are reported; if none
  reaches 0.8, U is the largest candidate and "qualified" is false.
- (d) a G1b record is a clone that raised the best cum; it belongs to the expansion e whose billed span
  (end_{e-1}, end_e] contains it (exp_meta column 4; a record after the last row is in the expansion cut by the
  cap). A segment is one reference run at a valid start of a kept unit: expansions after the one that reached M1
  up to the one that reached M_{k_ref}; a segment with no expansion has no p. G_4 (primary) is the mean over the
  segments of the first U kept units; also reported over every kept unit and over the O_4-dice starts only.
- (e) every G1b run of the burned games (pools_v83 DEV and CALIBRATION); a plateau starts at a record and ends at
  the next record or at the run's end ("open"); its state is that record's lineage (rec_offsets / rec_actions).
- (f) worktrees from `git worktree list`; .git, __pycache__, node_modules, site-packages and virtual environments
  skipped; a hit is 98dd not inside a longer number, word or decimal. Hits in this worktree's own v85 files
  (path containing "v85") are owned; any other hit is foreign; PASS iff no foreign hit. Branches: `git grep`
  over every refs/heads/* (*.py), each hit re-checked with the same pattern; a hit is owned only on this
  worktree's own branch in a v85 path (other branches' copies of v85 files are foreign: reviewed by hand).
- (d) dice_g4.json also gives "per_unit" {game: {start: G_4}} for the verdict's O_4-dice caption.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import pickle
import re
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")       # before numpy: BLAS reads it when it loads (§13.2)
import numpy as np                                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import ckpt_v85 as ck                             # noqa: E402
from ragnarok.evaluation import pools_v83 as pools                         # noqa: E402
from ragnarok.evaluation import start_v85 as s0lib                         # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                       # noqa: E402
from scripts import v85_calib as calib                                     # noqa: E402

OUT = ROOT / "craft_v6_out" / "v85" / "s0"
SCRIPT = pathlib.Path(__file__).resolve()
PATHS = ("ragnarok", "scripts", "tests")
STARTS = ("A", "B")
VERSION = 1
FLOOR, CAP = 5_000, 4_000_000
U_CANDIDATES, P_TARGET, KC_UNITS = (10, 12, 16), 0.8, 10
SPIKE = dict(n=4, tau=0.2, tau_pt=1.75, tau_lb=1.25)
KC_FLOOR_RATIO, STALL_MIN = 8.0, 1_000_000
RESTORE, STEP_FRAMES, ROLLOUT = 60, 4, 400
SEED_RE = re.compile(rb"(?<![0-9A-Za-z.])98\d\d(?![0-9A-Za-z])")
SKIP_DIRS = {".git", "__pycache__", "node_modules", "site-packages"}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _prov(inputs):
    p = stamp([str(x) for x in inputs], cwd=str(ROOT), paths=PATHS, full=True)
    h, pairs, foreign = ck.closure_hash(ROOT, [SCRIPT])
    p.update(closure_hash=h, closure_files=pairs, closure_foreign=foreign)
    return p


def _out(obj, inputs):
    return dict(obj, claim_header=ck.CLAIM_HEADER, senses=ck.SENSES, provenance=_prov(inputs))


def _write_bytes(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def _t(recs, m):
    """Billed frames of the first record with cum >= m (None if never)."""
    return next((int(r[0]) for r in recs if r[1] >= m), None)


def _ri(recs, m):
    return next((j for j, r in enumerate(recs) if r[1] >= m), None)


def _gap(recs, lad, k):
    """T_r(M_k) - T_r(M1) on a reference run (None if it misses M_k)."""
    a, b = _t(recs, lad[k]), _t(recs, lad[1])
    return None if a is None or b is None else a - b


def ref_inputs(game, v84=s0lib.V84):
    return [v84 / "bars" / f"{game}_9706.json"] + [v84 / "g1b" / f"{game}_{s}{x}" for s in s0lib.REF_SEEDS.values()
                                                  for x in (".json", ".npz")]


# ---------------------------------------------------------------- (b) and (d): pure helpers
def physics_floor(dd):
    """A.10.1(b): F = 4 Ddepth + 60 ceil(Ddepth / 400)."""
    return STEP_FRAMES * int(dd) + RESTORE * math.ceil(int(dd) / ROLLOUT)


def g4_segment(recs, exp_end, m1, mref):
    """A.10.1(d) on one run: p = share of the expansions after the one reaching M1, up to the one reaching
    M_{k_ref}, that contain a record (raised archive-best S); G_4 = (1 - (1 - p)^4) / p."""
    b = np.array([r[0] for r in recs], np.int64)
    e = np.searchsorted(np.asarray(exp_end, np.int64), b, side="left")
    i1, ir = _ri(recs, m1), _ri(recs, mref)
    if i1 is None or ir is None:
        return dict(e_M1=None, e_ref=None, n=0, raised=0, p=None, G4=None)
    e1, er = int(e[i1]), int(e[ir])
    n = er - e1
    raised = int(len(np.unique(e[(e > e1) & (e <= er)])))
    p = raised / n if n > 0 else None
    return dict(e_M1=e1, e_ref=er, n=int(max(n, 0)), raised=raised, p=p,
                G4=None if p is None else (1.0 - (1.0 - p) ** 4) / p)


def plateaus(recs, end, min_len=STALL_MIN):
    """(e): [(record index, billed from, billed to, open)] of the gaps >= min_len without a new record."""
    out = []
    for i, r in enumerate(recs):
        last = i + 1 == len(recs)
        nxt = int(end) if last else int(recs[i + 1][0])
        if nxt - int(r[0]) >= min_len:
            out.append((i, int(r[0]), nxt, last))
    return out


# ---------------------------------------------------------------- (a): one unit
def _rule(lad, k_star, S1s, refs, K_g):
    """The A.4 rule per k, reported (K_g itself is start_v85.counted_rungs)."""
    rows = {}
    for k in range(2, k_star + 1):
        gaps = [_gap(rs, lad, k) for rs in refs]
        med = None if None in gaps else float(np.median(gaps))
        rows[str(k)] = dict(M=float(lad[k]), above_S1=bool(all(lad[k] > s for s in S1s)), gaps=gaps,
                            median_gap=med, gap_ok=med is not None and med >= s0lib.MIN_GAP,
                            record=[_ri(rs, lad[k]) for rs in refs], counted=k in K_g)
    return rows


def do_unit(game, k_star, out, seeds=s0lib.ENTRY_SEEDS, v84=s0lib.V84):
    """A.3-A.4 (and the per-unit parts of (b), (c), (d)) for one unit; writes units/<game>.json last."""
    t0 = time.time()
    lad = s0lib.ladder(game, v84)
    refs, exp_end = {}, {}
    for st in STARTS:
        refs[st], npz = s0lib.ref_run(game, s0lib.REF_SEEDS[st], v84)
        exp_end[st] = np.asarray(npz["exp_meta"])[:, 4].astype(np.int64)
    starts, s1s, bodies = {}, {}, {}
    for st in STARTS:
        E, t1 = int(seeds[st]), time.time()
        s1 = s0lib.top_lineage(game, s0lib.REF_SEEDS[st], lad, k_star, v84)
        if s1 is None:
            starts[st] = dict(valid=False, E=E, reason="A.3.1: the top lineage matches in neither context")
            continue
        rel, blob = f"s1/{game}_{st}_s1.pkl", pickle.dumps(s1)
        _write_bytes(out / rel, blob)
        idt = s0lib.start_identity(game, E, s1, lad, k_star)
        d = dict(valid=bool(idt["ok"]), E=E, s1=rel, s1_sha256=_sha(blob), context=s1["context"],
                 record=int(s1["record"]), S1=float(s1["S"]), lives=int(s1["lives"]), step=int(s1["step"]),
                 steps_above=int(len(s1["actions"])), identity=idt)
        if idt["ok"]:
            if E not in bodies:                        # one Body per (unit, E); E differs per start (§12)
                rp = f"entry/{game}_{E}_body.pkl"
                c = s0lib.entry_cache(game, E, out / rp)
                bodies[E] = dict(path=rp, sha256=c["sha256"], entry_frames=int(c["entry_frames"]), report=c["entry"])
            bd = bodies[E]
            s0lib.load_body(out / bd["path"], bd["sha256"])     # the cached Body loads, hash-checked
            d.update(entry=dict(path=bd["path"], sha256=bd["sha256"], entry_frames=bd["entry_frames"]),
                     entry_report=bd["report"], body_loads=True)
            s1s[st] = s1
        else:
            d["reason"] = "A.3.4: start identity failed"
        d["wall_s"] = time.time() - t1
        starts[st] = d
    valid = [st for st in STARTS if starts[st]["valid"]]
    S1s = [starts[st]["S1"] for st in valid]
    K_g = s0lib.counted_rungs(lad, k_star, S1s, [refs[s] for s in STARTS]) if valid else []
    kept = bool(valid) and len(K_g) >= 2
    u = dict(version=VERSION, game=game, k_star=int(k_star), ladder=[float(x) for x in lad], entry_seeds=dict(seeds),
             K_g=[int(k) for k in K_g], k_first=min(K_g) if K_g else None, k_ref=max(K_g) if K_g else None, kept=kept,
             drop=None if kept else ("A.3.5: no valid start" if not valid else "A.4: |K_g| < 2"),
             rule=_rule(lad, k_star, S1s, [refs[s] for s in STARTS], K_g),
             natural={st: {str(k): _gap(refs[st], lad, k) for k in range(1, k_star + 1)} for st in STARTS},
             A=starts["A"], B=starts["B"])
    if kept:
        k_ref = u["k_ref"]
        u["ref_gap"] = float(min(CAP, np.mean([u["natural"][st][str(k_ref)] for st in STARTS])))
        for st in valid:                                                 # (b)
            j = s0lib.rung_steps(s1s[st]["cum"], lad, [k_ref])[k_ref]
            dd = int(j[0]) + 1
            gap = u["natural"][st][str(k_ref)]
            F = physics_floor(dd)
            starts[st]["floor"] = dict(k_ref=k_ref, delta_depth=dd, F=F, natural_gap=gap, ratio=gap / F)
        u["g4"] = {st: g4_segment(refs[st], exp_end[st], lad[1], lad[k_ref]) for st in STARTS}       # (d)
    u["wall_s"] = time.time() - t0
    ck.write_json(out / "units" / f"{game}.json", _out(u, ref_inputs(game, v84)))
    return u


def _unit_ok(out, game, seeds):
    """The unit's JSON exists and every file it names checks out (resume: the unit is skipped)."""
    p = out / "units" / f"{game}.json"
    if not p.exists():
        return None
    u = json.load(open(p, encoding="utf-8"))
    if u.get("entry_seeds") != {k: int(v) for k, v in seeds.items()}:
        raise ValueError(f"{p}: built with entry seeds {u.get('entry_seeds')}, asked {dict(seeds)}")
    if u.get("version") != VERSION:
        return None
    for st in STARTS:
        s = u[st]
        files = ([(s["s1"], s["s1_sha256"])] if "s1" in s else []) + ([(s["entry"]["path"], s["entry"]["sha256"])]
                                                                     if s.get("valid") else [])
        for rel, h in files:
            f = out / rel
            if not f.exists() or _sha(f.read_bytes()) != h:
                return None
    return u


# ---------------------------------------------------------------- (c) natural spike-ins
def natural_cells(units, kept):
    """{(g, s): array (2 runs, |K_g|)} of T_r(M_k) - T_r(M1) floored at 5,000, the unit's two G1b runs
    (9712, 9713) as one base placed at each valid start."""
    cells = {}
    for g in kept:
        u = units[g]
        a = np.array([[max(FLOOR, u["natural"][r][str(k)]) for k in u["K_g"]] for r in STARTS], float)
        for s in STARTS:
            if u[s]["valid"]:
                cells[(g, s)] = a
    return cells


def p_room_x2(cells, units, sims, B, seed=calib.SEED_SIM):
    return float(calib.p_room_x2(cells, units, n=SPIKE["n"], tau=SPIKE["tau"], tau_pt=SPIKE["tau_pt"],
                                 tau_lb=SPIKE["tau_lb"], sims=sims, B=B, seed=seed))


def choose_U(kept, p_fn):
    """The cheapest U in {10, 12, 16} (capped at the kept units) with P(ROOM | x2) >= 0.8, else the largest."""
    cands = sorted({min(U, len(kept)) for U in U_CANDIDATES}) if kept else []
    table = []
    for U in cands:
        P = p_fn(kept[:U])
        table.append(dict(U=U, P_room_x2=P, qualifies=bool(P >= P_TARGET)))
    ok = [t["U"] for t in table if t["qualifies"]]
    return (ok[0] if ok else (cands[-1] if cands else None)), bool(ok), table


# ---------------------------------------------------------------- (e) stalls, (f) seed grep
def stall_inventory(v84=s0lib.V84, units16=(), min_len=STALL_MIN):
    burned = set(pools.DEV) | set(pools.CALIBRATION)
    closed = set(pools.RESERVED) | set(pools.EXCLUDED) | set(pools.SEALED) | set(pools.SECOND)
    rows, runs, inputs = [], [], []
    for p in sorted((v84 / "g1b").glob("*_97[0-9][0-9].json")):
        game, seed = p.stem.rsplit("_", 1)
        if game not in burned or game in closed:
            continue
        d = json.load(open(p, encoding="utf-8"))
        bars = v84 / "bars" / f"{game}_9706.json"
        lad = json.load(open(bars, encoding="utf-8"))["ladder"] if bars.exists() else None
        recs, end = d["records"], int(d["summary"]["billed"])
        inputs += [p] + ([bars] if lad else [])
        pl = plateaus(recs, end, min_len)
        for i, b0, b1, last in pl:
            c = float(recs[i][1])
            rung = None if lad is None else max((k for k in range(len(lad)) if lad[k] <= c), default=None)
            rows.append(dict(game=game, seed=int(seed), record=i, billed_from=b0, billed_to=b1, length=b1 - b0,
                             open=last, cum=c, true=float(recs[i][2]), depth=int(recs[i][3]), rung=rung,
                             unit=game in units16))
        runs.append(dict(game=game, seed=int(seed), end=end, records=len(recs), plateaus=len(pl),
                         plateau_frames=int(sum(b1 - b0 for _, b0, b1, _ in pl)), npz_sha256=d.get("npz_sha256")))
    return rows, runs, inputs


def worktrees():
    try:
        r = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True,
                           timeout=30)
        ws = [pathlib.Path(x[len("worktree "):]) for x in r.stdout.splitlines() if x.startswith("worktree ")]
    except Exception:
        ws = []
    return ws or [ROOT]


def _py_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.startswith(".venv")
                       and not os.path.exists(os.path.join(dp, d, "pyvenv.cfg")))
        for f in sorted(fn):
            if f.endswith(".py"):
                yield pathlib.Path(dp) / f


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, timeout=300)


def branch_grep(repo):
    """(f) committed side: [(branch, path, line, text)] of the seed block in the *.py files of every local branch."""
    r = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads"], repo)
    if r.returncode != 0:
        raise RuntimeError(f"git for-each-ref failed in {repo}: {r.stderr.decode(errors='replace')}")
    hits = []
    for br in r.stdout.decode().split():
        g = _git(["grep", "-I", "-n", "-E", r"(^|[^0-9A-Za-z.])98[0-9]{2}([^0-9A-Za-z]|$)", br, "--", "*.py"], repo)
        if g.returncode not in (0, 1):
            raise RuntimeError(f"git grep failed on {br}: {g.stderr.decode(errors='replace')}")
        for ln in g.stdout.splitlines():
            ref, path, num, text = ln.split(b":", 3)
            if SEED_RE.search(text):
                hits.append((br, path.decode("utf-8", "replace"), int(num),
                             text.decode("utf-8", "replace").strip()[:160]))
    return hits


def seed_grep(roots=None, own=ROOT, repo="auto"):
    """(f): the seed block 9800-9899 in the project .py files of every worktree and, through `repo`, of every
    local branch (repo "auto": ROOT when the worktrees are listed from git, none for explicit roots)."""
    own = pathlib.Path(own).resolve()
    if repo == "auto":
        repo = ROOT if roots is None else None
    rep, n_foreign = [], 0
    for root in (roots or worktrees()):
        root = pathlib.Path(root).resolve()
        n, owned, foreign = 0, [], []
        for f in _py_files(root):
            n += 1
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if not SEED_RE.search(data):
                continue
            rel = f.relative_to(root).as_posix()
            if root == own and "v85" in rel.lower():
                owned.append(rel)
                continue
            lines = [[i + 1, ln.decode("utf-8", "replace").strip()[:160]]
                     for i, ln in enumerate(data.splitlines()) if SEED_RE.search(ln)]
            foreign.append(dict(path=rel, lines=lines))
        n_foreign += len(foreign)
        rep.append(dict(root=str(root), files_scanned=n, owned=owned, foreign=foreign))
    branches = None
    if repo is not None:
        cur = _git(["rev-parse", "--abbrev-ref", "HEAD"], own).stdout.decode().strip()
        owned_b, foreign_b = {}, {}
        for br, path, num, text in branch_grep(repo):
            d = owned_b if (br == cur and "v85" in path.lower()) else foreign_b
            d.setdefault(f"{br}:{path}", []).append([num, text])
        branches = dict(repo=str(pathlib.Path(repo).resolve()), current=cur, owned=sorted(owned_b),
                        foreign=[dict(path=k, lines=v) for k, v in sorted(foreign_b.items())])
        n_foreign += len(foreign_b)
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1(f), §3, §12: seed block 9800-9899",
                pattern=SEED_RE.pattern.decode(), worktrees=rep, branches=branches, foreign_files=n_foreign,
                PASS=n_foreign == 0)


# ---------------------------------------------------------------- the stage
def run(out=OUT, units=None, sims=calib.SIMS, B=calib.B_CAL, seeds=s0lib.ENTRY_SEEDS, v84=s0lib.V84, grep_roots=None,
        log=print, sim_seed=calib.SEED_SIM, grep_repo="auto"):
    """S0: (a) for the asked units (all 16 by default) not yet done, then (b)-(f) and the JSONs over every unit done.
    sim_seed: the spike-in simulations' [seed, b] (9802, §12; 9800 in tests)."""
    out = pathlib.Path(out)
    elig = s0lib.eligible_units(v84)
    draw = s0lib.unit_order(elig)
    todo = draw if units is None else [g for g in draw if g in set(units)]
    bad = set(units or ()) - set(elig)
    if bad:
        raise SystemExit(f"not S0 units (§12): {sorted(bad)}")
    done, skipped = {}, []
    for g in draw:
        u = _unit_ok(out, g, seeds)
        if u is None and g in todo:
            log(f"S0 unit {g} (k*={elig[g]}) ...")
            u = do_unit(g, elig[g], out, seeds, v84)
            log(f"  {g}: K_g={u['K_g']} kept={u['kept']} valid={[s for s in STARTS if u[s]['valid']]} "
                f"{u['wall_s']:.0f} s")
        elif u is not None and g in todo:
            skipped.append(g)
        if u is not None:
            done[g] = u
    complete = len(done) == len(draw)
    order = [g for g in draw if g in done]
    kept = [g for g in order if done[g]["kept"]]
    test = {k: int(v) for k, v in seeds.items()} != {k: int(v) for k, v in s0lib.ENTRY_SEEDS.items()}
    uin = [out / "units" / f"{g}.json" for g in order]
    base = dict(complete=complete, units_done=order, entry_seeds=dict(seeds), test_seeds=test)

    # starts.json: the layout scripts/v85_job.py (Starts) and queue_v85 / v85_verdict read
    keys = ("k_star", "ladder", "K_g", "k_first", "k_ref", "kept", "drop", "A", "B")
    su = {g: dict({k: done[g][k] for k in keys}, ref_gap=done[g].get("ref_gap")) for g in order}
    for g in order:
        for st in STARTS:
            su[g][st] = {k: v for k, v in su[g][st].items() if k not in ("identity", "entry_report")}
    starts_obj = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.2-A.3", order=order, kept=kept, units=su,
                      draw_order_all=draw, **base)
    ck.write_json(out / "starts.json", _out(starts_obj, uin))

    # rungs.json and KC-start
    rungs = {g: dict(k_star=done[g]["k_star"], K_g=done[g]["K_g"], k_first=done[g]["k_first"], k_ref=done[g]["k_ref"],
                     kept=done[g]["kept"], drop=done[g]["drop"],
                     S1={st: done[g][st].get("S1") for st in STARTS if done[g][st]["valid"]},
                     identity={st: done[g][st].get("identity") for st in STARTS}, rule=done[g]["rule"])
             for g in order}
    kc_start = (len(kept) < KC_UNITS) if complete else None
    ck.write_json(out / "rungs.json", _out(dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.3.5, A.4", min_gap=s0lib.MIN_GAP,
                                                units=rungs, kept=kept, n_kept=len(kept), KC_start=kc_start, **base),
                                           uin))

    # (c) natural spike-ins and U
    cells = natural_cells(done, kept)
    csha = _sha(json.dumps({f"{g}/{s}": a.tolist() for (g, s), a in sorted(cells.items())}).encode())
    prev = out / "natural_spikeins.json"
    old = json.load(open(prev, encoding="utf-8")) if prev.exists() else None
    if (old and old.get("cells_sha256") == csha and old.get("sims") == sims and old.get("B") == B
            and old.get("sim_seed", calib.SEED_SIM) == sim_seed and old.get("table")):
        U, qual, table = old["U_partial"], old["qualified"], old["table"]
    else:
        U, qual, table = choose_U(kept, lambda us: p_room_x2(cells, us, sims, B, sim_seed))
    first = kept[:U] if U else []
    ck.write_json(prev, _out(dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1(c), A.2", **SPIKE, sims=sims, B=B,
                                  seeds=f"[{sim_seed}, b] (scripts/v85_calib.simulate)", sim_seed=sim_seed,
                                  target=P_TARGET,
                                  candidates=list(U_CANDIDATES), cells={f"{g}/{s}": a.tolist() for (g, s), a in
                                                                        sorted(cells.items())},
                                  cells_sha256=csha, table=table, U=U if complete else None, U_partial=U,
                                  qualified=qual, units=first if complete else None, **base), uin))

    # (b) physics floor
    frows = [dict(game=g, start=st, first_U=g in first, **done[g][st]["floor"]) for g in kept for st in STARTS
             if done[g][st]["valid"]]
    med = float(np.median([r["ratio"] for r in frows])) if frows else None
    medU = [r["ratio"] for r in frows if r["first_U"]]
    kc_floor = bool(med < KC_FLOOR_RATIO) if complete and med is not None else None
    ck.write_json(out / "floor.json", _out(dict(
        rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1(b): F = 4 Ddepth + 60 ceil(Ddepth / 400), ratio = natural gap / F",
        rows=frows, median_ratio=med, median_ratio_first_U=float(np.median(medU)) if medU else None,
        KC_floor=kc_floor, threshold=KC_FLOOR_RATIO, **base), uin))

    # (d) G_4 for the dice caption
    segs = []
    for g in kept:
        pos = draw.index(g)                                  # 0-based draw position: A at even, B at odd (§12)
        for st in STARTS:
            if done[g][st]["valid"]:
                segs.append(dict(game=g, start=st, seed=s0lib.REF_SEEDS[st], first_U=g in first,
                                 dice=g in first and st == ("B" if pos % 2 else "A"), **done[g]["g4"][st]))
    mean = lambda xs: float(np.mean(xs)) if xs else None         # noqa: E731
    g4 = lambda f: mean([s["G4"] for s in segs if f(s) and s["G4"] is not None])   # noqa: E731
    ck.write_json(out / "dice_g4.json", _out(dict(
        rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1(d): G_4 = mean over segments of (1 - (1 - p)^4) / p",
        segments=segs, G4=g4(lambda s: s["first_U"]), G4_all_kept=g4(lambda s: True),
        G4_dice_starts=g4(lambda s: s["dice"]),
        per_unit={g: {s["start"]: s["G4"] for s in segs if s["game"] == g} for g in kept},
        **base), uin + [x for g in kept for x in ref_inputs(g, v84)]))

    # (e) stall inventory (report only)
    rows, runs, sin = stall_inventory(v84, set(elig))
    ck.write_json(out / "stalls.json", _out(dict(
        rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1(e), §11: plateaus of >= 1M billed frames without a new best record",
        min_len=STALL_MIN, n=len(rows), plateaus=rows, runs=runs, **base), sin))

    # (f) seed grep
    grep = seed_grep(grep_roots, repo=grep_repo)
    ck.write_json(out / "seed_grep.json", _out(grep, []))

    files = {f: _sha((out / f).read_bytes()) for f in ("starts.json", "rungs.json", "natural_spikeins.json",
                                                       "floor.json", "dice_g4.json", "stalls.json", "seed_grep.json")}
    summ = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.1", kept=kept, n_kept=len(kept), KC_start=kc_start,
                KC_floor=kc_floor, median_floor_ratio=med, U=U if complete else None, U_partial=U,
                U_qualified=qual, G4=g4(lambda s: s["first_U"]), seed_grep_PASS=grep["PASS"], skipped=skipped,
                files=files, **base)
    ck.write_json(out / "s0.json", _out(summ, [out / f for f in files]))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "grep"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--units", help="comma-separated S0 units (default: all 16, in draw order)")
    ap.add_argument("--sims", type=int, default=calib.SIMS)
    ap.add_argument("--B", type=int, default=calib.B_CAL)
    ap.add_argument("--json", help="grep: output JSON (default OUT/seed_grep.json)")
    a = ap.parse_args(argv)
    if a.cmd == "grep":
        g = seed_grep()
        ck.write_json(a.json or pathlib.Path(a.out) / "seed_grep.json", _out(g, []))
        print(json.dumps(dict(PASS=g["PASS"], foreign_files=g["foreign_files"])))
        return
    s = run(a.out, units=a.units.split(",") if a.units else None, sims=a.sims, B=a.B)
    print(json.dumps({k: s[k] for k in ("complete", "n_kept", "KC_start", "KC_floor", "U", "U_partial", "G4",
                                        "seed_grep_PASS")}))


if __name__ == "__main__":
    main()
