"""Family M objects (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2, §9, §12; Part B items 12-13). Evaluator side: reads
the v84 G1b logs as data only (rows: slot, it, t, rise_lag; npz: exp_meta's ADV column) and builds, hashes, writes
and loads the family-M libraries; the arm modules (arms_v85/{x_on, x_frozen, f_m, k_mine, dm_on, p_on, x4_on,
x12_on}.py) turn them into hooks through `arm_hooks`. The miner, DM and the P generator are agent_v85.macros.

Objects (one JSON file each, under <v85>/objects/m/; module docstring of macros for the recipes):
- X (x_<unit>.json): the miner on the sources of the unit: development, a burned unit: the other 40 burned games
  (leave-one-game-out); sealed, any other unit: all 41 burned games minus its near-sibling (pacman excludes
  ms_pacman, frogger excludes freeway, atlantis2 excludes atlantis), i.e. X_41 and its variants (§9).
- K_mine (kmine_<unit>.json): the miner on the unit's own G1b rows (one source: macros docstring, one-source rule).
  The K_mine arm also mines online (choice: §4.4 says "plus mining" for X_on, DM_on and P_on and nothing for
  K_mine, but family W's K_S also learns in-run, and G2a's K:F with F_M = "empty library plus mining" and
  phi = log R(X:F) / log R(K:F) compare libraries only if K mines as X_on does).
- P (p_<unit>.json): for each macro of X(unit), a matched window from the non-ADV rollouts of the same sources
  (macros.placebo_library); its uniforms are row CANON_UNITS.index(unit) of default_rng([9803, 1]).random((74, 32))
  (one row per unit of the canonical order, so a unit's draw never depends on which other units are built;
  choice).
- DM (dm.json): macros.DM, the 32 designer macros.
- Dose draws: X4 (x4_<unit>_d<1|2>.json, development, [9804, 1], 2 draws of 4 games, §8.2), X4s (x4s_<unit>.json,
  sealed, [9804, 2], 1 draw of 4 games) and X12s (x12s_<unit>.json, sealed, [9804, 3], 1 draw of 12 games): the
  miner on that many games drawn without replacement from the unit's sources. Per purpose, U =
  default_rng([9804, p]).random((74, draws, 41)); draw d of a unit = the first n games of its sorted sources in the
  order of argsort(U[CANON_UNITS.index(unit), d - 1, :len(sources)]) (choice: fixed consumption, independent draws
  per unit). One draw for each sealed dose (choice: B1 runs X4 and X12 twice per unit, one run per start, 9853 and
  9856, and the seed table maps two draws only for G2).
- Development dose runs: draw 1 on seeds 9826 (A) and 9829 (B), draw 2 on 9827 (A) and 9830 (B) (§12); a smoke run
  (seed 9800) uses draw 1.
- Dose seed guard (review fix M-7, `dose_kind`): a development (burned) unit runs X4_on only on those four seeds
  and X12_on only at 9826, the F1 closure build, whose explorer refuses to run (`closure_only_explorer`: the hooks
  and the explorer are built exactly, for the closure hash, and no frame is played; X12 is a B1 arm, §8.2, §9); a
  sealed unit runs X4_on and X12_on only at the B1 2-run seeds 9853 and 9856 (§12); a smoke run takes draw 1 at
  seed 9800 only. Any other seed is refused (ValueError), as family W's objects_w_v85.dose_draw.

Baseline windows ([9803, 4]): one stream, default_rng([9803, 4]).random(total), consumed in the canonical order of
runs (sorted burned games, seeds 9712 then 9713), one uniform per non-ADV exp_meta row (rollout) in `it` order; a
rollout with w >= 1 eight-row windows uses window floor(u * w), a rollout with none consumes its uniform and gives
nothing (choice: each source's baseline is the same whatever the source set). Rows of the rollout cut by the cap
(no exp_meta row) have no ADV flag and are not read.

File format: {"format", "kind", "unit", "draw", "seed", "sources", "macros": [[[slot, steps], ...], ...], "n0",
"report", "library_sha256" (SHA-256 of the canonical JSON of the macros), "inputs" ({run: SHA-256 of its npz and
row chunks})}, written atomically (ckpt_v85.write_json); `write_object` returns the file's SHA-256 and
`build_development` / `build_sealed` write manifest.json {file: SHA-256} (the F1 / F2 hashes). `load_object` checks
the format, the library hash and, when given (or listed in the directory's manifest), the file hash. Production
runs refuse a missing object; a smoke run (never a measurement) builds the missing object in its own directory.
The F1 and F2 builds are scripts/v85_objects.py (`--pool dev`, `--pool sealed`), which calls these builders and
hashes every file into its index. `build_sealed` here writes the same family-M sealed set on its own (review fix
M-4): X (the 41 burned games minus the near-sibling, the primary), P, X4s, X12s and DM, plus X_41 with the
near-sibling for pacman, frogger and atlantis2 under variants/all41/ with that directory's own manifest.json.

DM's report (review fix M-3) records that the frozen §4.4 list counts 33 macros against its stated total of 32
and which resolution macros.designer_library uses; that resolution awaits ratification in a dated build note.
"""
import hashlib
import json
import pathlib

import numpy as np

from ragnarok.agent_v85 import macros as M
from ragnarok.evaluation.ckpt_v85 import write_json
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEALED, SECOND

G1B = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84\g1b")
REF = (9712, 9713)                                     # G1b reference runs, read as data only (§12)
BURNED = tuple(sorted(DEV + CALIBRATION))              # the 41 source games (§12)
SEALED_UNITS = tuple(sorted(SEALED + SECOND))
CANON_UNITS = BURNED + SEALED_UNITS                    # canonical order of every per-unit draw (74)
SIBLING = {"pacman": "ms_pacman", "frogger": "freeway", "atlantis2": "atlantis"}
SEED_P, SEED_BASE = (9803, 1), (9803, 4)
DOSE = {"X4": ((9804, 1), 4, 2), "X4s": ((9804, 2), 4, 1), "X12s": ((9804, 3), 12, 1)}   # seed, games, draws
DEV_DRAW = {9826: 1, 9829: 1, 9827: 2, 9830: 2, 9800: 1}
SMOKE_SEED = 9800
CLOSURE_SEED = 9826                                    # v85_calib.arm_closures: the F1 closure build of X12_on
B1_DOSE_SEEDS = (9853, 9856)                           # §12: B1 2-run arms (X4, X12), one per start
DM_NOTE = ("§4.4 states 32 designer macros but its list counts 33; build note 6 (3 November 2026, before any v85 "
           "data) keeps every listed macro: run-then-jump is two 16-step macros (run 8, then the same direction with "
           "extra 8), so DM has 33 macros (review finding M-3).")
FORMAT = "arc4-v85-family-m-object-1"
NAMES = {"X": "x_{u}.json", "K_mine": "kmine_{u}.json", "P": "p_{u}.json", "DM": "dm.json",
         "X4": "x4_{u}_d{d}.json", "X4s": "x4s_{u}.json", "X12s": "x12s_{u}.json"}
_MEMO = {}


# ---------------------------------------------------------------- sources and draws
def sources_of(unit):
    """Development (a burned unit): the other 40 burned games. Sealed: the 41 minus the near-sibling (§9)."""
    if unit in BURNED:
        return [g for g in BURNED if g != unit]
    return [g for g in BURNED if g != SIBLING.get(unit)]


def dose_sources(unit, kind, draw=1, canon=CANON_UNITS, sources=None):
    seed, n, draws = DOSE[kind]
    assert 1 <= draw <= draws, (kind, draw)
    src = sorted(sources_of(unit) if sources is None else sources)
    U = np.random.default_rng(list(seed)).random((len(canon), draws, len(BURNED)))
    u = U[canon.index(unit), draw - 1, :len(src)]
    return sorted(src[i] for i in np.argsort(u, kind="stable")[:n])


def p_uniforms(unit, canon=CANON_UNITS):
    return np.random.default_rng(list(SEED_P)).random((len(canon), M.N_LIB))[canon.index(unit)]


# ---------------------------------------------------------------- G1b logs (data only)
def _sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run_files(game, seed, root=G1B):
    root = pathlib.Path(root)
    return root / f"{game}_{seed}.npz", sorted((root / f"{game}_{seed}_rows").glob("rows_*.npz"))


def run_adv(game, seed, root=G1B):
    with np.load(run_files(game, seed, root)[0]) as z:
        return z["exp_meta"][:, 3].astype(np.int64)


def run_rows(game, seed, root=G1B):
    """(it, t, slot, rise_lag) of one G1b run (the only row fields read)."""
    _, chunks = run_files(game, seed, root)
    out = {k: [] for k in ("it", "t", "slot", "rise_lag")}
    for f in chunks:
        with np.load(f) as z:
            for k in out:
                out[k].append(z[k])
    return tuple(np.concatenate(out[k]).astype(np.int64) if out[k] else np.zeros(0, np.int64)
                 for k in ("it", "t", "slot", "rise_lag"))


def baseline_uniforms(root=G1B, order=BURNED):
    """{(game, seed): uniforms, one per non-ADV rollout}, from one [9803, 4] stream in canonical run order."""
    key = ("U", str(root), tuple(order))
    if key not in _MEMO:
        runs = [(g, s) for g in order for s in REF]
        counts = [int(np.count_nonzero(run_adv(g, s, root) == 0)) for g, s in runs]
        u = np.random.default_rng(list(SEED_BASE)).random(int(sum(counts)))
        off = np.concatenate([[0], np.cumsum(counts)])
        _MEMO[key] = {r: u[off[i]:off[i + 1]] for i, r in enumerate(runs)}
    return _MEMO[key]


def run_windows(it, t, slot, rise_lag, adv, u):
    """One run's 8-row windows (macros docstring): (candidate keys, non-ADV window keys at every position, baseline
    keys). it/t/slot/rise_lag: the run's rows in log order; adv: exp_meta's ADV flag per expansion; u: one uniform
    per non-ADV rollout (adv == 0), in `it` order."""
    it, t, rl = (np.asarray(v, np.int64) for v in (it, t, rise_lag))
    adv = np.asarray(adv, np.int64)
    z = np.zeros(0, np.int64)
    assert len(u) == int(np.count_nonzero(adv == 0)), "one uniform per non-ADV rollout"
    if len(it) < M.ROWS:
        return z, z, z
    s = M.canon(np.asarray(slot, np.int64))
    keys = np.lib.stride_tricks.sliding_window_view(s, M.ROWS) @ M.POW
    first, last = it[:-M.ROWS + 1], it[M.ROWS - 1:]
    same = first == last
    assert np.all(t[M.ROWS - 1:][same] - t[:-M.ROWS + 1][same] == M.HOLD * (M.ROWS - 1)), "rows not contiguous"
    known = last < len(adv)
    flag = np.full(len(last), -1, np.int64)
    flag[known] = adv[last[known]]
    cand = same & (flag == 1) & (rl[M.ROWS - 1:] < 4)
    non = same & (flag == 0)
    w = np.flatnonzero(non)
    base = z
    if len(w):
        r, start, n = np.unique(last[w], return_index=True, return_counts=True)
        pos = np.searchsorted(np.flatnonzero(adv == 0), r)          # each rollout's uniform
        pick = w[start + np.minimum((np.asarray(u)[pos] * n).astype(np.int64), n - 1)]
        base = keys[pick]
    return keys[cand], keys[non], base


def game_stats(game, root=G1B, order=BURNED):
    """One game as a miner source (its runs pooled), plus its non-ADV window table (P) and input hashes."""
    key = ("S", str(root), tuple(order), game)
    if key not in _MEMO:
        U = baseline_uniforms(root, order)
        cand, non, base, inputs = [], [], [], {}
        for s in REF:
            npz, chunks = run_files(game, s, root)
            c, n, b = run_windows(*run_rows(game, s, root), run_adv(game, s, root), U[(game, s)])
            cand.append(c), non.append(n), base.append(b)
            h = hashlib.sha256(_sha_file(npz).encode())
            for f in chunks:
                h.update(_sha_file(f).encode())
            inputs[f"{game}_{s}"] = h.hexdigest()
        src = M.source(np.concatenate(cand), np.concatenate(base))
        nk, nc = M.table(np.concatenate(non))
        _MEMO[key] = dict(src, non_k=nk, non_c=nc, inputs=inputs, game=game)
    return _MEMO[key]


# ---------------------------------------------------------------- objects
def library_sha256(macros):
    return hashlib.sha256(json.dumps([[[int(s), int(n)] for s, n in m] for m in macros]).encode()).hexdigest()


def _object(kind, unit, draw, seed, sources, macros, report, inputs):
    macros = [[[int(s), int(n)] for s, n in m] for m in macros]
    return dict(format=FORMAT, kind=kind, unit=unit, draw=draw, seed=seed, sources=list(sources), macros=macros,
                n0=M.N0, report=report, library_sha256=library_sha256(macros), inputs=inputs)


def _mined(kind, unit, draw, seed, sources, root, order):
    st = [game_stats(g, root, order) for g in sources]
    keys, rows, summ = M.mine(st)
    for r in rows:
        r["dominant"] = None if r["dominant"] is None else sources[r["dominant"]]
    summ["sources_names"] = list(sources)
    inputs = {k: v for s in st for k, v in s["inputs"].items()}
    return _object(kind, unit, draw, seed, sources, [M.macro_of(k) for k in keys], dict(summary=summ, macros=rows),
                   inputs)


def build_x(unit, root=G1B, sources=None, order=BURNED):
    src = sources_of(unit) if sources is None else list(sources)
    return _mined("X", unit, None, None, src, root, order)


def build_k_mine(unit, root=G1B, order=BURNED):
    return _mined("K_mine", unit, None, None, [unit], root, order)


def build_dose(unit, kind, draw=1, root=G1B, sources=None, order=BURNED, canon=CANON_UNITS):
    src = dose_sources(unit, kind, draw, canon, sources)
    return _mined(kind, unit, draw, list(DOSE[kind][0]), src, root, order)


def build_p(unit, x, root=G1B, order=BURNED, canon=CANON_UNITS, u=None):
    """P_on's library for `unit`, matched to the X object `x` on the same sources."""
    st = [game_stats(g, root, order) for g in x["sources"]]
    k, c = np.concatenate([s["non_k"] for s in st]), np.concatenate([s["non_c"] for s in st])
    order_k = np.argsort(k, kind="stable")
    k, c = k[order_k], c[order_k]
    uk, start = np.unique(k, return_index=True)
    pool_c = np.add.reduceat(c, start) if len(k) else np.zeros(0, np.int64)
    u = p_uniforms(unit, canon) if u is None else np.asarray(u, np.float64)
    macros, rows = M.placebo_library(x["macros"], uk, pool_c, u)
    rep = dict(matched_to=x["library_sha256"], exact=int(sum(r["distance"] == 0 for r in rows)), macros=rows,
               pool_sequences=int(len(uk)), pool_windows=int(pool_c.sum()))
    inputs = {kk: v for s in st for kk, v in s["inputs"].items()}
    return _object("P", unit, None, list(SEED_P), x["sources"], macros, rep, inputs)


def build_dm():
    return _object("DM", None, None, None, [], M.DM, dict(size=len(M.DM), note=DM_NOTE), {})


def object_name(kind, unit=None, draw=None):
    return NAMES[kind].format(u=unit, d=draw)


def write_object(directory, obj):
    """Write obj under its name; returns (path, file SHA-256)."""
    p = pathlib.Path(directory) / object_name(obj["kind"], obj["unit"], obj["draw"])
    write_json(p, obj)
    return p, _sha_file(p)


def load_object(path, file_sha256=None):
    path = pathlib.Path(path)
    if file_sha256 is None:
        man = path.parent / "manifest.json"
        if man.exists():
            file_sha256 = json.loads(man.read_text(encoding="utf-8")).get(path.name)
    if file_sha256 is not None and _sha_file(path) != file_sha256:
        raise ValueError(f"{path.name}: file SHA-256 differs from the committed one")
    o = json.loads(path.read_text(encoding="utf-8"))
    if o.get("format") != FORMAT:
        raise ValueError(f"{path.name}: not a family-M object")
    if library_sha256(o["macros"]) != o["library_sha256"]:
        raise ValueError(f"{path.name}: library SHA-256 mismatch")
    for m in o["macros"]:
        n = M.steps(m)
        assert 8 <= n <= 64 and n % 4 == 0 and all(0 <= s < M.BASE for s, _ in m), (path.name, m)
    return o


def _write_all(directory, objs):
    man = {}
    for o in objs:
        p, h = write_object(directory, o)
        man[p.name] = h
    write_json(pathlib.Path(directory) / "manifest.json", man)
    return man


def build_development(units, directory, root=G1B):
    """X, K_mine, P and the two X4 draws per development unit, plus DM; writes them and manifest.json. Returns the
    manifest {file: SHA-256}. (Called by scripts/v85_objects.py at F1; about 15 s and 1.2 GB peak commit per unit
    measured on alien; never in tests on all units.) X12s is also written per development unit: no development
    row runs X12_on, but F1 hashes the closure of every arm module on a development unit (v85_calib.arm_closures,
    `v85_job --closure-only` builds the arm's hooks), and x12_on's build loads its object."""
    objs = [build_dm()]
    for g in units:
        x = build_x(g, root)
        objs += [x, build_k_mine(g, root), build_p(g, x, root)]
        objs += [build_dose(g, "X4", d, root) for d in (1, 2)] + [build_dose(g, "X12s", 1, root)]
    return _write_all(directory, objs)


VARIANT_ALL41 = pathlib.PurePosixPath("variants") / "all41"


def build_sealed(units, directory, root=G1B):
    """Per sealed unit: X on its sources (the 41 burned games minus its near-sibling, sources_of), P, X4s and X12s;
    plus DM; and, for a unit with a near-sibling (pacman, frogger, atlantis2), X_41 on all 41 burned games under
    variants/all41/ with its own manifest.json (§9 "X_41 and its near-sibling variants"; module docstring). Writes
    them and the manifests; returns {path relative to `directory`: SHA-256} (the variants as variants/all41/<file>).
    The F2 build is scripts/v85_objects.py --pool sealed, which writes the same files."""
    objs, var = [build_dm()], []
    for g in units:
        x = build_x(g, root)
        objs += [x, build_p(g, x, root), build_dose(g, "X4s", 1, root), build_dose(g, "X12s", 1, root)]
        if g in SIBLING:
            var.append(build_x(g, root, sources=list(BURNED)))
    out = dict(_write_all(directory, objs))
    if var:
        vman = _write_all(pathlib.Path(directory) / VARIANT_ALL41, var)
        out.update({(VARIANT_ALL41 / n).as_posix(): h for n, h in vman.items()})
    return out


# ---------------------------------------------------------------- arms
def object_dir(ctx):
    return pathlib.Path(ctx.paths["v85"]) / "objects" / "m"


def obtain(ctx, kind, draw=None):
    """The (hash-checked) object of this run's unit; a smoke run builds a missing one in its own directory."""
    if kind == "DM":
        return build_dm()
    p = object_dir(ctx) / object_name(kind, ctx.game, draw)
    if not p.exists():
        if not ctx.smoke:
            raise FileNotFoundError(f"{p}: family-M object missing (built at F1 by scripts/v85_objects.py)")
        if kind == "X":
            o = build_x(ctx.game)
        elif kind == "K_mine":
            o = build_k_mine(ctx.game)
        elif kind == "P":
            o = build_p(ctx.game, obtain(ctx, "X"))
        else:
            o = build_dose(ctx.game, kind, draw)
        write_object(object_dir(ctx), o)
    return load_object(p)


def dose_kind(ctx, sealed_kind):
    """X4_on / X12_on: (kind, draw) of this run; refuses a seed with no dose draw (module docstring, M-7)."""
    seed, dev = int(ctx.seed), ctx.game in BURNED
    name = "X4_on" if sealed_kind == "X4s" else "X12_on"
    if ctx.smoke and seed == SMOKE_SEED:
        return ("X4" if sealed_kind == "X4s" and dev else sealed_kind), 1
    if dev and sealed_kind == "X4s":
        if seed == SMOKE_SEED or seed not in DEV_DRAW:
            raise ValueError(f"{name}: seed {seed} has no dose draw on a development unit (§12: "
                             f"{sorted(k for k in DEV_DRAW if k != SMOKE_SEED)})")
        return "X4", DEV_DRAW[seed]
    if dev:
        if seed != CLOSURE_SEED:
            raise ValueError(f"{name}: no development run (§8.2, §9: B1 only); seed {seed} is not the F1 closure "
                             f"build's {CLOSURE_SEED}")
        return sealed_kind, 1
    if seed not in B1_DOSE_SEEDS:
        raise ValueError(f"{name}: seed {seed} has no dose draw on a sealed unit (§12: {list(B1_DOSE_SEEDS)})")
    return sealed_kind, 1


def closure_only_explorer(reason, make=M.Explorer2m):
    """An explorer factory for a run built only to hash its closure (§13.4, A.10.6; v85_job --closure-only):
    it builds the explorer exactly and its run() refuses, so no frame is played."""
    def build(*a, **k):
        ex = make(*a, **k)

        def refuse(*_a, **_k):
            raise RuntimeError(reason)
        ex.run = refuse
        return ex
    return build


def arm_hooks(ctx, kind=None, mining=True, draw=None):
    """Hooks of a family-M arm: the object's macros as Q5's prior (n0 each), online mining or not. X12_on on a
    development unit (the F1 closure build, `dose_kind`) gets an explorer that refuses to run."""
    o = None if kind is None else obtain(ctx, kind, draw)
    h = M.family_hooks([] if o is None else o["macros"], mining=mining)
    closure_only = bool(kind == "X12s" and ctx.game in BURNED and not ctx.smoke)
    h["object"] = dict(kind=kind, draw=draw, size=0 if o is None else len(o["macros"]),
                       library_sha256=None if o is None else o["library_sha256"],
                       sources=None if o is None else len(o["sources"]), closure_only=closure_only)
    if closure_only:
        h["explorer"] = closure_only_explorer(
            f"X12_on on development unit {ctx.game} at seed {ctx.seed}: built for the F1 closure hash only; X12 "
            f"runs in B1 only (§8.2, §9; seeds 9853/9856, §12)", h["explorer"])
    return h


def captions(hooks, run):
    ex = run["ex"]
    return dict(object=hooks["object"], **hooks["q5"].caption(), bandit_k=int(ex.bandit.k),
                q5_expansions=int(sum(1 for r in ex.exp_meta if int(r[2]) == 4)))
