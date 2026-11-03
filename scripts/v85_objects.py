"""v85 family objects (docs/ARC4_V85_PROTOCOL_EN.md Part B item 13; §4.4, §8.2, §9, §12, A.10.6). Evaluator side.

Builds and hashes the family objects with the frozen builders of ragnarok/evaluation/objects_w_v85.py (family W,
"which proposal to bill") and ragnarok/evaluation/objects_m_v85.py (family M, "how to move"), from the v84 G1b
logs (read as data only, every input hash-checked by the builders), and writes an index whose keys are the F1
hashes of A.10.6 (`scripts/v85_calib.py f1 --hashes <index>`: family_code, development_objects, kappa).

  python scripts/v85_objects.py [--units U ..] [--out V85] [--skip-kappa]           # development (F1)
  python scripts/v85_objects.py --pool sealed --out SEALED_ROOT [--units U ..]
                                [--kappa-from V85/objects/w/kappa.json] [--sealed-primary minus-sibling|all41]
  python scripts/v85_objects.py [--pool P] --index-only [--out DIR]                 # the index from what exists

Development (`--pool dev`, the default; F1, before G1). Units: the 16 room development games with k* >= 4 in the
A.2 order of default_rng([9801, 1]) (§12; the rule of start_v85.unit_order(eligible_units()), recomputed here
from the same room verdict file so that the builder's closure holds no emulator module). Per unit, LOGO over the
other 40 burned games:
- W (R/w/<arm>/<unit>[_d<k>].npz + .json sidecars): F_S, X_S, K_S, P_S, U_S, T, S, the two X4_S draws
  ([9804, 1, i_unit], objects_w_v85.dose_dev); and X12_S's closure object (below).
- M (R/m/*.json): X (x_<unit>), K_mine, P_on, the two X4 draws ([9804, 1]), X12s's closure object; DM once.
- kappa by the frozen rule (§4.4, v84 A.9 with the edge rule) on the 4 calibration games, once: R/w/kappa.json.
Sealed (`--pool sealed`, F2, same frozen code): the 17 non-hard sealed games plus the 9 second-round games (the
top-up pool, so that a second round needs no new build after F2); the 7 hard-exploration games stay unopened.
Per unit: W F_S, X_S, P_S, U_S, T, S, X4_S ([9804, 2, i_unit]) and X12_S ([9804, 3, i_unit]), one draw each per
unit (objects_w_v85.dose_sealed; review fix W-1/M-5: the draws were not indexed by unit); M X, P_on, X4s, X12s; DM.
No K object (the K arms are development only). kappa is the F1 kappa.json, copied byte for byte from --kappa-from
and hash-checked against the F1-committed kappa SHA-256 (review fix M-6: f1_thresholds.json's hashes beside it,
else the development index; refused when they differ, and when there is none outside a test build) before any
unit is built (the sealed arms read it from their own root). The sealed index is complete only if its code
identity equals F1's family_code and its kappa.json F1's kappa (same_code_as_f1, same_kappa_as_f1).

Files (R = <out>/objects, the root the arms read, objects_w_v85.objects_root / objects_m_v85.object_dir):
  R/w/..., R/m/...        the objects and the families' own manifests (rewritten after every unit)
  R/build/<pool>/<unit>.json, R/build/<pool>/_kappa.json   completion markers (resume)
  R/index.json (dev) or R/index_sealed.json (sealed)        the index
Resume (per unit): a unit is skipped when its marker exists, was written under the same code identity and
options, and every file it lists exists with the listed SHA-256; otherwise it is rebuilt (the builders are
deterministic: a rebuild gives the same M bytes and the same W array contents, objects_w_v85.content_sha256; only
the W npz zip timestamps and the npz file hash in a W sidecar differ). Code identity = the closure hash
(ckpt_v85.closure_hash, §13.4) of this script, every project module loaded once the builders' modules are
imported up front, and the family files (agent_v85 macros and critic_s, the objects builders and every family arm
module); the index ignores a marker of another code identity (listed as stale) and is complete only if the
closure did not grow during the build. Killing the process loses at most the unit in progress.

Index keys (F1, A.10.6 "SHA-256 of ... family code, development objects and kappa"):
- family_code: {sha256 (the code identity above), files {path: SHA-256}, family_files, arm_modules, foreign};
- development_objects (sealed: sealed_objects): {sha256 over the sorted "path SHA-256" lines of every object file
  (W npz and sidecars, M JSON, kappa.json), files {path: SHA-256}, w (objects_w_v85.hashes: the content-hash
  aggregate), m_manifest (SHA-256 of R/m/manifest.json), units};
- kappa: {sha256 of R/w/kappa.json, kappa, derived, extended, kappas, n_curves, skipped};
- objects_build: pool, complete, expected and built units, missing, extra and stray files, options, peak memory.
An incomplete build (a unit of the pool missing, or kappa missing) writes the index with family_code,
development_objects and kappa set to null, so `v85_calib.py f1` refuses it ("F1 hashes lack ...").

Choices (recorded; the frozen text leaves them open):
- X12 closure objects in development: no development row runs X12_S or X12_on, but F1's closures are hashed for
  every arm module on a development unit (v85_calib.arm_closures, `v85_job --closure-only` builds the arm's hooks
  at seed 9826). Family M already writes x12s_<unit>.json for development units; family W gets the same: X12_S's
  recipe ([9804, 3, i_unit], 12 of the unit's heads, one draw) as x12_s/<unit>_d1, and objects_w_v85.DOSE maps
  seed 9826 to draw 1 for x12_s on a development unit only. Such a run is built for the closure hash and its
  explorer refuses to run (review fix W-3/M-7; objects_w_v85.closure_only_explorer, objects_m_v85 likewise).
- Sealed primary: §4.4 says "all 41 minus near-siblings" for the sealed sources, §9 says the frozen code builds
  "X_41 and its near-sibling variants". Default (--sealed-primary minus-sibling): the files the arms load use the
  41 burned games minus the unit's near-sibling (pacman: ms_pacman, frogger: freeway, atlantis2: atlantis), as
  family M's frozen builder (objects_m_v85.sources_of) does, and the other variant (X_41 with the sibling) is
  also built for those three units' X objects: W x_s/<unit>~all41.npz, M m/variants/all41/x_<unit>.json. With
  --sealed-primary all41 the roles swap (W ~nosib, M variants/nosib). F2 decides; both are hashed in the index.
- The sealed root is never the development root (--out is required for --pool sealed and must differ from the
  development output root).
- Test-only knobs (Python API only, never on the command line): `games` restricts the source pool and `n_max`
  truncates the W streams; an index built with them says so (objects_build.options) and is never complete.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")       # before numpy (§13.2)
import numpy as np                                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import ckpt_v85 as ck              # noqa: E402
from ragnarok.evaluation import g0_stages_v84               # noqa: E402,F401  (builders' lazy imports, up front)
from ragnarok.evaluation import objects_m_v85 as OM         # noqa: E402
from ragnarok.evaluation import objects_w_v85 as W          # noqa: E402
from ragnarok.evaluation.pools_v83 import HARD_EXPLORATION, SEALED, SECOND  # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp       # noqa: E402

V85 = ROOT / "craft_v6_out" / "v85"
V84 = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84")
SCRIPT = pathlib.Path(__file__).resolve()
PATHS = ("ragnarok", "scripts", "tests")
POOLS = ("dev", "sealed")
FORMAT = "v85-objects-index-1"
RULE = "docs/ARC4_V85_PROTOCOL_EN.md Part B item 13; §4.4, §8.2, §9, §12; A.10.6 (F1 family code, objects, kappa)"
FAMILY_ARMS = ("f_s", "x_s", "k_s", "p_s", "u_s", "t", "s", "gf", "x4_s", "x12_s",          # FAM["W"]
               "k_mine", "f_m", "x_on", "x_frozen", "dm_on", "p_on", "x4_on", "x12_on")       # FAM["M"] (+ S)
FAMILY_FILES = ("ragnarok/agent_v85/macros.py", "ragnarok/agent_v85/critic_s.py",
                "ragnarok/evaluation/objects_w_v85.py", "ragnarok/evaluation/objects_m_v85.py") + \
    tuple(f"ragnarok/evaluation/arms_v85/{a}.py" for a in FAMILY_ARMS)
SIBLING = dict(OM.SIBLING)                                 # pacman: ms_pacman, frogger: freeway, atlantis2: atlantis
SEALED_UNITS = tuple(g for g in SEALED if g not in HARD_EXPLORATION) + tuple(SECOND)
PRIMARY = ("minus-sibling", "all41")
VARIANT = {"minus-sibling": "all41", "all41": "nosib"}     # the other variant's name, per primary choice
W_DEV_ARMS = tuple(W.DEV_ARMS) + ("x12_s",)
W_SEALED_ARMS = tuple(W.SEALED_ARMS)


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw), default=str), flush=True)


def sha_file(p):
    return W.file_sha256(p)


def agg_sha(files):
    """SHA-256 over the sorted 'path sha256' lines (as the closure hash of §13.4)."""
    return hashlib.sha256("\n".join(f"{k} {files[k]}" for k in sorted(files)).encode()).hexdigest()


def dev_units(v84=V84):
    """§12 development units: the 16 room development games with k* >= 4 (v84 room verdict), in the A.2 order
    default_rng([9801, 1]).permutation(sorted) (the rule of start_v85.unit_order(eligible_units()))."""
    v = json.load(open(pathlib.Path(v84) / "room_verdict_v84.json", encoding="utf-8"))
    games = sorted(r["game"] for r in v["rows"] if r["game"] in v["room_dev"] and r["k_star"] >= 4)
    return [str(g) for g in np.random.default_rng([9801, 1]).permutation(games)]


def code_identity():
    """(sha256, {path: sha256}, foreign): closure of this process (§13.4) plus the family files."""
    extra = [SCRIPT] + [ROOT / f for f in FAMILY_FILES if (ROOT / f).exists()]
    h, pairs, foreign = ck.closure_hash(ROOT, extra)
    return h, {a: b for a, b in pairs}, foreign


class Build:
    """One pool's build under an objects root R = <out>/objects (module docstring)."""

    def __init__(self, out, pool="dev", g1b=W.G1B, games=None, n_max=None, data=None, sealed_primary=PRIMARY[0],
                 kappa_from=None, kappa_curve=None):
        assert pool in POOLS, pool
        assert sealed_primary in PRIMARY, sealed_primary
        self.out, self.pool = pathlib.Path(out), pool
        self.R = self.out / "objects"
        self.g1b = pathlib.Path(g1b)
        self.test_games = None if games is None else tuple(games)
        assert games is None or set(games) <= set(W.GAMES), "sources are burned games only (§4.4, §12)"
        self.games = list(W.GAMES) if games is None else [g for g in W.GAMES if g in set(games)]
        self.n_max, self._data, self.sealed_primary = n_max, data, sealed_primary
        self.kappa_from, self.kappa_curve = kappa_from, kappa_curve
        self._heads = self._sums = None
        self.code, self.code_files, self.foreign = code_identity()
        self.options = json.loads(json.dumps(dict(            # as a marker stores them (tuples -> lists)
            pool=pool, g1b=str(self.g1b), games=self.test_games, n_max=n_max, data_injected=data is not None,
            sealed_primary=sealed_primary if pool == "sealed" else None, kappa_curve_injected=kappa_curve is not None)))
        self.test_build = bool(games is not None or n_max is not None or data is not None or kappa_curve is not None)

    # ------------------------------------------------------------ inputs (loaded once, on first need)
    @property
    def data(self):
        if self._data is None:
            t0 = time.time()
            self._data = W.load_data(self.games, self.g1b, self.n_max)
            log(step="load_w_data", games=len(self.games), wall_s=round(time.time() - t0, 1))
        return self._data

    def heads_sums(self):
        if self._heads is None:
            t0 = time.time()
            self._heads = W.source_heads(self.data, self.games)
            self._sums = W.game_sums(self.data, self.games)
            log(step="source_heads", heads=len(self._heads), wall_s=round(time.time() - t0, 1))
        return self._heads, self._sums

    def expected_units(self):
        if self.pool == "dev":
            return dev_units()
        return list(SEALED_UNITS)

    # ------------------------------------------------------------ markers
    def marker_path(self, unit):
        return self.R / "build" / self.pool / f"{unit}.json"

    def done(self, unit):
        p = self.marker_path(unit)
        if not p.exists():
            return False
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("code") != self.code or m.get("options") != self.options or m.get("unit") != unit:
            return False
        return all((self.R / f).exists() and sha_file(self.R / f) == h for f, h in m["files"].items())

    def _mark(self, unit, files, t0):
        ck.write_json(self.marker_path(unit), dict(format=FORMAT, unit=unit, pool=self.pool, code=self.code,
                                                  options=self.options, files=dict(sorted(files.items())),
                                                  wall_s=round(time.time() - t0, 2),
                                                  written=time.strftime("%Y-%m-%d %H:%M:%S")))

    # ------------------------------------------------------------ sources
    def w_sources(self, unit, variant=False):
        if self.pool == "dev":
            return [h for h in self.games if h != unit]
        drop = SIBLING.get(unit) if (self.sealed_primary == "minus-sibling") != variant else None
        return [h for h in self.games if h != unit and h != drop]

    def m_sources(self, unit, variant=False):
        src = sorted(set(self.w_sources(unit, variant)))
        if self.test_games is None:                        # the frozen builder's own rule, checked
            if self.pool == "dev" or (self.sealed_primary == "minus-sibling") != variant:
                assert src == sorted(OM.sources_of(unit)), (unit, "source rule")
        return src

    # ------------------------------------------------------------ one unit
    def _w_files(self, built):
        files = {}
        for side in built.values():
            stem = f"w/{side['arm']}/{W.obj_name(side['unit'], side['draw'])}"
            for ext in (".npz", ".json"):
                files[stem + ext] = sha_file(self.R / (stem + ext))
        return files

    def _m_write(self, objs, directory=None):
        d = self.R / "m" if directory is None else directory
        files = {}
        for o in objs:
            p, h = OM.write_object(d, o)
            files[p.relative_to(self.R).as_posix()] = h
        man_p = d / "manifest.json"
        man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
        man.update({pathlib.PurePosixPath(f).name: h for f, h in files.items()})
        ck.write_json(man_p, dict(sorted(man.items())))
        return files

    def build_unit(self, unit):
        t0 = time.time()
        heads, sums = self.heads_sums()
        data, g1b = self.data, self.g1b
        files = {}
        if self.pool == "dev":
            built = W.build_unit(self.R, unit, data, heads, sums, self.w_sources(unit), W_DEV_ARMS, W.dose_dev(unit))
            files.update(self._w_files(built))
            src = self.m_sources(unit)
            x = OM.build_x(unit, g1b, sources=src)
            objs = [x, OM.build_k_mine(unit, g1b), OM.build_p(unit, x, g1b)]
            objs += [OM.build_dose(unit, "X4", d, g1b, sources=src) for d in (1, 2)]
            objs += [OM.build_dose(unit, "X12s", 1, g1b, sources=src)]
            files.update(self._m_write(objs))
        else:
            built = W.build_unit(self.R, unit, data, heads, sums, self.w_sources(unit), W_SEALED_ARMS,
                                 W.dose_sealed(unit))
            files.update(self._w_files(built))
            src = self.m_sources(unit)
            x = OM.build_x(unit, g1b, sources=src)
            objs = [x, OM.build_p(unit, x, g1b), OM.build_dose(unit, "X4s", 1, g1b, sources=src),
                    OM.build_dose(unit, "X12s", 1, g1b, sources=src)]
            files.update(self._m_write(objs))
            if unit in SIBLING:                            # the other near-sibling variant of X (§9)
                var = VARIANT[self.sealed_primary]
                vb = W.build_unit(self.R, unit, data, heads, sums, self.w_sources(unit, True), ("x_s",), None,
                                  tag=f"~{var}")
                files.update(self._w_files(vb))
                xv = OM.build_x(unit, g1b, sources=self.m_sources(unit, True))
                files.update(self._m_write([xv], self.R / "m" / "variants" / var))
        files.update(self._m_write([OM.build_dm()]))
        W.write_manifest(self.R)
        self._mark(unit, files, t0)
        log(step="unit", pool=self.pool, unit=unit, files=len(files), wall_s=round(time.time() - t0, 1))
        return files

    # ------------------------------------------------------------ kappa
    def kappa_done(self):
        return self.done("_kappa")

    def kappa_path(self):
        return pathlib.Path(self.kappa_from or (V85 / "objects" / "w" / "kappa.json"))

    def f1_reference(self):
        """Sealed pool: the F1-committed family code and kappa SHA-256 (§4.4 "kappa ... committed at F1", §9 "the
        frozen code"), read beside the carried kappa.json (<dev out>/objects/w/kappa.json): the `hashes` of
        <dev out>/f1_thresholds.json when it exists, else the development index <dev out>/objects/index.json."""
        src = self.kappa_path()
        f1p, idxp = src.parent.parent.parent / "f1_thresholds.json", src.parent.parent / "index.json"
        if f1p.exists():
            path, h = f1p, json.loads(f1p.read_text(encoding="utf-8")).get("hashes") or {}
        elif idxp.exists():
            path, h = idxp, json.loads(idxp.read_text(encoding="utf-8"))
        else:
            return dict(path=None, family_code=None, kappa=None)
        return dict(path=str(path), family_code=(h.get("family_code") or {}).get("sha256"),
                    kappa=(h.get("kappa") or {}).get("sha256"))

    def kappa_source(self):
        """Sealed pool: the F1 kappa.json to carry, checked before any unit is built: complete, and byte-identical
        to the F1-committed kappa (review fix M-6; a test build may lack an F1 reference, never contradict it)."""
        src = self.kappa_path()
        if not src.exists():
            raise SystemExit(f"sealed pool: no F1 kappa.json at {src} (--kappa-from)")
        k = json.loads(src.read_text(encoding="utf-8"))
        if k.get("format") != W.FORMAT or any((k.get("kappa") or {}).get(a) is None for a in W.TUNED):
            raise SystemExit(f"{src}: not a complete family W kappa.json")
        ref = self.f1_reference()
        if ref["kappa"] is None and not self.test_build:
            raise SystemExit(f"{src}: no F1-committed kappa SHA-256 beside it (f1_thresholds.json hashes or the "
                             f"development index.json) to check it against")
        if ref["kappa"] is not None and sha_file(src) != ref["kappa"]:
            raise SystemExit(f"{src}: SHA-256 differs from the F1-committed kappa {ref['kappa']} ({ref['path']})")
        return src

    def build_kappa(self):
        t0 = time.time()
        kp = self.R / "w" / "kappa.json"
        if self.pool == "dev":
            heads, sums = self.heads_sums()
            games = [c for c in W.KAPPA_GAMES if c in self.data]
            res = W.kappa_rule(self.data, heads, sums, games, self.games, curve=self.kappa_curve)
            W.write_kappa(self.R, res)
        else:
            src = self.kappa_source()
            kp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, kp)
            assert sha_file(kp) == sha_file(src)
        W.write_manifest(self.R)
        self._mark("_kappa", {"w/kappa.json": sha_file(kp)}, t0)
        log(step="kappa", pool=self.pool, kappa=W.load_kappa(self.R)[0]["kappa"], wall_s=round(time.time() - t0, 1))

    # ------------------------------------------------------------ run and index
    def run(self, units=None, kappa=True):
        units = self.expected_units() if units is None else list(units)
        if self.pool == "dev":
            bad = [u for u in units if u not in W.GAMES]
            if bad:
                raise SystemExit(f"development units must be burned games (LOGO sources): {bad}")
        else:
            bad = [u for u in units if u not in SEALED_UNITS]
            if bad:
                raise SystemExit(f"sealed units are non-hard sealed or second-round games: {bad}")
            if kappa:
                self.kappa_source()
        for u in units:
            if self.done(u):
                log(step="unit", pool=self.pool, unit=u, skipped="done")
                continue
            self.build_unit(u)
        if kappa and not self.kappa_done():
            self.build_kappa()
        return self.index()

    def _markers(self):
        d = self.R / "build" / self.pool
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}

    def index(self):
        expected = self.expected_units()
        marks = self._markers()
        ok = {u: m for u, m in marks.items() if m.get("code") == self.code and m.get("options") == self.options}
        stale = sorted(u for u in marks if u not in ok)
        files, bad = {}, []
        for u, m in ok.items():
            for f, h in m["files"].items():
                p = self.R / f
                if not p.exists() or sha_file(p) != h:
                    bad.append(f)
                files[f] = h
        units_built = sorted(u for u in ok if u != "_kappa")
        missing = [u for u in expected if u not in ok]
        have = set()
        for sub in ("w", "m"):
            base = self.R / sub
            if base.exists():
                have |= {p.relative_to(self.R).as_posix() for p in base.rglob("*")
                         if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")}
        stray = sorted(have - set(files))
        kap = None
        ref = self.f1_reference() if self.pool == "sealed" else None
        if "_kappa" in ok and (self.R / "w" / "kappa.json").exists():
            k, ksha = W.load_kappa(self.R)
            kap = dict(sha256=ksha, kappa=k["kappa"], derived=k.get("derived"), extended=k.get("extended"),
                       kappas=k.get("kappas"), n_curves=k.get("n_curves"), skipped=k.get("skipped"),
                       copied_from=str(self.kappa_path()) if self.pool == "sealed" else None)
            if self.pool == "sealed":                      # §4.4 "kappa ... committed at F1" (review fix M-6)
                kap["f1_sha256"] = ref["kappa"]
                kap["same_kappa_as_f1"] = ksha == ref["kappa"] if ref["kappa"] else None
        now = code_identity()[0]                           # a module loaded during the build would escape the identity
        complete = bool(not missing and kap is not None and not bad and not self.test_build and now == self.code
                        and all(v is not None for v in kap["kappa"].values()))
        mman = self.R / "m" / "manifest.json"
        objects = dict(sha256=agg_sha(files), files=dict(sorted(files.items())), n_files=len(files),
                       w=W.hashes(self.R)["development_objects"] if (self.R / "w").exists() else None,
                       m_manifest=sha_file(mman) if mman.exists() else None, units=units_built)
        fam = dict(sha256=self.code, files=self.code_files, family_files=list(FAMILY_FILES),
                   arm_modules=list(FAMILY_ARMS), foreign=self.foreign)
        okey = "development_objects" if self.pool == "dev" else "sealed_objects"
        if self.pool == "sealed":                          # §9 "the frozen code": the F1-committed code and kappa
            f1_code = ref["family_code"]
            fam["f1_reference"] = ref["path"]
            fam["f1_family_code"], fam["same_code_as_f1"] = f1_code, f1_code == self.code if f1_code else None
            complete = bool(complete and fam["same_code_as_f1"] is True and kap["same_kappa_as_f1"] is True)
        build = dict(pool=self.pool, complete=complete, expected_units=expected, built_units=units_built,
                     missing_units=missing, extra_units=[u for u in units_built if u not in expected],
                     stale_markers=stale, hash_mismatch=sorted(bad), stray_files=stray, options=self.options,
                     closure_unchanged=now == self.code,
                     test_build=self.test_build, objects_root=str(self.R),
                     peak_commit_mb=round(ck.peak_mem_mb()[1], 1),
                     note=None if complete else "incomplete or test build: the F1 keys are null (f1 refuses)")
        idx = dict(rule=RULE, format=FORMAT, objects_build=build,
                   family_code=fam if complete else None, kappa=kap if complete else None,
                   provenance=dict(stamp((), cwd=str(ROOT), paths=PATHS), closure_hash=self.code))
        idx[okey] = objects if complete else None
        if not complete:
            build["partial"] = dict(family_code=fam, objects=objects, kappa=kap)
        path = self.R / ("index.json" if self.pool == "dev" else "index_sealed.json")
        ck.write_json(path, idx)
        log(step="index", pool=self.pool, complete=complete, path=str(path), built=len(units_built),
            missing=missing, stray=len(stray))
        return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description="v85 family objects (Part B item 13)")
    ap.add_argument("--pool", choices=POOLS, default="dev")
    ap.add_argument("--units", nargs="+", default=None, help="restrict the build to these units (resumable)")
    ap.add_argument("--out", default=None, help="output root (objects under <out>/objects); dev default "
                                                "craft_v6_out/v85; required for --pool sealed")
    ap.add_argument("--skip-kappa", action="store_true", help="dev: postpone the kappa rule (index incomplete)")
    ap.add_argument("--kappa-from", default=None, help="sealed: the F1 kappa.json to carry")
    ap.add_argument("--sealed-primary", choices=PRIMARY, default=PRIMARY[0])
    ap.add_argument("--index-only", action="store_true", help="write the index from the existing markers")
    a = ap.parse_args(argv)
    if a.pool == "sealed":
        if a.out is None:
            ap.error("--pool sealed needs --out (the sealed root, never the development root)")
        if pathlib.Path(a.out).resolve() == V85.resolve():
            ap.error("--pool sealed: --out must not be the development root")
    out = pathlib.Path(a.out) if a.out else V85
    b = Build(out, a.pool, sealed_primary=a.sealed_primary, kappa_from=a.kappa_from)
    t0 = time.time()
    idx = b.index() if a.index_only else b.run(a.units, kappa=not a.skip_kappa)
    ob = idx["objects_build"]
    print(json.dumps(dict(pool=a.pool, complete=ob["complete"], built=len(ob["built_units"]),
                          missing=ob["missing_units"], wall_s=round(time.time() - t0, 1),
                          peak_commit_mb=ob["peak_commit_mb"])))


if __name__ == "__main__":
    main()
