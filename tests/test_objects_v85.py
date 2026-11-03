"""Part B item 13, scripts/v85_objects.py: the development and sealed family objects, kappa and the F1 index, built
on 2 units into a temp dir (never the full development build). Source pool restricted to POOL (15 small burned
games; family W reads their G1b streams truncated to N_MAX expansions, family M their full row logs), through the
driver's test-only Python knobs; the builders use their own §4.4/§12 seeds; no run here.

1. The pools: the 16 development units in the A.2 order (start_v85's rule), the sealed list (17 non-hard + 9
   second-round), every family arm module named in FAM (v85_verdict, canon) and covered by family_code.
2. Development build on alien and zaxxon: every object of both families (plus X12_S's and X12s's closure
   objects), hash-checked by the families' loaders; byte-identical to the families' own build_development;
   kappa by the rule on the calibration games of the pool; no stray file; an incomplete or test build writes
   null F1 keys (f1 refuses), a complete one the keys f1 merges (family_code, development_objects, kappa).
3. Resume per unit: everything done is skipped; a kill mid-unit and a tampered file rebuild only their unit, to
   the same bytes; a marker of another code identity is stale.
4. Sealed build on frogger (near-sibling freeway in the pool) and defender: no K object, one dose draw each per
   unit ([9804, 2, i_unit], [9804, 3, i_unit]; review fix W-1/M-5), the near-sibling variants, the F1 kappa.json
   carried byte for byte and checked against the F1-committed kappa SHA-256; the sealed index is complete only
   with F1's code and kappa (review fix M-6); refusals.
5. Every family arm builds its hooks on the driver's objects in production mode at seed 9826 (v85_calib's
   closure build; X12_S and X12_on closure-only, review fix W-3/M-7), and refuses a missing object or a seed
   with no dose draw.
6. The command line (argument refusals; --index-only under the frozen options marks test markers stale).
"""
import json
import shutil

import numpy as np
import pytest

from ragnarok.agent_v85 import macros as M
from ragnarok.evaluation import arms_v85
from ragnarok.evaluation import objects_m_v85 as OM
from ragnarok.evaluation import objects_w_v85 as W
from ragnarok.evaluation import senses_v85  # noqa: F401  (hooks_w's lazy import, loaded before any Build)
from ragnarok.evaluation import start_v85 as ST
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, HARD_EXPLORATION, SEALED, SECOND
from scripts import v85_calib as C
from scripts import v85_objects as OB
from scripts import v85_verdict as V

for _a in OB.FAMILY_ARMS:                      # every arm module loaded up front: the code identity is the closure
    arms_v85.load(_a)

UNITS = ("alien", "zaxxon")
POOL = ("alien", "zaxxon", "breakout", "pong", "bowling", "jamesbond", "time_pilot", "hero", "asterix", "bank_heist",
        "video_pinball", "gopher", "carnival", "phoenix", "freeway")
N_MAX = 1000
HAVE = all((ST.V84 / "g1b" / f"{g}_{r}.npz").exists() for g in POOL for r in (9712, 9713))
needs_v84 = pytest.mark.skipif(not HAVE, reason="v84 G1b logs absent")
W_UNIT = ("f_s", "x_s", "k_s", "p_s", "u_s", "t", "s")


class Kill(Exception):
    pass


def dev_files(u):
    w = [f"w/{a}/{u}" for a in W_UNIT] + [f"w/x4_s/{u}_d1", f"w/x4_s/{u}_d2", f"w/x12_s/{u}_d1"]
    m = [f"m/x_{u}.json", f"m/kmine_{u}.json", f"m/p_{u}.json", f"m/x4_{u}_d1.json", f"m/x4_{u}_d2.json",
         f"m/x12s_{u}.json"]
    return {s + e for s in w for e in (".npz", ".json")} | set(m)


def unit_files(idx, units):
    files = (idx.get("development_objects") or idx["objects_build"]["partial"]["objects"])["files"]
    return {f: h for f, h in files.items() if any(u in f for u in units)}


def comparable(R, files):
    """Per object file, what a rebuild must reproduce: M JSON and kappa.json by file SHA-256, W npz by content
    hash (np.savez's zip timestamps vary, objects_w_v85.content_sha256) and W sidecars without their npz hash."""
    out = {}
    for f in sorted(files):
        p = R / f
        if f.startswith("w/") and f.endswith(".npz"):
            with np.load(p) as z:
                out[f] = W.content_sha256({k: z[k] for k in z.files})
        elif f.startswith("w/") and f != "w/kappa.json":
            side = json.loads(p.read_text(encoding="utf-8"))
            side.pop("file_sha256")
            out[f] = side
        else:
            out[f] = W.file_sha256(p)
    return out


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    if not HAVE:
        pytest.skip("v84 G1b logs absent")
    out = tmp_path_factory.mktemp("objdev")
    b = OB.Build(out, "dev", games=POOL, n_max=N_MAX)
    idx = b.run(UNITS)
    return dict(out=out, R=out / "objects", b=b, idx=idx)


# ---------------------------------------------------------------- 1. pools and family lists
def test_pools_and_family_lists():
    assert OB.dev_units() == ST.unit_order(ST.eligible_units()) and len(OB.dev_units()) == 16
    assert len(OB.SEALED_UNITS) == 26 and not set(OB.SEALED_UNITS) & set(HARD_EXPLORATION)
    assert set(OB.SEALED_UNITS) == (set(SEALED) - set(HARD_EXPLORATION)) | set(SECOND)
    assert not set(OB.SEALED_UNITS) & set(DEV + CALIBRATION) and set(W.GAMES) == set(DEV + CALIBRATION)
    names = set()
    for fam in V.FAM.values():
        for k, x in fam.items():
            if k in ("K", "F", "X", "frozen", "X4", "X12") and x:
                names.add(V.canon(x))
            elif k in ("tuned", "placebos"):
                names |= {V.canon(y) for y in x}
    assert {V.canon(a) for a in OB.FAMILY_ARMS} == names
    extra = set(C.arm_modules()) - set(C.GATE_ARMS) - set(OB.FAMILY_ARMS)        # family_code covers them all
    assert extra == {"o1"}                                                         # G-build test 6's O_1 probe
    assert all((OB.ROOT / f).exists() for f in OB.FAMILY_FILES)
    assert OB.VARIANT == {"minus-sibling": "all41", "all41": "nosib"} and OB.SIBLING["frogger"] == "freeway"


# ---------------------------------------------------------------- 2. development build
@needs_v84
def test_dev_build_layout_and_loaders(dev):
    R, idx = dev["R"], dev["idx"]
    ob = idx["objects_build"]
    want = set().union(*(dev_files(u) for u in UNITS)) | {"m/dm.json", "w/kappa.json"}
    part = ob["partial"]
    assert set(part["objects"]["files"]) == want and ob["stray_files"] == [] and ob["hash_mismatch"] == []
    assert ob["built_units"] == sorted(UNITS) and ob["missing_units"] == [u for u in OB.dev_units() if u not in UNITS]
    assert not ob["complete"] and ob["test_build"] and ob["closure_unchanged"]
    assert idx["family_code"] is None and idx["development_objects"] is None and idx["kappa"] is None
    assert json.loads((R / "index.json").read_text(encoding="utf-8")) == json.loads(json.dumps(idx))
    for f, h in part["objects"]["files"].items():
        assert W.file_sha256(R / f) == h, f
    pool_w = [g for g in W.GAMES if g in POOL]
    for u in UNITS:
        for a in W_UNIT:
            arr, side = W.load_object(R, a, u)                                      # content hash checked
            assert side["sources"] == [g for g in pool_w if g != u] and side["truncated"], (u, a)
        for k in (1, 2):
            assert W.load_object(R, "x4_s", u, k)[1]["seeds"] == [9804, 1, W.GI[u]]
        s12 = W.load_object(R, "x12_s", u, 1)[1]
        assert s12["seeds"] == [9804, 3, W.UI[u]] and len(set(s12["draw_sources"])) == 12 and u not in s12["draw_sources"]
        for name in ("x", "kmine", "p", "x4", "x12s"):
            fn = f"{name}_{u}.json" if name not in ("x4",) else f"x4_{u}_d1.json"
            o = OM.load_object(R / "m" / fn)                                        # manifest hash checked
            assert o["unit"] == u
        assert OM.load_object(R / "m" / f"x_{u}.json")["sources"] == sorted(g for g in POOL if g != u)
        assert OM.load_object(R / "m" / f"kmine_{u}.json")["sources"] == [u]
        assert len(OM.load_object(R / "m" / f"x12s_{u}.json")["sources"]) == 12
    assert OM.load_object(R / "m" / "dm.json")["macros"] == [[[int(a), int(n)] for a, n in m] for m in M.DM]
    k = W.load_kappa(R)[0]
    assert k["games"] == ["pong", "breakout"] and k["pool_size"] == len(POOL) and set(k["kappa"]) == set(W.TUNED)
    assert all(v is not None for v in k["kappa"].values()) and part["kappa"]["kappa"] == k["kappa"]
    man = json.loads((R / "m" / "manifest.json").read_text(encoding="utf-8"))
    assert {f"m/{n}" for n in man} == {f for f in want if f.startswith("m/")}


@needs_v84
def test_dev_objects_equal_the_family_builders(dev, tmp_path, monkeypatch):
    """The driver adds nothing to the frozen builders: W content hashes equal build_development's, M bytes equal
    objects_m_v85.build_development's (sources restricted to the same pool)."""
    R, b = dev["R"], dev["b"]
    out = W.build_development(tmp_path / "w", list(UNITS), data=b.data, games=b.games, kappa=False)
    for u in UNITS:
        for name, side in out["built"][u].items():
            arm, draw = side["arm"], side["draw"]
            assert W.load_object(R, arm, u, draw)[1]["content_sha256"] == side["content_sha256"], (u, name)
    monkeypatch.setattr(OM, "sources_of", lambda unit: sorted(g for g in POOL if g != unit))
    man = OM.build_development(list(UNITS), tmp_path / "m")
    files = dev["idx"]["objects_build"]["partial"]["objects"]["files"]
    assert {f"m/{n}" for n in man} == {f for f in files if f.startswith("m/")}
    assert all(files[f"m/{n}"] == h for n, h in man.items())


@needs_v84
def test_complete_index_has_the_f1_keys(dev, monkeypatch):
    R = dev["R"]
    monkeypatch.setattr(OB, "dev_units", lambda v84=None: list(UNITS))
    b = OB.Build(dev["out"], "dev", games=POOL, n_max=N_MAX)
    b.test_build = False                             # what a full build on its pool writes
    idx = b.index()
    ob = idx["objects_build"]
    assert ob["complete"] and ob["missing_units"] == [] and "partial" not in ob
    do, fc, kp = idx["development_objects"], idx["family_code"], idx["kappa"]
    assert do["files"] == dev["idx"]["objects_build"]["partial"]["objects"]["files"]
    assert do["sha256"] == OB.agg_sha(do["files"]) and do["w"] == W.hashes(R)["development_objects"]
    assert do["m_manifest"] == W.file_sha256(R / "m" / "manifest.json") and do["units"] == sorted(UNITS)
    assert kp["sha256"] == W.file_sha256(R / "w" / "kappa.json") == W.load_kappa(R)[1]
    assert fc["sha256"] == b.code == dev["b"].code and "scripts/v85_objects.py" in fc["files"]
    assert set(OB.FAMILY_FILES) <= set(fc["files"]) and fc["foreign"] == []
    # v85_calib f1 merges the index's keys (all but provenance and rule) and needs HASH_KEYS
    merged = {k: v for k, v in json.loads((R / "index.json").read_text(encoding="utf-8")).items()
              if k not in ("provenance", "rule")}
    assert [k for k in C.HASH_KEYS if k not in ("maps", "libraries") and merged.get(k) is None] == []
    assert "starts" not in merged and "maps" not in merged and "libraries" not in merged
    partial = {k: v for k, v in dev["idx"].items() if k not in ("provenance", "rule")}
    assert [k for k in C.HASH_KEYS if k not in ("maps", "libraries") and partial.get(k) is None] == \
        ["family_code", "development_objects", "kappa"]


# ---------------------------------------------------------------- 3. resume
@needs_v84
def test_resume_per_unit_after_a_kill_and_a_tamper(dev, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nothing may be rebuilt")
    b = OB.Build(dev["out"], "dev", games=POOL, n_max=N_MAX)
    with monkeypatch.context() as mp:
        mp.setattr(OB.Build, "build_unit", boom)
        mp.setattr(OB.Build, "build_kappa", boom)
        assert b.run(UNITS)["objects_build"]["built_units"] == sorted(UNITS)
    names = unit_files(dev["idx"], UNITS)
    ref = comparable(dev["R"], names)
    out = tmp_path / "k"
    real_p, calls = OM.build_p, []

    def build_p(unit, *a, **k):
        if unit == "zaxxon":
            raise Kill
        return real_p(unit, *a, **k)
    with monkeypatch.context() as mp:
        mp.setattr(OM, "build_p", build_p)
        with pytest.raises(Kill):
            OB.Build(out, "dev", games=POOL, n_max=N_MAX).run(UNITS, kappa=False)
    R = out / "objects"
    assert (R / "build" / "dev" / "alien.json").exists() and not (R / "build" / "dev" / "zaxxon.json").exists()
    real_unit = OB.Build.build_unit

    def counted(self, unit):
        calls.append(unit)
        return real_unit(self, unit)
    monkeypatch.setattr(OB.Build, "build_unit", counted)
    idx = OB.Build(out, "dev", games=POOL, n_max=N_MAX).run(UNITS, kappa=False)
    assert calls == ["zaxxon"] and set(unit_files(idx, UNITS)) == set(names)
    assert comparable(R, names) == ref                                               # as the uninterrupted build
    p = R / "m" / "x_zaxxon.json"
    o = json.loads(p.read_text(encoding="utf-8"))
    o["report"]["tampered"] = True
    p.write_text(json.dumps(o), encoding="utf-8")
    b2 = OB.Build(out, "dev", games=POOL, n_max=N_MAX)
    assert b2.done("alien") and not b2.done("zaxxon")
    idx = b2.run(UNITS, kappa=False)
    assert calls == ["zaxxon", "zaxxon"] and comparable(R, names) == ref
    b3 = OB.Build(out, "dev", games=POOL, n_max=N_MAX)
    b3.code = "0" * 64                                                               # other code: stale markers
    assert not b3.done("alien")
    ob = b3.index()["objects_build"]
    assert ob["stale_markers"] == ["alien", "zaxxon"] and ob["built_units"] == [] and not ob["complete"]


# ---------------------------------------------------------------- 4. sealed
@needs_v84
def test_sealed_pool_two_units(dev, tmp_path):
    f1 = tmp_path / "f1" / "objects"                       # an F1 layout: kappa.json and the index's family code
    (f1 / "w").mkdir(parents=True)
    kf = f1 / "w" / "kappa.json"
    shutil.copyfile(dev["R"] / "w" / "kappa.json", kf)
    f1_index = dict(family_code=dict(sha256=dev["b"].code), kappa=dict(sha256=W.file_sha256(kf)))
    (f1 / "index.json").write_text(json.dumps(f1_index), encoding="utf-8")
    s = OB.Build(tmp_path / "s", "sealed", games=POOL, n_max=N_MAX, kappa_from=kf)
    idx = s.run(["frogger", "defender"])
    R, ob = tmp_path / "s" / "objects", idx["objects_build"]
    want = {"m/dm.json", "w/kappa.json", "w/x_s/frogger~all41.npz", "w/x_s/frogger~all41.json",
            "m/variants/all41/x_frogger.json"}
    for u in ("frogger", "defender"):
        w = [f"w/{a}/{u}" for a in ("f_s", "x_s", "p_s", "u_s", "t", "s")] + [f"w/x4_s/{u}_d1", f"w/x12_s/{u}_d1"]
        want |= {x + e for x in w for e in (".npz", ".json")}
        want |= {f"m/x_{u}.json", f"m/p_{u}.json", f"m/x4s_{u}.json", f"m/x12s_{u}.json"}
    assert set(ob["partial"]["objects"]["files"]) == want and ob["stray_files"] == [] and ob["pool"] == "sealed"
    assert "sealed_objects" in idx and "development_objects" not in idx and (R / "index_sealed.json").exists()
    assert ob["missing_units"] == [u for u in OB.SEALED_UNITS if u not in ("frogger", "defender")]
    pool_w = [g for g in W.GAMES if g in POOL]
    assert W.load_object(R, "x_s", "frogger")[1]["sources"] == [g for g in pool_w if g != "freeway"]
    assert W.load_object(R, "x_s", "frogger~all41")[1]["sources"] == pool_w
    assert W.load_object(R, "x_s", "defender")[1]["sources"] == pool_w
    s4, s12 = W.load_object(R, "x4_s", "frogger", 1)[1], W.load_object(R, "x12_s", "frogger", 1)[1]
    assert s4["seeds"] == [9804, 2, W.UI["frogger"]] and len(s4["draw_sources"]) == 4 and "freeway" not in s4["draw_sources"]
    assert s12["seeds"] == [9804, 3, W.UI["frogger"]] and len(set(s12["draw_sources"])) == 12 \
        and "freeway" not in s12["sources"]
    assert W.load_object(R, "x4_s", "defender", 1)[1]["seeds"] == [9804, 2, W.UI["defender"]]   # per unit (W-1/M-5)
    assert W.load_object(R, "x12_s", "defender", 1)[1]["seeds"] == [9804, 3, W.UI["defender"]]
    assert OM.load_object(R / "m" / "x_frogger.json")["sources"] == sorted(g for g in POOL if g != "freeway")
    assert OM.load_object(R / "m" / "variants" / "all41" / "x_frogger.json")["sources"] == sorted(POOL)
    m4, m12 = OM.load_object(R / "m" / "x4s_frogger.json"), OM.load_object(R / "m" / "x12s_frogger.json")
    assert m4["seed"] == [9804, 2] and len(m4["sources"]) == 4 and m12["seed"] == [9804, 3] and len(m12["sources"]) == 12
    assert (R / "w" / "kappa.json").read_bytes() == kf.read_bytes()
    assert ob["partial"]["kappa"]["sha256"] == W.file_sha256(kf)
    fc = ob["partial"]["family_code"]                                                # §9 "the same frozen code"
    assert fc["f1_family_code"] == dev["b"].code == s.code and fc["same_code_as_f1"] is True
    assert fc["f1_reference"] == str(f1 / "index.json")
    assert ob["partial"]["kappa"]["same_kappa_as_f1"] is True and ob["partial"]["kappa"]["f1_sha256"] == W.file_sha256(kf)
    # review fix M-6: the sealed index is complete only with F1's code and kappa (a full pool, no test knobs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(OB, "SEALED_UNITS", ("frogger", "defender"))
        s.test_build = False
        assert s.index()["objects_build"]["complete"]
        for bad in (dict(f1_index, family_code=dict(sha256="0" * 64)), dict(f1_index, kappa=dict(sha256="0" * 64))):
            (f1 / "index.json").write_text(json.dumps(bad), encoding="utf-8")
            idx2 = s.index()
            assert not idx2["objects_build"]["complete"] and idx2["sealed_objects"] is None, bad
        (f1 / "index.json").write_text(json.dumps(f1_index), encoding="utf-8")
        s.test_build = True
    s.index()
    # review fix M-6: a carried kappa.json must be the F1-committed one
    kb = tmp_path / "f1b" / "objects" / "w" / "kappa.json"
    kb.parent.mkdir(parents=True)
    kj = json.loads(kf.read_text(encoding="utf-8"))
    kj["kappa"]["x_s"] = kj["kappa"]["x_s"] * 4
    kb.write_text(json.dumps(kj), encoding="utf-8")
    (kb.parent.parent / "index.json").write_text(json.dumps(f1_index), encoding="utf-8")   # F1 committed kf's bytes
    with pytest.raises(SystemExit, match="differs from the F1-committed kappa"):
        OB.Build(tmp_path / "s4", "sealed", games=POOL, n_max=N_MAX, kappa_from=kb).run(["defender"])
    assert not (tmp_path / "s4" / "objects" / "w").exists()                          # refused before any build
    f1t = tmp_path / "f1" / "f1_thresholds.json"                                      # f1_thresholds.json wins
    f1t.write_text(json.dumps(dict(hashes=dict(kappa=dict(sha256="0" * 64)))), encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the F1-committed kappa"):
        OB.Build(tmp_path / "s5", "sealed", games=POOL, n_max=N_MAX, kappa_from=kf).kappa_source()
    f1t.unlink()
    kc = tmp_path / "f1c" / "objects" / "w" / "kappa.json"                           # no F1 reference beside it
    kc.parent.mkdir(parents=True)
    shutil.copyfile(kf, kc)
    nb = OB.Build(tmp_path / "s6", "sealed", games=POOL, n_max=N_MAX, kappa_from=kc)
    assert nb.kappa_source() == kc                                                    # a test build may lack one
    nb.test_build = False
    with pytest.raises(SystemExit, match="no F1-committed kappa"):
        nb.kappa_source()
    with pytest.raises(SystemExit):
        s.run(["alien"])                                                             # a burned game
    with pytest.raises(SystemExit):
        s.run(["pitfall"])                                                           # hard games stay unopened
    with pytest.raises(SystemExit):
        OB.Build(tmp_path / "s2", "sealed", games=POOL, n_max=N_MAX, kappa_from=tmp_path / "none.json").run(["defender"])
    assert not (tmp_path / "s2" / "objects" / "w").exists()                          # refused before any build
    a = OB.Build(tmp_path / "s3", "sealed", games=POOL, n_max=N_MAX, kappa_from=kf, sealed_primary="all41")
    a.run(["frogger"])
    R3 = tmp_path / "s3" / "objects"
    assert W.load_object(R3, "x_s", "frogger")[1]["sources"] == pool_w
    assert W.load_object(R3, "x_s", "frogger~nosib")[1]["sources"] == [g for g in pool_w if g != "freeway"]
    assert OM.load_object(R3 / "m" / "x_frogger.json")["sources"] == sorted(POOL)
    assert OM.load_object(R3 / "m" / "variants" / "nosib" / "x_frogger.json")["sources"] == \
        sorted(g for g in POOL if g != "freeway")


# ---------------------------------------------------------------- 5. arms on the driver's objects
def ctx(out, game="alien", seed=9826, smoke=False):
    return arms_v85.Ctx(game=game, start="A", seed=seed, E=9819, s1=None, other_s1=None, body=None, ladder=[],
                        rungs=[], k_first=0, k_ref=0, cap=4_000_000, units=[game], paths=dict(v85=str(out)),
                        starts={}, load_s1=None, load_body=None, smoke=smoke)


@needs_v84
def test_every_family_arm_builds_on_the_driver_objects(dev):
    R = dev["R"]
    for arm in OB.FAMILY_ARMS:
        mod, hooks = arms_v85.build(arm, ctx(dev["out"]))                            # production mode, seed 9826
        if "w_object" in hooks:
            draw = hooks["w_object"].get("draw")
            assert hooks["w_object"]["content_sha256"] == W.load_object(R, arm, "alien", draw)[1]["content_sha256"]
        if hooks.get("object") and hooks["object"]["kind"]:
            assert hooks["object"]["library_sha256"]
    assert arms_v85.build("x12_s", ctx(dev["out"]))[1]["w_object"]["draw"] == 1
    for arm, key in (("x12_s", "w_object"), ("x12_on", "object")):                  # W-3 / M-7: closure build only
        h = arms_v85.build(arm, ctx(dev["out"]))[1]
        assert h[key]["closure_only"] and h["explorer"] is not None, arm
    assert not arms_v85.build("x4_s", ctx(dev["out"]))[1]["w_object"]["closure_only"]
    for arm, seed in (("x12_s", 9827), ("x12_on", 9827), ("x4_s", 9853), ("x4_on", 9853)):
        with pytest.raises(ValueError):
            arms_v85.build(arm, ctx(dev["out"], seed=seed))                          # not a closure or dose seed
    for arm in ("x_s", "s", "x_on", "k_mine"):
        with pytest.raises(FileNotFoundError):
            arms_v85.build(arm, ctx(dev["out"], game="qbert"))                       # no object: no run


# ---------------------------------------------------------------- 6. command line
@needs_v84
def test_command_line(dev, tmp_path, capsys):
    with pytest.raises(SystemExit):
        OB.main(["--pool", "sealed"])                                                # the sealed root is explicit
    with pytest.raises(SystemExit):
        OB.main(["--pool", "sealed", "--out", str(OB.V85)])                          # never the development root
    cp = tmp_path / "copy"
    shutil.copytree(dev["out"], cp)
    capsys.readouterr()
    OB.main(["--index-only", "--out", str(cp)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["pool"] == "dev" and not res["complete"] and res["built"] == 0
    ob = json.loads((cp / "objects" / "index.json").read_text(encoding="utf-8"))["objects_build"]
    assert ob["stale_markers"] == ["_kappa", "alien", "zaxxon"]                      # built with test knobs
    assert ob["options"]["games"] is None and ob["options"]["n_max"] is None
