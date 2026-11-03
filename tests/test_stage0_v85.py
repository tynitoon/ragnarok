"""v85_stage0 (Part B item 8, A.10.1 (a)-(f)): pure helpers on synthetic data, then S0 on two real units
(alien, zaxxon: k* = 4) into a temp dir, read back through v85_job's Starts. G-build test 3: start identity on
2 units, both starts, from the cached files. Seed 9800 only (§12): entry and emulator seeds 9800 at both starts,
spike-in simulations forced to [9800, b], few simulations.
"""
import functools
import json
import pathlib

import numpy as np
import pytest

from ragnarok.evaluation import start_v85 as s0lib
from ragnarok.evaluation.oracle_v85 import d_script
from scripts import v85_calib as C
from scripts import v85_job as J
from scripts import v85_stage0 as S0

SEED = 9800
TEST_SEEDS = {"A": SEED, "B": SEED}
UNITS = ("alien", "zaxxon")
HAVE_V84 = all((s0lib.V84 / "g1b" / f"{g}_{s}.npz").exists() for g in UNITS for s in (9712, 9713))
JSONS = ("starts.json", "rungs.json", "natural_spikeins.json", "floor.json", "dice_g4.json", "stalls.json",
         "seed_grep.json", "s0.json")


def test_physics_floor():
    assert S0.physics_floor(1) == 4 + 60
    assert S0.physics_floor(400) == 1600 + 60
    assert S0.physics_floor(401) == 1604 + 120


def test_g4_segment():
    end = [10, 20, 30, 40, 50]                                   # exp_meta billed at each expansion's end
    # records [billed, cum]: root (exp 0), M1 = 10 in exp 1, two rises in exp 2, M_ref = 30 in exp 4
    recs = [[5, 0], [12, 10], [25, 15], [27, 18], [45, 30]]
    s = S0.g4_segment(recs, end, 10, 30)
    assert (s["e_M1"], s["e_ref"], s["n"], s["raised"]) == (1, 4, 3, 2)
    p = 2 / 3
    assert s["p"] == pytest.approx(p) and s["G4"] == pytest.approx((1 - (1 - p) ** 4) / p)
    # a record after the last row belongs to the expansion cut by the cap
    s = S0.g4_segment(recs + [[57, 40]], end, 10, 40)
    assert (s["e_ref"], s["n"], s["raised"]) == (5, 4, 3)
    # M1 and M_ref in the same expansion: no expansion in the segment, no p
    s = S0.g4_segment([[5, 0], [12, 10], [14, 30]], end, 10, 30)
    assert s["n"] == 0 and s["p"] is None and s["G4"] is None
    assert S0.g4_segment([[5, 0], [12, 10], [21, 30]], end, 10, 30)["G4"] == pytest.approx(1.0)


def test_plateaus():
    recs = [[100, 0], [200, 5], [1_300_000, 9], [1_400_000, 12]]
    assert S0.plateaus(recs, 4_000_000) == [(1, 200, 1_300_000, False), (3, 1_400_000, 4_000_000, True)]
    assert S0.plateaus(recs, 2_000_000) == [(1, 200, 1_300_000, False)]


def test_choose_U():
    P = {10: 0.7, 12: 0.85, 16: 0.95}
    U, ok, table = S0.choose_U([f"u{i}" for i in range(16)], lambda us: P[len(us)])
    assert (U, ok) == (12, True) and [t["U"] for t in table] == [10, 12, 16]
    # capped at the kept units; nothing qualifies: the largest, flagged
    U, ok, table = S0.choose_U([f"u{i}" for i in range(11)], lambda us: 0.5)
    assert (U, ok) == (11, False) and [t["U"] for t in table] == [10, 11]
    assert S0.choose_U([], lambda us: 1.0) == (None, False, [])


def test_natural_cells():
    units = {"g": dict(K_g=[3, 4], natural={"A": {"3": 0, "4": 90_000}, "B": {"3": 60_000, "4": 70_000}},
                       A=dict(valid=True), B=dict(valid=False))}
    cells = S0.natural_cells(units, ["g"])
    assert list(cells) == [("g", "A")]
    assert cells[("g", "A")].tolist() == [[5000.0, 90000.0], [60000.0, 70000.0]]


def test_seed_grep(tmp_path):
    r = tmp_path / "wt"
    files = {"ragnarok/agent_v85/a.py": "SEED = 9800\n", "scripts/old.py": "x = 1\nseed = 9850  # reused\n",
             "hashes.py": "h = 'ab9800cd'\n", "decimals.py": "q = 0.9850\n", "long.py": "n = 198000\n",
             ".venv-x/lib/v.py": "s = 9801\n", "env2/lib/w.py": "s = 9801\n", "under.py": "f'{g}_9899'\n"}
    for rel, txt in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    (r / "env2" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    g = S0.seed_grep([r], own=r)
    w = g["worktrees"][0]
    assert w["owned"] == ["ragnarok/agent_v85/a.py"]
    assert sorted(f["path"] for f in w["foreign"]) == ["scripts/old.py", "under.py"]
    assert next(f for f in w["foreign"] if f["path"] == "scripts/old.py")["lines"] == [[2, "seed = 9850  # reused"]]
    assert w["files_scanned"] == 6 and g["foreign_files"] == 2 and not g["PASS"]
    assert S0.seed_grep([r], own=tmp_path / "other")["foreign_files"] == 3      # v85 files elsewhere are foreign


@pytest.fixture(scope="module")
def s0(tmp_path_factory):
    if not HAVE_V84:
        pytest.skip("v84 G1b logs absent")
    out = tmp_path_factory.mktemp("s0")
    grep_root = tmp_path_factory.mktemp("grep")
    (grep_root / "x.py").write_text("y = 2\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "simulate", functools.partial(C.simulate, seed=SEED))
        summ = S0.run(out, units=list(UNITS), sims=8, B=40, seeds=TEST_SEEDS, grep_roots=[grep_root],
                      log=lambda s: None)
        first = {k: _load(out, k)[x] for k, x in (("starts.json", "units"), ("natural_spikeins.json", "table"))}
        again = S0.run(out, units=list(UNITS), sims=8, B=40, seeds=TEST_SEEDS, grep_roots=[grep_root],
                       log=lambda s: None)
    return out, summ, dict(again, first=first)


def _load(out, name):
    return json.load(open(out / name, encoding="utf-8"))


def test_real_units_starts_and_rungs(s0):
    out, summ, _ = s0
    st = _load(out, "starts.json")
    draw = s0lib.unit_order(s0lib.eligible_units())
    assert st["draw_order_all"] == draw and st["order"] == [g for g in draw if g in UNITS]
    assert st["kept"] == st["order"] and not st["complete"] and st["test_seeds"]
    for g in UNITS:
        u = st["units"][g]
        assert u["K_g"] == [3, 4] and (u["k_first"], u["k_ref"]) == (3, 4) and u["kept"]     # §3: alien, zaxxon {3, 4}
        for s in "AB":
            assert u[s]["valid"] and u[s]["E"] == SEED and u[s]["context"] in ("after_step", "post_reset")
    assert summ["KC_start"] is None and summ["U"] is None and summ["U_partial"] == 2 and summ["KC_floor"] is None
    r = _load(out, "rungs.json")
    assert r["units"]["alien"]["rule"]["2"]["counted"] is False and not r["units"]["alien"]["rule"]["2"]["gap_ok"]
    for name in JSONS:
        o = _load(out, name)
        p = o["provenance"]
        assert o["claim_header"] and o["senses"] and p["code_sha"] and p["closure_hash"]
        assert "dirty" in p and "inputs" in p
    assert _load(out, "units/alien.json")["provenance"]["inputs"]


def test_real_units_job_loads_and_identity(s0):
    """G-build test 3: identity on 2 units, both starts, from the cached s1 files and Bodies (read like v85_job)."""
    out, _, _ = s0
    S = J.Starts(out)
    for g in UNITS:
        u = S.unit(g)
        for s in "AB":
            s1 = S.load_s1(g, s)
            body = S.load_body(g, s)
            assert body.tracker is not None and S.start(g, s)["entry"]["entry_frames"] > 0
            idt = s0lib.start_identity(g, S.start(g, s)["E"], s1, u["ladder"], u["k_star"])
            assert idt["ok"] and idt["all_steps_equal"], idt
            assert len(d_script(s1, u["ladder"], u["k_ref"])) == S.start(g, s)["floor"]["delta_depth"]


def test_real_units_reports(s0):
    out, summ, _ = s0
    f = _load(out, "floor.json")
    assert len(f["rows"]) == 4
    for row in f["rows"]:
        assert row["F"] == S0.physics_floor(row["delta_depth"])
        assert row["ratio"] == pytest.approx(row["natural_gap"] / row["F"])
    n = _load(out, "natural_spikeins.json")
    assert sorted(n["cells"]) == ["alien/A", "alien/B", "zaxxon/A", "zaxxon/B"]
    assert [t["U"] for t in n["table"]] == [2] and 0.0 <= n["table"][0]["P_room_x2"] <= 1.0 and n["U"] is None
    assert n["cells"]["zaxxon/A"] == n["cells"]["zaxxon/B"] and np.min(n["cells"]["alien/A"]) >= 5000
    d = _load(out, "dice_g4.json")
    assert len(d["segments"]) == 4 and all(sg["G4"] is None or 1.0 <= sg["G4"] <= 4.0 for sg in d["segments"])
    draw = s0lib.unit_order(s0lib.eligible_units())
    dice = {(sg["game"], sg["start"]) for sg in d["segments"] if sg["dice"]}
    assert dice == {(g, "B" if draw.index(g) % 2 else "A") for g in UNITS}
    sl = _load(out, "stalls.json")
    assert sl["runs"] and all(p["length"] >= S0.STALL_MIN for p in sl["plateaus"])
    assert not any(r["game"] in ("boxing", "frostbite", "riverraid", "kangaroo", "enduro") for r in sl["runs"])
    assert _load(out, "seed_grep.json")["PASS"]


def test_real_units_resume(s0):
    out, summ, again = s0
    assert summ["skipped"] == [] and again["skipped"] == [g for g in s0lib.unit_order(s0lib.eligible_units())
                                                         if g in UNITS]
    a = _load(out, "starts.json")
    assert a["units"] == again["first"]["starts.json"]
    assert _load(out, "natural_spikeins.json")["table"] == again["first"]["natural_spikeins.json"]
    # a tampered s1 file makes its unit redo; other entry seeds are refused
    p = out / a["units"]["alien"]["A"]["s1"]
    blob = p.read_bytes()
    p.write_bytes(blob + b"x")
    try:
        assert S0._unit_ok(out, "alien", TEST_SEEDS) is None
    finally:
        p.write_bytes(blob)
    assert S0._unit_ok(out, "alien", TEST_SEEDS) is not None
    with pytest.raises(ValueError):
        S0._unit_ok(out, "alien", s0lib.ENTRY_SEEDS)
