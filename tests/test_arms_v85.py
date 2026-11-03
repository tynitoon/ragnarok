"""v85 instrument and placebo arms through scripts/v85_job.py (Part B items 5 and 7; A.7, §7.2; G-build tests 5 and 8).

Each arm module of ragnarok/evaluation/arms_v85/ (d, o4_dice, gf4, gf, kmac, kmac_exact, kmac_tilde) runs a short
breakout smoke job (seed 9800 only, §12; small caps) and writes a run JSON valid under the shared schema, with
its A.7 / A.13 captions. D plays the smoke lineage open-loop to its end at both starts and the evaluator stop
ends the run there; O_4-dice's re-execution matches every look-ahead, and a tampered look-ahead aborts the run
through the job (ABORTS, counted); K*mac's library is the instrument's; K~mac on a twin unit equals K*mac and,
with no derangement (one smoke unit), equals V2s; kills inside an O_4-dice sandbox pass and inside a Q5
rollout resume to the same JSON (volatile fields apart).
"""
import dataclasses
import json
import shutil

import numpy as np
import pytest

from ragnarok.evaluation import arms_v85
from ragnarok.evaluation import ckpt_v85 as ck
from ragnarok.evaluation.oracle_v85 import d_script, kmac_library
from scripts import v85_job as J
from tests.test_job_v85 import check_schema, stable

SEED, GAME, TWIN = 9800, "breakout", "breakout_twin"
CAP, CAP_O = 20_000, 12_000
HAVE_V84 = (J.s0lib.V84 / "g1b" / f"{GAME}_9712.npz").exists()
pytestmark = pytest.mark.skipif(not HAVE_V84, reason="v84 G1b logs absent")


class Kill(Exception):
    pass


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return J.build_smoke(tmp_path_factory.mktemp("v85arms"), GAME)


def run(d, arm, out, cap, start="A", **kw):
    return J.run(GAME, start, arm, SEED, out=out, v85=d, s0=d, cap=cap, smoke=True, **kw)


def written(path):
    """The run JSON as written on disk, checked against the shared schema."""
    o = json.loads(path.read_text(encoding="utf-8"))
    check_schema(o)
    assert o["arm"] in path.name and o["seed"] == SEED
    return o


@pytest.fixture(scope="module")
def ref(smoke, tmp_path_factory):
    """Uninterrupted runs, computed once per (arm, cap): (JSON as written, path)."""
    d, cache = tmp_path_factory.mktemp("v85armsref"), {}

    def get(arm, cap):
        if (arm, cap) not in cache:
            out = d / f"{arm}_{cap}.json"
            run(smoke, arm, out, cap)
            cache[arm, cap] = (written(out), out)
        return cache[arm, cap]
    return get


def ctx_of(sd, **kw):
    S = J.Starts(sd)
    u = S.unit(GAME)
    c = arms_v85.Ctx(game=GAME, start="A", seed=SEED, E=SEED, s1=S.load_s1(GAME, "A"), other_s1=S.load_s1(GAME, "B"),
                     body=S.load_body(GAME, "A"), ladder=u["ladder"], rungs=u["K_g"], k_first=min(u["K_g"]),
                     k_ref=max(u["K_g"]), cap=CAP, units=[GAME], paths={}, starts=S.data, load_s1=S.load_s1,
                     load_body=S.load_body, smoke=True)
    return dataclasses.replace(c, **kw)


# ---------------------------------------------------------------- registry
def test_arm_hooks_and_refusals(smoke):
    want = dict(d=["kwargs"], o4_dice=["chooser", "kwargs"], gf4=["chooser"], gf=["steer"], kmac=["q5"],
                kmac_exact=["q5"], kmac_tilde=["q5"])
    for arm, hooks_set in want.items():
        mod, hooks = arms_v85.build(arm, ctx_of(smoke))
        assert [k for k in arms_v85.HOOKS if hooks[k] is not None and hooks[k] != {}] == hooks_set, arm
        assert list(arms_v85.stateful(hooks)) == (["chooser"] if "chooser" in hooks_set else []), arm
    assert arms_v85.load("o4_dice").ABORTS[0].__name__ == "DiceMismatch"
    assert len(arms_v85.build("kmac_tilde", ctx_of(smoke))[1]["q5"]) == 0          # smoke, one unit: no derangement
    with pytest.raises(AssertionError):
        arms_v85.build("d", ctx_of(smoke, E=-1))                                 # D's agent seed is E_s
    with pytest.raises(ValueError):
        arms_v85.build("kmac", ctx_of(smoke, other_s1=None))                     # no K*mac row
    with pytest.raises(AssertionError):
        arms_v85.build("kmac_tilde", ctx_of(smoke, smoke=False))                 # a gate run needs >= 2 units


# ---------------------------------------------------------------- D (A.7, A.10.2)
@pytest.mark.parametrize("start", ["A", "B"])
def test_d_reaches_the_smoke_lineage_end(smoke, tmp_path, start):
    S = J.Starts(smoke)
    s1, lad, K_g = S.load_s1(GAME, start), S.unit(GAME)["ladder"], S.unit(GAME)["K_g"]
    n = len(d_script(s1, lad, max(K_g)))
    assert n == len(s1["actions"]), "the smoke lineage ends on its first step at M_k_ref"
    out, cap = tmp_path / f"{GAME}_{start}_d_{SEED}.json", 60 + 4 * n + 20_000
    if start == "A":
        run(smoke, "d", out, cap, start=start)
    else:                                                                       # the CLI path (--smoke)
        J.main([GAME, start, "d", str(SEED), "--smoke", str(smoke), "--cap", str(cap), "--out", str(out)])
    o = written(out)
    ins = o["instrument"]
    assert o["E"] == o["seed"] and o["stopped"] and o["summary"]["expansions"] == 1 and o["billed"] == 60 + 4 * n
    assert o["T"][str(o["k_ref"])] == 60 + 4 * n and all(o["T"][str(k)] is not None for k in o["rungs"])
    assert o["replay"] and not o["interruption"] and o["oracle_frames"] == 0 and o["aborted"] is None
    assert ins["script_steps"] == ins["lineage_steps"] == n and ins["stop_at_script_end"] and ins["lineage_match"]
    assert ins["records_in_script"] == len(o["records"]) - 1 and ins["M_k_ref"] == lad[o["k_ref"]]


# ---------------------------------------------------------------- O_4-dice (A.7, G-build test 8)
def test_o4_dice_matches_its_lookaheads(ref):
    o, _ = ref("o4_dice", CAP_O)
    ins, n = o["instrument"], o["summary"]["expansions"]
    assert o["aborted"] is None and ins["dice_ok"] and ins["dice_checked"] == n == ins["n_pairs"] > 5
    assert ins["m"] == 4 and sum(ins["winner_counts"]) == ins["expansions"] >= n and ins["winner_counts"][0] < n
    assert o["oracle_frames"] > 0 and ins["oracle_frames"] == o["oracle_frames"] and 0 <= ins["o_ties"] <= 1
    assert all(o["T_total"][k] >= o["T"][k] for k in o["T"] if o["T"][k] is not None)


def test_o4_dice_mismatch_aborts_the_run(smoke, tmp_path):
    """A look-ahead tampered at expansion 3 (one frame more) aborts the run through the job: written, counted."""
    def observe(d):
        ex, o = d["ex"], d["hooks"]["chooser"]

        def tampered(ex_, env, e):
            p = o(ex_, env, e)
            if e == 3:
                o._want = o._want[:3] + (o._want[3] + 1,) + o._want[4:]
            return p
        ex.chooser = tampered
    out = tmp_path / f"{GAME}_A_o4_dice_{SEED}.json"
    run(smoke, "o4_dice", out, CAP_O, observe=observe)
    o = written(out)
    assert o["aborted"].startswith("DiceMismatch: expansion 3, winner") and o["summary"]["aborted"]
    assert o["summary"]["expansions"] == 3 and o["exp"]["n"] == 3
    assert o["instrument"]["dice_checked"] == 3 and not o["instrument"]["dice_ok"]
    assert not out.with_suffix(".ckpt").exists()


def test_o4_dice_resumes_after_a_kill_inside_a_sandbox_pass(smoke, ref, tmp_path):
    o, path = ref("o4_dice", CAP_O)
    rng = np.random.default_rng([SEED, 60])
    e_kill, j_kill = int(rng.integers(3, 15)), int(rng.integers(1, 300))
    out, seen = tmp_path / f"{GAME}_A_o4_dice_{SEED}.json", {}

    def observe(d):
        env, ex, real, n = d["env"], d["ex"], d["env"].step, [0]

        def step(a):
            if env._sandbox and ex.exp_n >= e_kill:
                n[0] += 1
                if n[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(a)
        env.step = step
    with pytest.raises(Kill):
        run(smoke, "o4_dice", out, CAP_O, ckpt_s=0, observe=observe)
    assert ck.load(out.with_suffix(".ckpt"))["exp"] == seen["e"] >= e_kill and not out.exists()
    o2 = run(smoke, "o4_dice", out, CAP_O)
    assert stable(out) == stable(path) and o2["instrument"]["dice_checked"] == o["instrument"]["dice_checked"]
    assert o2["provenance"]["segments"][1]["resumed_at_exp"] == seen["e"]


# ---------------------------------------------------------------- placebos GF4 and GF (§7.2)
def test_gf4_and_gf(ref):
    o, _ = ref("gf4", CAP)
    ins, n = o["instrument"], o["summary"]["expansions"]
    assert o["oracle_frames"] == 0 and ins["m"] == 4 and ins["expansions"] >= n > 10
    assert 0 < ins["share_not_first"] < 1 and 0 <= ins["share_one_cell"] < 1
    g, _ = ref("gf", CAP)
    ins, n = g["instrument"], g["summary"]["expansions"]
    assert g["oracle_frames"] == 0 and ins["expansions"] == n > 10 and 0 < ins["greedy_picks"] < n
    assert ins["greedy_share"] == ins["greedy_picks"] / n


# ---------------------------------------------------------------- K*mac, K*mac-exact (A.7)
def test_kmac_and_kmac_exact(smoke, ref):
    S = J.Starts(smoke)
    body = S.load_body(GAME, "A")
    for arm, lin, hold in (("kmac", S.load_s1(GAME, "B"), 4), ("kmac_exact", S.load_s1(GAME, "A"), 1)):
        o, _ = ref(arm, CAP)
        ins = o["instrument"]
        _, rep = kmac_library(lin, body.adapter, hold=hold, keep_priv=hold == 1)
        assert {k: ins[k] for k in rep} == json.loads(json.dumps(rep)), arm
        assert ins["hold"] == hold and ins["size"] > 0 and 0 <= ins["fidelity"] <= 1 and len(ins["sha256"]) == 64
        assert 0 <= ins["share_priv_or_noop"] <= 1 and ins["bandit_k"] == 5 and ins["q5_expansions"] > 0, arm
        assert o["oracle_frames"] == 0 and f"ragnarok/evaluation/arms_v85/{arm}.py" in \
            [f for f, _ in o["provenance"]["closure_files"]]
    ex_ins = ref("kmac_exact", CAP)[0]["instrument"]
    assert ex_ins["keep_priv"] and ex_ins["share_unmapped"] == 0 and ex_ins["fidelity_slot"] == 1.0


def test_kmac_resumes_after_a_kill_inside_a_q5_rollout(smoke, ref, tmp_path):
    o, path = ref("kmac", CAP)
    meta = np.load(path.with_suffix(".npz"))["exp_meta"]
    q5 = [int(r[0]) for r in meta if int(r[2]) == 4 and r[0] >= 2 and r[7] >= 2]
    rng = np.random.default_rng([SEED, 61])
    e_kill = q5[int(rng.integers(len(q5)))]
    j_kill = int(rng.integers(1, int(meta[e_kill, 7]) + 1))
    out, seen = tmp_path / f"{GAME}_A_kmac_{SEED}.json", {}

    def observe(d):
        env, ex, real, n = d["env"], d["ex"], d["env"].step, [0]

        def step(a):
            if ex.exp_n == e_kill and not env._sandbox:
                n[0] += 1
                if n[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(a)
        env.step = step
    with pytest.raises(Kill):
        run(smoke, "kmac", out, CAP, ckpt_s=0, observe=observe)
    assert ck.load(out.with_suffix(".ckpt"))["exp"] == seen["e"] == e_kill and not out.exists()
    o2 = run(smoke, "kmac", out, CAP)
    assert stable(out) == stable(path) and o2["provenance"]["segments"][1]["resumed_at_exp"] == e_kill


# ---------------------------------------------------------------- K~mac (§7.2)
def _same_run(a, b):
    keys = ("records", "T", "T_total", "replay", "summary", "exp", "billed", "stopped")
    return all(a[k] == b[k] for k in keys)


def test_kmac_tilde_without_derangement_equals_v2s(ref):
    o, _ = ref("kmac_tilde", CAP)
    ins = o["instrument"]
    assert ins["no_derangement"] and ins["partner"] is None and ins["size"] == 0 and ins["bandit_k"] == 4
    assert ins["q5_expansions"] == 0 and _same_run(o, ref("v2s", CAP)[0])


def test_kmac_tilde_on_a_twin_unit_equals_kmac(smoke, ref, tmp_path):
    """Two units, breakout and a twin with the same start files: the derangement pairs them, so K~mac at A plays
    the twin's cross-start library at A, which is breakout's own K*mac library."""
    d = tmp_path / "twin"
    d.mkdir()
    for f in smoke.glob("*.pkl"):
        shutil.copy2(f, d / f.name)
    st = json.loads((smoke / "starts.json").read_text(encoding="utf-8"))
    st["units"][TWIN], st["order"] = st["units"][GAME], [GAME, TWIN]
    ck.write_json(d / "starts.json", st)
    out = tmp_path / f"{GAME}_A_kmac_tilde_{SEED}.json"
    run(d, "kmac_tilde", out, CAP)
    o, k = written(out), ref("kmac", CAP)[0]
    ins = o["instrument"]
    assert ins["partner"] == TWIN and ins["partner_start"] == "A" and not ins["no_derangement"]
    assert ins["sha256"] == k["instrument"]["sha256"] and ins["size"] == k["instrument"]["size"] > 0
    assert _same_run(o, k) and ins["q5_expansions"] == k["instrument"]["q5_expansions"] > 0
