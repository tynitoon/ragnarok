"""v85 G-build tests 1-9 (docs/ARC4_V85_PROTOCOL_EN.md Part B, G-build tests; A.6, A.7, §13.2-13.4).

Failures of tests 1-8 after 2 fix rounds give KC-build (confined to the O path: only the selection row and the
dice caption are dropped); test 9 is never a kill. Each test is implemented here or referenced: GBUILD maps
every test number to its nodes, here and in the other v85 test files (test_0 checks that they exist). The
G-build gate is this file plus every referenced node (one pytest process, sequentially). Seed 9800 only (§12);
smoke caps are small, so no run here is a measurement.

1. Isolation: v84's test 1 rules on ragnarok/agent_v85/*.py (imports, game names, banned names, RAM and
   private attributes); importing explorer2s loads no other agent_v85 module (choice: every other agent_v85
   module is a family or process module, item 12 and Q5) and the agent package loads no evaluation module;
   instruments only in ragnarok/evaluation (choice: no agent_v85 file defines an instrument name or touches
   the evaluator-only API sandbox / oracle_frames / replay85 / stop_at / stopped / records / start_from /
   clock_start; every instrument object is defined in ragnarok/evaluation; no module of ragnarok/ outside
   evaluation imports oracle_v85 or arms_v85).
2. Ledger and sandbox: billed never decreases outside the sandbox, a live restore bills exactly 60, a step 4,
   a clone 0; sandbox frames go to oracle_frames only. The rung-crossing case is constructed (choice): from the
   root (s1, below M_{k_first}) the smoke lineage up to M_{k_ref} is played as a dry explorer rollout inside
   env.sandbox(), the tracker saved and restored as OChooser does; Trace and every state hash are unchanged and
   the stop does not fire, while the same rollout played live is traced and fires the stop. State hashes of
   env, ALE, trace, explorer, archive, bandit, hist and tracker are also equal around real O_4 sandbox passes.
3. Start identity on alien and zaxxon (§3: K_g = {3, 4}), both starts, seed 9800 at both (one Body per unit,
   as S0 keys Bodies by (unit, E)): start_v85.start_identity, then a D run from the hash-checked cached Body
   reproduces the lineage's cum at every record and stops at M_{k_ref} on the script's last step.
4. Determinism: for an arm of each hook path (V2s, O_4, O_4-dice, K*map, K*mac, D on breakout; V2s on pong)
   a second smoke run in a fresh child process (PYTHONHASHSEED=0, as the queue) gives the same run JSON
   (volatile fields apart) and the same exp_meta / exp_feat bytes.
5. Resume identity for the hook paths not killed elsewhere: D (mid-script), GF (steering), GF4 (chooser);
   kills mid-sandbox (O_4, O_4-dice), mid-rollout (K*map, Q5), a real process kill and a torn write are in
   test_job_v85 and test_arms_v85.
6. Prefix identity through the job: O_1 (all but the oracle-frame column and T_total), K*map with an empty
   map and Q5 with an empty library (temporary arm modules; everything but arm and instrument) equal V2s,
   including the exp_meta and exp_feat hashes. F_M and the W arms do not exist yet (item 12): skipped.
7. Verdict and calibration code: tests/test_verdict_v85.py (referenced).
8. O_4-dice re-execution equals its look-ahead (actions, frames after the restore, max S) on smoke runs at
   breakout A and B and pong A, through the job.
9. Engineering report, craft_v6_out/v85/gbuild_engineering.json: peak commit (fresh processes), frames per
   second per arm type, checkpoint size and write time (one write of each reference run's final state, plus the
   kill tests' checkpoint files), W steering cost (null: no W arm yet), worker count = min(4, floor(3.6 GB /
   peak)) with 3.6 GB read as 3.6 * 1024 MiB (peak_commit_mb is in MiB). Reports what this module ran.
"""
import ast
import hashlib
import importlib
import json
import math
import os
import pathlib
import pickle
import re
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from ragnarok.agent_v85.explorer2s import Explorer2s
from ragnarok.agent_v85.q5_v85 import Q5
from ragnarok.evaluation import arms_v85, oracle_v85
from ragnarok.evaluation import ckpt_v85 as ck
from ragnarok.evaluation import start_v85 as ST
from ragnarok.evaluation.oracle_v85 import KMap, OChooser, d_script
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, EXCLUDED, RESERVED, SEALED, SECOND
from ragnarok.evaluation.senses_v85 import LOAD_FRAMES, SenseEnv85, Spent, Trace85
from scripts import queue_v85 as Q
from scripts import v85_job as J
from tests.test_job_v85 import Kill, check_schema, stable
from tests.test_senses_v85 import make_start
from tests.test_v2_v84 import ALLOWED, BAD_ATTR, BAD_NAMES

SEED, GAME = 9800, "breakout"
CAP, CAP_O, SMALL = 20_000, 12_000, 6_000
ROOT = J.ROOT
AGENT = ROOT / "ragnarok" / "agent_v85"
FILES = sorted(AGENT.glob("*.py"))
AGENT_PKGS = ("ragnarok.agent_v83", "ragnarok.agent_v84", "ragnarok.agent_v85")
INSTRUMENTS = ("OChooser", "ODice", "DiceMismatch", "GF4", "GF", "KMap", "d_script", "lineage_map", "kmac_library",
               "lib_sha256", "tilde_partners", "tilde_library", "tilde_kmac")
EVAL_API = {"sandbox", "oracle_frames", "replay85", "stop_at", "stopped", "records", "start_from", "clock_start",
            "billed_since_start"}
SMOKE_GAMES = ("breakout", "pong")
HAVE_V84 = all((ST.V84 / "g1b" / f"{g}_{r}.npz").exists() for g in SMOKE_GAMES for r in (9712, 9713))
needs_v84 = pytest.mark.skipif(not HAVE_V84, reason="v84 G1b logs absent")
UNITS3 = ("alien", "zaxxon")
HAVE3 = all((ST.V84 / "g1b" / f"{g}_{r}.npz").exists() for g in UNITS3 for r in (9712, 9713))
ENGINEERING = ROOT / "craft_v6_out" / "v85" / "gbuild_engineering.json"
ENG = dict(kills=[])

GBUILD = {
    1: ["test_1_isolation_imports_names_attributes", "test_1_isolation_import_closure",
        "test_1_instruments_only_in_evaluation"],
    2: ["test_2_ledger_never_decreases_and_each_restore_bills_60",
        "test_2_state_hashes_equal_around_o4_sandbox_passes",
        "test_2_sandbox_rollout_crossing_rungs_leaves_trace_unchanged",
        "tests/test_senses_v85.py::test_start_from_billing_clock_and_cap",
        "tests/test_senses_v85.py::test_sandbox_leaves_no_trace",
        "tests/test_explorer2s_v85.py::test_propose_and_dry_rollout_have_no_side_effects"],
    3: ["test_3_start_identity_both_starts_from_cached_bodies",
        "tests/test_stage0_v85.py::test_real_units_job_loads_and_identity",
        "tests/test_start_v85.py::test_real_start_alien_A"],
    4: ["test_4_two_smoke_runs_are_identical",
        "tests/test_job_v85.py::test_smoke_determinism_and_schema",
        "tests/test_explorer2s_v85.py::test_determinism_and_resume"],
    5: ["test_5_resume_after_a_kill_at_a_random_expansion",
        "tests/test_job_v85.py::test_resume_after_a_real_process_kill",
        "tests/test_job_v85.py::test_resume_after_a_kill_mid_rollout_kmap",
        "tests/test_job_v85.py::test_resume_after_a_kill_inside_an_o4_sandbox_pass_and_a_torn_write",
        "tests/test_arms_v85.py::test_o4_dice_resumes_after_a_kill_inside_a_sandbox_pass",
        "tests/test_arms_v85.py::test_kmac_resumes_after_a_kill_inside_a_q5_rollout",
        "tests/test_senses_v85.py::test_checkpoint_resume_identical",
        "tests/test_explorer2s_v85.py::test_determinism_and_resume"],
    6: ["test_6_prefix_identity_to_v2s", "test_6_f_m_identical_to_v2s_for_250_expansions",
        "test_6_w_arms_identical_to_v2s_until_their_first_two_class_refit",
        "tests/test_oracle_v85.py::test_prefix_identities",
        "tests/test_job_v85.py::test_o1_job_equals_v2s",
        "tests/test_arms_v85.py::test_kmac_tilde_without_derangement_equals_v2s"],
    7: ["tests/test_verdict_v85.py::test_exact_R_when_neither_floor_nor_cap_binds",
        "tests/test_verdict_v85.py::test_censoring_in_the_yardstick",
        "tests/test_verdict_v85.py::test_censoring_treatments_in_the_calibration",
        "tests/test_verdict_v85.py::test_calibration_recovers_planted_dilations_exactly",
        "tests/test_verdict_v85.py::test_room_clauses_intersection_union",
        "tests/test_verdict_v85.py::test_label_precedence_and_gate_verdict",
        "tests/test_verdict_v85.py::test_g1p_g2_and_routing_each_contrast_on_its_own",
        "tests/test_verdict_v85.py::test_cross_fit_rule",
        "tests/test_verdict_v85.py::test_comparator_certificate_and_interruptions",
        "tests/test_verdict_v85.py::test_bootstrap_matches_a_naive_two_level_loop",
        "tests/test_verdict_v85.py::test_g1_report_end_to_end_on_synthetic_runs",
        "tests/test_verdict_v85.py::test_selection_rule_order",
        "tests/test_verdict_v85.py::test_calibration_runtime_under_30_minutes"],
    8: ["test_8_o4_dice_reexecution_matches_its_lookahead",
        "tests/test_oracle_v85.py::test_dice_reexecution_matches_lookahead",
        "tests/test_arms_v85.py::test_o4_dice_matches_its_lookaheads",
        "tests/test_arms_v85.py::test_o4_dice_mismatch_aborts_the_run"],
    9: ["test_9_engineering_report"],
}


# ---------------------------------------------------------------- registry
@pytest.mark.parametrize("n", sorted(GBUILD))
def test_0_gbuild_registry(n):
    """Every G-build test is implemented here or referenced to an existing test function."""
    assert sorted(GBUILD) == list(range(1, 10)) and GBUILD[n]
    for node in GBUILD[n]:
        path, _, name = node.rpartition("::")
        if path:
            assert (ROOT / path).exists(), node
            fn = getattr(importlib.import_module(path[:-3].replace("/", ".")), name, None)
        else:
            fn = globals().get(node)
        assert callable(fn) and fn.__name__.startswith("test_"), node


# ---------------------------------------------------------------- shared smoke starts and runs
@pytest.fixture(scope="module")
def smokes(tmp_path_factory):
    """The S0 layout of each smoke game (v85_job.build_smoke), built once."""
    if not HAVE_V84:
        pytest.skip("v84 G1b logs absent")
    d, cache = tmp_path_factory.mktemp("gbsmoke"), {}

    def get(game):
        if game not in cache:
            cache[game] = J.build_smoke(d / game, game)
        return cache[game]
    return get


def cap_of(sd, game, start, arm):
    if arm == "d":                                    # the script, then room to spare (the stop ends the run)
        S = J.Starts(sd)
        u = S.unit(game)
        return LOAD_FRAMES + 4 * len(d_script(S.load_s1(game, start), u["ladder"], max(u["K_g"]))) + CAP
    return CAP_O if arm in ("o4", "o4_dice") else CAP


def _ckpt_cost(seen, path):
    """One checkpoint write of a finished run's final state (the largest: the archive only grows)."""
    try:
        cp = ck.Checkpointer(path, every_s=0).attach(seen["env"], seen["trace"], arms_v85.stateful(seen["hooks"]))
        cp.write(seen["ex"], seen["ex"].exp_n)
        st = dict(cp.stats(), exp=int(seen["ex"].exp_n))
        cp.remove()
        return st
    except Exception as err:                          # test 9 is never a failure
        return dict(error=repr(err))


@pytest.fixture(scope="module")
def ref(smokes, tmp_path_factory):
    """Uninterrupted in-process smoke runs, computed once per (game, start, arm): (JSON as written, path)."""
    d, cache, ckpts = tmp_path_factory.mktemp("gbref"), {}, {}

    def get(game, start, arm):
        key = (game, start, arm)
        if key not in cache:
            sd = smokes(game)
            out, seen = d / f"{game}_{start}_{arm}_{SEED}.json", {}
            J.run(game, start, arm, SEED, out=out, v85=sd, s0=sd, cap=cap_of(sd, game, start, arm), smoke=True,
                  observe=seen.update)
            cache[key] = (json.loads(out.read_text(encoding="utf-8")), out)
            ckpts[key] = _ckpt_cost(seen, d / f"{game}_{start}_{arm}.eng.ckpt")
        return cache[key]
    get.cache, get.ckpts = cache, ckpts
    return get


@pytest.fixture(scope="module")
def fresh(smokes, tmp_path_factory):
    """The same smoke runs through the CLI in a fresh child process (the parent waits)."""
    d, cache = tmp_path_factory.mktemp("gbfresh"), {}

    def get(game, start, arm):
        key = (game, start, arm)
        if key not in cache:
            sd = smokes(game)
            out = d / f"{game}_{start}_{arm}_{SEED}.json"
            cmd = [sys.executable, str(J.JOB), game, start, arm, str(SEED), "--smoke", str(sd),
                   "--cap", str(cap_of(sd, game, start, arm)), "--out", str(out)]
            env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2")
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
            assert r.returncode == 0, r.stderr[-3000:]
            cache[key] = (json.loads(out.read_text(encoding="utf-8")), out)
        return cache[key]
    get.cache = cache
    return get


def npz_bytes(path):
    z = np.load(pathlib.Path(path).with_suffix(".npz"))
    return z["exp_meta"].tobytes(), z["exp_feat"].tobytes()


def killer(e_kill, j_kill, seen, sandbox=False):
    """An observe hook that raises Kill at the j-th (live or sandbox) step of expansion e_kill."""
    def observe(d):
        env, ex, real, n = d["env"], d["ex"], d["env"].step, [0]

        def step(a):
            if ex.exp_n == e_kill and env._sandbox == sandbox:
                n[0] += 1
                if n[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(a)
        env.step = step
    return observe


def _sha(o):
    return hashlib.sha256(pickle.dumps(o, protocol=4)).hexdigest()


def state_hashes(env, tr, ex):
    """Hashes of env (all but oracle_frames, the one counter a sandbox pass moves), ALE and screens, trace,
    explorer, archive, bandit, hist and tracker."""
    es = env.to_state()
    es.pop("oracle_frames")
    e = env._e
    return dict(env=_sha(es), ale=_sha(e.ale.cloneState()), screen=_sha((e._prev, e._last, e._true_score)),
                counters=(e.frames, e.loads, env.billed), trace=_sha(tr.to_state()), explorer=_sha(ex.to_state()),
                archive=_sha(ex.arch), bandit=_sha(ex.bandit), tracker=_sha(ex.body.tracker),
                hist=_sha((ex.hist, ex.coarse_count, ex.best_S, ex.best_S_billed, ex.best_S_exp, ex.any_rise, ex.exp_n)))


def started(sd, start="A", stop_at=None, cap=CAP, **hooks):
    """A direct V2s set-up on a smoke start (no job): env, trace, explorer, obs, lives, s1."""
    S = J.Starts(sd)
    s1 = S.load_s1(GAME, start)
    tr = Trace85(stop_at)
    env = SenseEnv85(GAME, seed=SEED, trace=tr)
    obs, lives = env.start_from(s1["blob"], s1["S"], cap=cap)
    ex = Explorer2s(S.load_body(GAME, start), budget=cap, seed=SEED, **hooks)
    return env, tr, ex, obs, lives, s1


# ---------------------------------------------------------------- 1. isolation
def _agent_trees():
    return [(f, ast.parse(f.read_text(encoding="utf-8"))) for f in FILES]


def test_1_isolation_imports_names_attributes():
    names = DEV + CALIBRATION + SEALED + SECOND + RESERVED + EXCLUDED
    pat = re.compile(r"\b(" + "|".join(names) + r")\b", re.I)
    assert {"explorer2s.py", "q5_v85.py"} <= {f.name for f in FILES}
    for f, tree in _agent_trees():
        assert not pat.search(f.read_text(encoding="utf-8")), f.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mods:
                    assert m in ALLOWED or m.startswith(AGENT_PKGS), (f.name, m)
            if isinstance(node, ast.Attribute):
                own = isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")
                assert own or node.attr not in BAD_ATTR, (f.name, node.lineno, node.attr)
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    assert own, (f.name, node.lineno, node.attr)
            if isinstance(node, ast.Name):
                assert node.id not in BAD_NAMES, (f.name, node.lineno, node.id)


def test_1_isolation_import_closure():
    """explorer2s imports no family module (no other agent_v85 module); the agent package loads no evaluation
    module, no emulator and no torch."""
    tree = ast.parse((AGENT / "explorer2s.py").read_text(encoding="utf-8"))
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert mods and not any(m.startswith("ragnarok.agent_v85") or "macros" in m or "critic" in m for m in mods), mods
    rest = ", ".join(f"ragnarok.agent_v85.{f.stem}" for f in FILES if f.stem != "__init__")
    code = ("import sys\n"
            "import ragnarok.agent_v85.explorer2s\n"
            "own = sorted(m for m in sys.modules if m.startswith('ragnarok.agent_v85.'))\n"
            "assert own == ['ragnarok.agent_v85.explorer2s'], own\n"
            f"import {rest}\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'ale_py')\n"
            "       or m.startswith('ragnarok.evaluation') or m.startswith('scripts.v85')]\n"
            "print(bad); assert not bad\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_1_instruments_only_in_evaluation():
    for f, tree in _agent_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in INSTRUMENTS, (f.name, node.name)
            if isinstance(node, ast.Attribute):
                assert node.attr not in EVAL_API, (f.name, node.lineno, node.attr)
    ev = ROOT / "ragnarok" / "evaluation"
    for n in INSTRUMENTS:
        obj = getattr(oracle_v85, n)
        assert obj.__module__ == "ragnarok.evaluation.oracle_v85", n
    assert ev in pathlib.Path(oracle_v85.__file__).resolve().parents
    pat = re.compile(r"\b(oracle_v85|arms_v85)\b")
    for p in (ROOT / "ragnarok").rglob("*.py"):                  # no module outside evaluation imports them
        text = p.read_text(encoding="utf-8", errors="replace")
        if ev in p.resolve().parents or not pat.search(text):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not any(pat.search(m) for m in mods), (p, node.lineno)


# ---------------------------------------------------------------- 2. ledger and sandbox
def test_2_ledger_never_decreases_and_each_restore_bills_60():
    blob, S = make_start(GAME)
    tr = Trace85()
    env = SenseEnv85(GAME, seed=SEED, trace=tr)
    env.start_from(blob, S)
    rng = np.random.default_rng([SEED, 20])
    hs, seen, passes = [env.clone()], [env.billed], 0
    for t in range(800):
        b = env.billed
        end = env.step(int(rng.integers(env.n_actions)))[3]
        assert env.billed == b + 4
        if t % 40 == 0:
            b = env.billed
            hs.append(env.clone())
            assert env.billed == b
        if end or t % 60 == 59:
            b = env.billed
            env.restore(hs[int(rng.integers(len(hs)))])
            assert env.billed == b + LOAD_FRAMES
        if t % 50 == 25:
            b, of, n, recs = env.billed, env.oracle_frames, int(rng.integers(5, 60)), list(tr.records)
            with env.sandbox():
                env.restore(hs[int(rng.integers(len(hs)))])
                for _ in range(n):
                    env.step(int(rng.integers(env.n_actions)))
            assert env.billed == b and env.oracle_frames == of + LOAD_FRAMES + 4 * n and tr.records == recs
            passes += 1
        seen.append(env.billed)
    assert all(b >= a for a, b in zip(seen, seen[1:])) and passes == 16 and env.oracle_frames > 0


class CheckedO4(OChooser):
    """O_4 whose sandbox pass (every `every`-th expansion) is wrapped in state-hash checks."""

    def __init__(self, tr, every=2):
        super().__init__(4)
        self.tr, self.every, self.passes = tr, every, 0

    def __call__(self, ex, env, e):
        if e % self.every:
            return super().__call__(ex, env, e)
        before, b, of = state_hashes(env, self.tr, ex), env.billed, env.oracle_frames
        w = super().__call__(ex, env, e)
        after = state_hashes(env, self.tr, ex)
        assert after == before, (e, [k for k in before if before[k] != after[k]])
        assert env.billed == b and env.oracle_frames > of
        self.passes += 1
        return w


@needs_v84
def test_2_state_hashes_equal_around_o4_sandbox_passes(smokes):
    sd = smokes(GAME)
    tr0 = Trace85()
    o = CheckedO4(tr0)
    env, tr, ex, obs, lives, s1 = started(sd, cap=SMALL, chooser=o, after=o.after)
    o.tr = tr
    led, real_restore, real_step = [], env.restore, env.step

    def restore(h):
        b = env.billed
        out = real_restore(h)
        if not env._sandbox:
            assert env.billed == b + LOAD_FRAMES
            led.append(env.billed)
        return out

    def step(a):
        b = env.billed
        out = real_step(a)
        if not env._sandbox:
            assert env.billed == b + 4
            led.append(env.billed)
        return out
    env.restore, env.step = restore, step
    summ = ex.run(env, obs, lives, s1["S"], spent=Spent)
    assert o.passes >= 3 and summ["expansions"] == ex.exp_n == len(o.outcomes) and env.oracle_frames > 0
    assert led and all(b >= a for a, b in zip(led, led[1:])) and env.billed_since_start >= SMALL


@needs_v84
def test_2_sandbox_rollout_crossing_rungs_leaves_trace_unchanged(smokes):
    sd = smokes(GAME)
    S = J.Starts(sd)
    u = S.unit(GAME)
    lad, K_g = u["ladder"], u["K_g"]
    k_ref, E0 = max(K_g), 4

    def stop(x, e):
        if e == E0:
            raise Kill
    env, tr, ex, obs, lives, s1 = started(sd, stop_at=lad[k_ref], boundary=stop)
    with pytest.raises(Kill):
        ex.run(env, obs, lives, s1["S"], spent=Spent)
    script = d_script(s1, lad, k_ref)
    A = ex.arch
    root = int(np.flatnonzero(A.used & (A.depth == 0))[0])
    assert A.S[root] == s1["S"] < lad[min(K_g)] and not tr.stopped
    before, of = state_hashes(env, tr, ex), env.oracle_frames
    saved = pickle.dumps(ex.body.tracker)
    with env.sandbox():
        out = ex._rollout(env, root, 0, ex.slots[0], ex.stream(E0, 0, 1), E0, dry=True, script=script)
    ex.body.tracker = pickle.loads(saved)
    assert out["steps"] == len(script) and out["max_S"] == float(s1["cum"][len(script) - 1]) >= lad[k_ref]
    assert state_hashes(env, tr, ex) == before and not tr.stopped
    assert env.oracle_frames == of + LOAD_FRAMES + 4 * len(script)
    # the same rollout played live: traced, the rungs recorded, the evaluator stop fires
    n_rec = len(tr.records)
    live = ex._rollout(env, root, 0, ex.slots[0], ex.stream(E0, 0, 1000), E0, script=script)
    assert live["acts"] == out["acts"] and live["max_S"] == out["max_S"]
    assert tr.stopped and len(tr.records) > n_rec and tr.records[-1][1] >= lad[k_ref]
    assert all(any(r[1] >= lad[k] for r in tr.records[n_rec:]) for k in K_g)
    with pytest.raises(Spent):
        env.restore(int(A.handle[root]))


# ---------------------------------------------------------------- 3. start identity
@pytest.mark.skipif(not HAVE3, reason="v84 G1b logs of alien / zaxxon absent")
@pytest.mark.parametrize("game", UNITS3)
def test_3_start_identity_both_starts_from_cached_bodies(game, tmp_path):
    k_star, lad = ST.eligible_units()[game], ST.ladder(game)
    c = ST.entry_cache(game, SEED, tmp_path / f"{game}_{SEED}_body.pkl")
    s1s = {s: ST.top_lineage(game, ST.REF_SEEDS[s], lad, k_star) for s in "AB"}
    assert all(v is not None for v in s1s.values()), "invalid start"
    K_g = ST.counted_rungs(lad, k_star, [v["S"] for v in s1s.values()],
                           [ST.ref_run(game, ST.REF_SEEDS[s])[0] for s in "AB"])
    assert len(K_g) >= 2
    k_ref = max(K_g)
    for st, s1 in s1s.items():
        idt = ST.start_identity(game, SEED, s1, lad, k_star)
        assert idt["ok"] and idt["all_steps_equal"], (st, idt)
        body = ST.load_body(c["path"], c["sha256"])           # every arm at this start loads this Body
        script = d_script(s1, lad, k_ref)
        n, cap = len(script), LOAD_FRAMES + 4 * len(script) + 1000
        tr = Trace85(stop_at=lad[k_ref])
        env = SenseEnv85(game, seed=SEED, trace=tr)
        obs, lives = env.start_from(s1["blob"], s1["S"], cap=cap)
        ex = Explorer2s(body, budget=cap, seed=SEED, script=script)
        ex.run(env, obs, lives, s1["S"], spent=Spent)
        assert tr.stopped and ex.exp_n == 1 and env.billed_since_start == LOAD_FRAMES + 4 * n, st
        recs = tr.records[1:]
        assert recs and all(r[1] == float(s1["cum"][r[4] - 1]) for r in recs), st
        top = next(r for r in recs if r[1] >= lad[k_ref])
        assert top[4] == n and tr.replay85(top[2]) == (top[1], top[3], top[4]), st


# ---------------------------------------------------------------- 4. determinism
ARMS4 = (("breakout", "A", "v2s"), ("breakout", "A", "o4"), ("breakout", "A", "o4_dice"), ("breakout", "A", "kmap"),
         ("breakout", "A", "kmac"), ("breakout", "A", "d"), ("pong", "A", "v2s"))


@needs_v84
@pytest.mark.parametrize("game,start,arm", ARMS4)
def test_4_two_smoke_runs_are_identical(ref, fresh, game, start, arm):
    o, path = ref(game, start, arm)
    o2, path2 = fresh(game, start, arm)
    check_schema(o)
    assert stable(path2) == stable(path) and npz_bytes(path2) == npz_bytes(path)
    assert o2["provenance"]["hashseed"] == "0" and o2["aborted"] is None and o2["summary"]["expansions"] >= 1


# ---------------------------------------------------------------- 5. resume identity
KILLS = dict(d=70, gf=71, gf4=72)


@needs_v84
@pytest.mark.parametrize("arm", list(KILLS))
def test_5_resume_after_a_kill_at_a_random_expansion(smokes, ref, tmp_path, arm):
    o, path = ref(GAME, "A", arm)
    meta = np.load(path.with_suffix(".npz"))["exp_meta"]
    rng = np.random.default_rng([SEED, KILLS[arm]])
    if arm == "d":                                            # inside the scripted rollout of expansion 0
        e_kill = 0
        j_kill = int(rng.integers(1, int(meta[0, 7])))
    else:
        assert len(meta) > 5
        e_kill = int(rng.integers(2, min(40, len(meta) - 1)))
        j_kill = int(rng.integers(1, int(meta[e_kill, 7]) + 1))
    out, seen, sd = tmp_path / f"{GAME}_A_{arm}_{SEED}.json", {}, smokes(GAME)
    kw = dict(out=out, v85=sd, s0=sd, cap=cap_of(sd, GAME, "A", arm), smoke=True)
    with pytest.raises(Kill):
        J.run(GAME, "A", arm, SEED, ckpt_s=0, observe=killer(e_kill, j_kill, seen), **kw)
    ckp = out.with_suffix(".ckpt")
    assert ck.load(ckp)["exp"] == seen["e"] == e_kill and not out.exists()
    ENG["kills"].append(dict(arm=arm, exp=e_kill, ckpt_mb=round(ckp.stat().st_size / 2 ** 20, 3)))
    o2 = J.run(GAME, "A", arm, SEED, **kw)
    assert stable(out) == stable(path) and npz_bytes(out) == npz_bytes(path)
    assert o2["provenance"]["segments"][1]["resumed_at_exp"] == e_kill and not ckp.exists()


# ---------------------------------------------------------------- 6. prefix identity
TEMP_ARMS = dict(zz_kmap_empty=lambda ctx: dict(steer=KMap([])), zz_q5_empty=lambda ctx: dict(q5=Q5([])))


@needs_v84
@pytest.mark.parametrize("arm", ["o1", "zz_kmap_empty", "zz_q5_empty"])
def test_6_prefix_identity_to_v2s(ref, monkeypatch, arm):
    for name, build in TEMP_ARMS.items():
        mod = types.ModuleType(f"ragnarok.evaluation.arms_v85.{name}")
        mod.build = build
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    v, _ = ref(GAME, "A", "v2s")
    o, _ = ref(GAME, "A", arm)
    skip = set(J.VOLATILE) | {"arm", "instrument", "records", "T_total", "oracle_frames"}
    keys = [k for k in v if k not in skip]
    assert set(keys) <= set(o) and {k: o[k] for k in keys} == {k: v[k] for k in keys}    # incl. exp hashes
    assert [r[:4] for r in o["records"]] == [r[:4] for r in v["records"]] and v["summary"]["expansions"] > 10
    if arm == "o1":
        assert o["oracle_frames"] > 0 and o["instrument"]["m"] == 1 and o["instrument"]["o_ties"] == 1.0
    else:
        assert o["records"] == v["records"] and o["T_total"] == v["T_total"] and o["oracle_frames"] == 0


@pytest.mark.skip(reason="item 12, written at F1")
def test_6_f_m_identical_to_v2s_for_250_expansions():
    """F_M is byte-identical to V2s for its first 250 expansions (its arm module is written at F1)."""


@pytest.mark.skip(reason="item 12, written at F1")
def test_6_w_arms_identical_to_v2s_until_their_first_two_class_refit():
    """Every W arm is byte-identical to V2s until its first two-class refit (written at F1)."""


# ---------------------------------------------------------------- 8. O_4-dice
@needs_v84
@pytest.mark.parametrize("game,start", [("breakout", "A"), ("breakout", "B"), ("pong", "A")])
def test_8_o4_dice_reexecution_matches_its_lookahead(ref, game, start):
    o, _ = ref(game, start, "o4_dice")
    ins, n = o["instrument"], o["summary"]["expansions"]
    assert o["aborted"] is None and ins["dice_ok"] and ins["dice_checked"] == n == ins["n_pairs"] > 5
    assert o["oracle_frames"] > 0 and sum(ins["winner_counts"]) == ins["expansions"] >= n


# ---------------------------------------------------------------- 9. engineering (never a failure)
TYPES = dict(v2s="V2s", o1="O (sandbox)", o4="O (sandbox)", o4_dice="O (sandbox)", kmap="steering", gf="steering",
             kmac="Q5", kmac_exact="Q5", kmac_tilde="Q5", gf4="chooser, no look-ahead", d="D (script)")


def _row(o, process):
    ins = o.get("instrument") or {}
    mf = int(ins.get("map_frames", 0) or 0)
    w = max(float(o["wall_s"]), 1e-9)
    emu = int(o["billed"]) + int(o["oracle_frames"]) + mf
    return dict(game=o["game"], start=o["start"], arm=o["arm"], type=TYPES.get(o["arm"], "identity probe"),
                process=process, cap=o["cap"], billed=o["billed"], oracle_frames=o["oracle_frames"], map_frames=mf,
                expansions=o["summary"]["expansions"], wall_s=round(w, 2), billed_per_s=round(o["billed"] / w),
                emulated_per_s=round(emu / w), peak_ws_mb=round(o["peak_ws_mb"], 1),
                peak_commit_mb=round(o["peak_commit_mb"], 1))


def test_9_engineering_report(ref, fresh):
    rep = dict(test="G-build test 9 (Part B): engineering figures, never a kill",
               note="smoke runs on seed 9800 at small caps (never a measurement); a 4M run's archive grows to the "
                    "20k-cell cap, so its peak commit and checkpoints are larger than here. 'fresh' rows are child "
                    "processes (their peak commit is the run's own); 'pytest' rows share the test process (its "
                    "peak is cumulative over the session).",
               runs=[], per_type={}, checkpoints=[], kill_checkpoints=list(ENG["kills"]), errors=[])
    try:
        rep["provenance"] = J.stamp((), cwd=str(ROOT), paths=J.PATHS)
        rows = [_row(o, "fresh") for o, _ in fresh.cache.values()] + [_row(o, "pytest") for o, _ in ref.cache.values()]
        rep["runs"] = rows
        for t in sorted({r["type"] for r in rows}):
            rs = [r for r in rows if r["type"] == t]
            fr = [r for r in rs if r["process"] == "fresh"] or rs
            rep["per_type"][t] = dict(runs=len(rs), emulated_per_s=round(float(np.mean([r["emulated_per_s"] for r in fr]))),
                                      billed_per_s=round(float(np.mean([r["billed_per_s"] for r in fr]))),
                                      peak_commit_mb_fresh=max((r["peak_commit_mb"] for r in rs
                                                                if r["process"] == "fresh"), default=None))
        rep["checkpoints"] = [dict(game=g, start=s, arm=a, **st) for (g, s, a), st in ref.ckpts.items()]
        ok = [c for c in rep["checkpoints"] if "max_mb" in c]
        rep["checkpoint_max_mb"] = max((c["max_mb"] for c in ok), default=None)
        rep["checkpoint_max_write_s"] = max((c["max_s"] for c in ok), default=None)
        peak = max((r["peak_commit_mb"] for r in rows if r["process"] == "fresh"), default=None)
        rep["peak_commit_mb"] = peak
        rep["worker_count"] = None if not peak else min(4, math.floor(3.6 * 1024 / peak))
        rep["worker_rule"] = "min(4, floor(3.6 GB / measured peak commit)) (§13.2); 3.6 GB = 3.6 * 1024 MiB"
        rep["W_steering_cost"] = None
        rep["W_note"] = "no W arm yet (Part B item 12, written at F1)"
        rep["free_gb_min_ram_commit"] = round(Q.free_gb(), 2) if os.name == "nt" else None
        rep["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as err:                                   # never a failure
        rep["errors"].append(repr(err))
    try:
        ck.write_json(ENGINEERING, rep)
    except Exception as err:
        print("engineering report not written:", repr(err))
