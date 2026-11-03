"""v85_job, ckpt_v85, arms_v85 and queue_v85 (Part B items 6, 7, 11).

G-build test 4: two smoke runs on seed 9800 give the same run JSON. Test 5: a kill (a real process kill at a
random moment, a kill mid-rollout at a random expansion, a kill inside an O_4 sandbox pass, a torn checkpoint
write) then a resume gives the same run JSON apart from the volatile fields (wall, memory, provenance). Also:
the shared run-JSON schema, O_1 = V2s through the job, the arm registry, the closure hash, the queue's job
lists, claims and gates. Seed 9800 only (§12); smoke caps are small.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from ragnarok.evaluation import arms_v85
from ragnarok.evaluation import ckpt_v85 as ck
from scripts import queue_v85 as Q
from scripts import v85_job as J

SEED, GAME = 9800, "breakout"
CAP_V, CAP_O = 30_000, 12_000
HAVE_V84 = (J.s0lib.V84 / "g1b" / f"{GAME}_9712.npz").exists()
needs_v84 = pytest.mark.skipif(not HAVE_V84, reason="v84 G1b logs absent")
REQ = ("game", "start", "arm", "seed", "cap", "k_ref", "ladder", "S1", "rungs", "records", "T", "T_total", "replay",
       "interruption", "summary", "instrument", "entry", "oracle_frames", "wall_s", "peak_ws_mb", "peak_commit_mb",
       "claim_header", "senses", "provenance")


class Kill(Exception):
    pass


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    if not HAVE_V84:
        pytest.skip("v84 G1b logs absent")
    return J.build_smoke(tmp_path_factory.mktemp("v85smoke"), GAME)


def run(d, arm, out, cap, **kw):
    return J.run(GAME, "A", arm, SEED, out=out, v85=d, s0=d, cap=cap, smoke=True, **kw)


def stable(path):
    """The run JSON's bytes without its volatile fields."""
    o = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return json.dumps({k: v for k, v in o.items() if k not in J.VOLATILE}, indent=1)


@pytest.fixture(scope="module")
def ref(smoke, tmp_path_factory):
    """Uninterrupted reference runs, computed once."""
    d, cache = tmp_path_factory.mktemp("v85ref"), {}

    def get(arm, cap):
        if (arm, cap) not in cache:
            out = d / f"{arm}_{cap}.json"
            cache[arm, cap] = (run(smoke, arm, out, cap), out)
        return cache[arm, cap]
    return get


def check_schema(o, lad_len=11):
    assert all(k in o for k in REQ)
    assert o["start"] in ("A", "B") and isinstance(o["seed"], int) and o["cap"] > 0 and len(o["ladder"]) == lad_len
    assert set(o["T"]) == set(o["T_total"]) == {str(k) for k in range(1, lad_len)}
    recs = o["records"]
    assert recs and all(len(r) == 5 for r in recs) and recs[0][0] == 0 and recs[0][1] == o["S1"]
    assert all(a[0] <= b[0] and a[1] < b[1] and a[4] <= b[4] for a, b in zip(recs, recs[1:]))
    for k in range(1, lad_len):
        r = next((r for r in recs if r[1] >= o["ladder"][k]), None)
        assert o["T"][str(k)] == (None if r is None else r[0])
        assert o["T_total"][str(k)] == (None if r is None else r[0] + r[4])
    assert o["k_ref"] == max(o["rungs"]) and o["k_first"] == min(o["rungs"]) and o["stopped"] in (True, False)
    assert all(set(x) == {"k", "ok", "replayed", "recorded"} for x in o["replay"])
    assert o["interruption"] == (not all(x["ok"] for x in o["replay"]))
    assert set(o["entry"]) == {"sha256", "entry_frames"} and o["claim_header"] == ck.CLAIM_HEADER
    assert o["senses"] == ck.SENSES
    assert o["exp"]["n"] == o["summary"]["expansions"] and len(o["exp"]["meta_sha256"]) == 64
    p = o["provenance"]
    assert all(k in p for k in ("commit", "dirty", "closure_hash", "inputs", "segments")) and len(p["closure_hash"]) == 64
    assert "scripts/v85_job.py" in [f for f, _ in p["closure_files"]] and not p["closure_foreign"]


# ---------------------------------------------------------------- G-build test 4 and the schema
@needs_v84
def test_smoke_determinism_and_schema(smoke, ref, tmp_path):
    o, path = ref("v2s", CAP_V)
    check_schema(o)
    assert o["billed"] >= CAP_V and o["oracle_frames"] == 0 and o["instrument"] == {} and o["aborted"] is None
    assert o["T"]["2"] is not None and o["replay"] and not o["interruption"], "the smoke run should cross M2"
    assert not path.with_suffix(".ckpt").exists()                      # the rolling checkpoint is removed
    z = np.load(path.with_suffix(".npz"))
    assert hashlib.sha256(z["exp_meta"].tobytes()).hexdigest() == o["exp"]["meta_sha256"]
    assert z["exp_feat"].shape == (o["exp"]["n"], 44) and z["exp_meta"][-1, 0] == o["exp"]["n"] - 1
    o2 = run(smoke, "v2s", tmp_path / "again.json", CAP_V)
    assert stable(tmp_path / "again.json") == stable(path) and o2["records"] == o["records"]
    done = json.loads((tmp_path / "again.json").read_text(encoding="utf-8"))
    assert run(smoke, "v2s", tmp_path / "again.json", CAP_V) == done        # done: not rerun


# ---------------------------------------------------------------- test 5: kills and resumes
@needs_v84
def test_resume_after_a_real_process_kill(smoke, ref, tmp_path):
    o, path = ref("v2s", CAP_V)
    out = tmp_path / "killed.json"
    cmd = [sys.executable, str(J.ROOT / "scripts" / "v85_job.py"), GAME, "A", "v2s", str(SEED), "--smoke", str(smoke),
           "--cap", str(CAP_V), "--ckpt-s", "0", "--out", str(out)]
    p = subprocess.Popen(cmd, cwd=J.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while not out.with_suffix(".ckpt").exists():
        assert p.poll() is None and time.time() - t0 < 180, "no checkpoint written"
        time.sleep(0.05)
    time.sleep(float(np.random.default_rng([SEED, 50]).uniform(1.0, 4.0)))
    p.kill()
    p.wait()
    assert not out.exists(), "the run ended before the kill"
    e_ck = ck.load(out.with_suffix(".ckpt"))["exp"]
    r = subprocess.run(cmd, cwd=J.ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert stable(out) == stable(path)
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert len(prov["segments"]) == 2 and prov["segments"][1]["resumed_at_exp"] == e_ck > 0
    assert prov["closure_consistent"] and not out.with_suffix(".ckpt").exists()
    files = [f for f, _ in prov["closure_files"]]                     # a fresh process: V2s's own closure
    assert "ragnarok/evaluation/arms_v85/v2s.py" in files and "ragnarok/evaluation/oracle_v85.py" not in files


@needs_v84
def test_resume_after_a_kill_mid_rollout_kmap(smoke, ref, tmp_path):
    o, path = ref("kmap", CAP_V)
    check_schema(o)
    ins = o["instrument"]
    assert ins["coins"] > 0 and ins["map_size"] > 0 and len(ins["map_sha256"]) == 64 and "MAP_WEAK" in ins
    rng = np.random.default_rng([SEED, 51])
    e_kill, j_kill = int(rng.integers(5, 40)), int(rng.integers(1, 100))
    out = tmp_path / "k.json"

    def observe(d):
        env, ex, real, n = d["env"], d["ex"], d["env"].step, [0]

        def step(a):
            if ex.exp_n >= e_kill:
                n[0] += 1
                if n[0] == j_kill:
                    seen["e"] = ex.exp_n
                    raise Kill
            return real(a)
        env.step = step
    seen = {}
    with pytest.raises(Kill):
        run(smoke, "kmap", out, CAP_V, ckpt_s=0, observe=observe)
    assert ck.load(out.with_suffix(".ckpt"))["exp"] == seen["e"] >= e_kill and not out.exists()
    o2 = run(smoke, "kmap", out, CAP_V)                               # resumed (no further checkpoint)
    assert stable(out) == stable(path) and o2["provenance"]["segments"][1]["resumed_at_exp"] == seen["e"]


@needs_v84
def test_resume_after_a_kill_inside_an_o4_sandbox_pass_and_a_torn_write(smoke, ref, tmp_path):
    o, path = ref("o4", CAP_O)
    check_schema(o)
    ins = o["instrument"]
    assert o["oracle_frames"] > 0 and ins["oracle_frames"] == o["oracle_frames"] and ins["m"] == 4
    assert ins["expansions"] >= o["summary"]["expansions"] and sum(ins["winner_counts"]) == ins["expansions"]
    assert 0 <= ins["o_ties"] <= 1 and ins["n_pairs"] == o["summary"]["expansions"] and "rank_corr" in ins
    rng = np.random.default_rng([SEED, 52])
    e_kill, j_kill = int(rng.integers(3, 15)), int(rng.integers(1, 300))
    out = tmp_path / "o.json"
    seen = {}

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
        run(smoke, "o4", out, CAP_O, ckpt_s=0, observe=observe)
    ckp = out.with_suffix(".ckpt")
    assert ck.load(ckp)["exp"] == seen["e"] >= e_kill and not out.exists()
    tmp = ckp.with_name(ckp.name + ".tmp")
    tmp.write_bytes(b"\x80\x05torn")                                    # a write cut by the kill: never read
    o2 = run(smoke, "o4", out, CAP_O)
    assert stable(out) == stable(path) and not tmp.exists()
    assert o2["provenance"]["segments"][1]["resumed_at_exp"] == seen["e"]


# ---------------------------------------------------------------- O_1 through the job = V2s (test 6 plumbing)
@needs_v84
def test_o1_job_equals_v2s(smoke, ref, tmp_path):
    v, _ = ref("v2s", CAP_V)
    o = run(smoke, "o1", tmp_path / "o1.json", CAP_V)
    assert [r[:4] for r in o["records"]] == [r[:4] for r in v["records"]] and o["T"] == v["T"]
    assert o["summary"] == v["summary"] and o["replay"] == v["replay"] and o["oracle_frames"] > 0
    assert o["instrument"]["m"] == 1 and o["instrument"]["o_ties"] == 1.0


# ---------------------------------------------------------------- generic hooks: kwargs, stateful objects, ABORTS
class Boom(RuntimeError):
    pass


class Bomb:
    """A chooser that aborts the run at expansion 5."""

    def __call__(self, ex, env, e):
        if e == 5:
            raise Boom("at 5")
        return ex.propose(e, 0)

    def to_state(self):
        return {}

    def from_state(self, d):
        pass


class After:
    def __init__(self):
        self.calls = 0

    def __call__(self, ex, env, e, out):
        self.calls += 1

    def to_state(self):
        return dict(calls=self.calls)

    def from_state(self, d):
        self.calls = d["calls"]


@needs_v84
def test_generic_hooks_and_aborts(smoke, tmp_path, monkeypatch):
    import types
    mod = types.ModuleType("ragnarok.evaluation.arms_v85.zz_test_abort")
    mod.ABORTS = (Boom,)
    mod.build = lambda ctx: dict(chooser=Bomb(), kwargs=dict(after=After()))
    mod.captions = lambda hooks: dict(after_calls=hooks["kwargs"]["after"].calls)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    assert list(arms_v85.stateful(arms_v85.build("zz_test_abort", None)[1])) == ["chooser", "after"]
    o = run(smoke, "zz_test_abort", tmp_path / "a.json", 10_000)
    assert o["aborted"] == "Boom: at 5" and o["summary"]["expansions"] == 5 and o["summary"]["aborted"]
    assert o["instrument"] == dict(after_calls=5) and o["exp"]["n"] == 5


# ---------------------------------------------------------------- units
def test_checkpointer_rules(tmp_path):
    with pytest.raises(AssertionError):
        ck.Checkpointer(tmp_path / "x.ckpt", every_s=301)
    assert ck.load(tmp_path / "none.ckpt") is None

    class Ex:
        def to_state(self):
            return dict(a=1)

    class Env(Ex):
        pass
    c = ck.Checkpointer(tmp_path / "x.ckpt", every_s=0, key=dict(k=1)).attach(Env(), Env(), None, lambda: dict(m=2))
    c(Ex(), 7)
    st = ck.load(tmp_path / "x.ckpt", dict(k=1))
    assert st["exp"] == 7 and st["meta"] == dict(m=2) and st["hooks"] == {} and c.stats()["writes"] == 1
    with pytest.raises(ValueError):
        ck.load(tmp_path / "x.ckpt", dict(k=2))
    c.remove()
    assert not (tmp_path / "x.ckpt").exists()


def test_closure_hash(tmp_path):
    root = tmp_path / "proj"
    (root / "zz_clos_pkg").mkdir(parents=True)
    f, g = root / "zz_clos_pkg" / "m.py", root / "job.py"
    f.write_text("x = 1\n")
    g.write_text("y = 2\n")
    sys.path.insert(0, str(root))
    try:
        import zz_clos_pkg.m  # noqa: F401
        h1, pairs, foreign = ck.closure_hash(root, [g])
        assert [p for p, _ in pairs] == ["job.py", "zz_clos_pkg/m.py"] and h1 == ck.closure_hash(root, [g])[0]
        g.write_text("y = 3\n")
        assert ck.closure_hash(root, [g])[0] != h1
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("zz_clos_pkg.m", None)
        sys.modules.pop("zz_clos_pkg", None)


@needs_v84
def test_arm_registry(smoke):
    S = J.Starts(smoke)
    s1 = S.load_s1(GAME, "A")
    ctx = arms_v85.Ctx(game=GAME, start="A", seed=SEED, E=SEED, s1=s1, other_s1=S.load_s1(GAME, "B"),
                       body=S.load_body(GAME, "A"), ladder=S.unit(GAME)["ladder"], rungs=[2, 3], k_first=2, k_ref=3,
                       cap=1000, units=[GAME], paths={}, starts=S.data, load_s1=S.load_s1, load_body=S.load_body,
                       smoke=True)
    for arm, want in (("v2s", []), ("o1", ["chooser", "kwargs"]), ("o4", ["chooser", "kwargs"]), ("kmap", ["steer"])):
        mod, hooks = arms_v85.build(arm, ctx)
        assert set(arms_v85.HOOKS) <= set(hooks)
        assert [k for k in arms_v85.HOOKS if hooks[k]] == want
        assert list(arms_v85.stateful(hooks)) == (["chooser"] if "chooser" in want else [])
    assert arms_v85.captions(arms_v85.load("v2s"), {}, {}) == {}
    for bad in ("../x", "V2s", "o4.x", ""):
        with pytest.raises(ValueError):
            arms_v85.load(bad)


# ---------------------------------------------------------------- queue
def _fake_v85(tmp_path, monkeypatch):
    v = tmp_path / "v85"
    (v / "s0").mkdir(parents=True)
    units = {}
    for i, g in enumerate(("u0", "u1", "u2", "space_invaders")):
        units[g] = dict(K_g=[2, 3] if g != "u2" else [2], ref_gap=(i + 1) * 1000,
                        A=dict(valid=True, E=9819), B=dict(valid=g != "u1", E=9820))
    json.dump(dict(order=list(units), units=units), open(v / "s0" / "starts.json", "w"))
    monkeypatch.setattr(Q, "V85", v)
    return v


def test_queue_istage_and_g1(tmp_path, monkeypatch):
    v = _fake_v85(tmp_path, monkeypatch)
    js = Q.jobs("istage", U=3)                                # u2 is dropped (|K_g| < 2): u0, u1, space_invaders
    names = [j.name for j in js]
    assert {Q._parse(n)[0] for n in names} == {"u0", "u1", "space_invaders"}
    assert sorted(Q._parse(n)[3] for n in names if n.startswith("u0_A_v2s")) == [9806, 9807, 9808]
    assert "u1_B_v2s_9809" not in names and "u0_B_v2s_9811" in names and "u0_A_d_9819" in names
    assert names[0].startswith("space_invaders")                  # longest first (largest reference gap)
    assert Q._parse("space_invaders_B_kmac_exact_9818") == ("space_invaders", "B", "kmac_exact", 9818)
    json.dump(dict(units=["u0", "u1", "space_invaders"], n=4, K_gs={"u0": {"A": [2, 3], "B": [2]}}),
              open(v / "f1_thresholds.json", "w"))
    js = Q.jobs("g1")
    names = [j.name for j in js]
    assert "u0_A_o4_9812" in names and "u0_A_o4_9813" in names and "u0_A_o4_9814" not in names
    assert not any(n.startswith("u0_B_") for n in names)          # |K_{u0,B}| < 2: dropped
    assert "u1_A_kmac_9812" not in names and "u1_A_kmap_9813" in names   # u1's B start invalid: no K*mac
    assert "space_invaders_B_kmap_9816" in names
    tiers = [j.tier for j in js]
    assert tiers == sorted(tiers) and {Q._parse(j.name)[2] for j in js if j.tier} == {"o4_dice", "kmac_exact"}
    assert "u1_A_kmac_exact_9818" in names and "u1_B_kmac_exact_9818" not in names
    dice = sorted(j.name for j in js if "o4_dice" in j.name)
    assert dice == ["space_invaders_B_o4_dice_9818", "u0_A_o4_dice_9818"]   # positions 3 (odd) and 0 (even); u1: B invalid
    (v / "g1").mkdir()
    json.dump(dict(rows={"O_4": "ROOM-K", "K*map": "NO-ROOM-K", "K*mac": "INCONCLUSIVE-K"}), open(v / "g1" / "g1_verdict.json", "w"))
    p = Q.jobs("g1p")
    assert p and all(Q._parse(j.name)[2] == "gf4" for j in p) and "u0_A_gf4_9812" in [j.name for j in p]


def test_queue_smoke_list():
    js = Q.jobs("smoke")
    assert js and all(Q._parse(j.name)[3] == 9800 and "--smoke" in j.args for j in js)
    assert {Q._parse(j.name)[2] for j in js} >= {"v2s", "o4", "kmap"}


def test_queue_gate_claims_and_terminate_watch(tmp_path):
    now = 10_000.0
    assert Q.gate_ok({}, now, 15.0) and not Q.gate_ok({}, now, 14.0)
    assert not Q.gate_ok(dict(last_start=now - 100), now, 20) and Q.gate_ok(dict(last_start=now - 121), now, 20)
    assert not Q.gate_ok(dict(last_terminate=now - 24 * 60), now, 20) and Q.gate_ok(dict(last_terminate=now - 1501), now, 20)
    j = Q.Job("g_A_v2s_9806", tmp_path / "o.json", [], 1.0)
    assert Q.claim(tmp_path, "istage", j) and Q.claimed(tmp_path, "istage", j) and not Q.claim(tmp_path, "istage", j)
    Q._claim_path(tmp_path, "istage", j).write_text("999999")          # a dead worker's claim is taken over
    assert not Q.claimed(tmp_path, "istage", j) and Q.claim(tmp_path, "istage", j)
    Q.release(tmp_path, "istage", j)
    assert not Q.claimed(tmp_path, "istage", j)
    log = tmp_path / "sup.log"
    log.write_text("01:00:00 commit 11 GB < 12: TERMINATE 1 old\n")
    w = Q.SupWatch([log])
    assert not w.poll()
    with open(log, "a") as f:
        f.write("02:00:00 avail 11 GB < 12: SUSPEND 2\n")
    assert not w.poll()
    with open(log, "a") as f:
        f.write("03:00:00 commit 11 GB < 12: TERMINATE 3 python scripts/v85_job.py\n")
    assert w.poll() and not w.poll()
    Q.mark_terminate(tmp_path)
    assert time.time() - Q._state(tmp_path)["last_terminate"] < 5 and not (tmp_path / "gate.lock").exists()
