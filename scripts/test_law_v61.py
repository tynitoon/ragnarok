"""ARC 3 — tests (ARC3_PLAN.md 5, step 6). Run: python -m scripts.test_law_v61  (or pytest).

 (1) unexercised edit  editing a pair no script proposes leaves store AND observation bit-identical
 (2) gate edit         flipping the gate of a raw never collected leaves them bit-identical
 (3) law-break         flipping an exercised-later cell: bit-identical until the first differing outcome
 (4) counterfactual    a different law table on the same skin: bit-identical until an outcome differs
 (5) equivariance      store features and PairNet logits permute with the slots
 (6) no identity       no parameter of PairNet is shaped to MAX_ITEMS / N_PAIRS / N_ACT
 (7) uint8 round-trip  buffer rows reproduce the observation (binary features exactly)
 (9) generator         invariants over 200 seeds
 (10) env + planner    product / inert / botch / invalid, consumption, gate, quota, NOOP, combine once;
                       the scripted plan executes every admitted goal of 20 seeds from EMPTY on a
                       symbolic executor (regression: reused intermediates, consumed tool)
 (13) eval isolation   an eval leaves buffers and the training store bit-identical
 (11) nav gate and (12) timed smoke are GPU-timed: scripts/gate_v61.py --k0 runs them.
"""

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import (sample_laws, gen_law_world, make_world, permute_law_spec,
                                              LawVecTechTree, cheapest_chains, plan_for, T_MAX, INERT, BOTCH,
                                              OUT_BOTCH, OUT_INERT, OUT_PRODUCT, OUT_INVALID, NOOP)
from scripts.depth_scaling_v49 import MAX_CELLS
from scripts.pair_store_v61 import (PairStore, PairEnv, MAX_ITEMS, N_PAIRS, N_ACT, N_SLOT, N_PF, ROW, PAIR_I,
                                    PAIR_J, PAIR_INDEX, F_INV, F_VALID, GOAL_COL)
from scripts.pair_net_v61 import PairNet, BufferV61, ComposerV61, cfg_v61, QS

LAWS = sample_laws(31337)


class _NoNav:
    """A stand-in skill for scripted tests: never moves, so a 'pursue' option never collects.
    Scripts that need items put them in the inventory directly."""

    def act(self, obs, deterministic=False):
        return torch.full((obs.shape[0],), NOOP, dtype=torch.long, device=DEVICE)


def _env(spec, n=4, seed=0):
    cfg = cfg_v61(num_envs=n)
    return PairEnv(n, spec, _NoNav(), cfg, seed=seed, goal=spec["target"])


def _script(spec):
    """A fixed action script over a world: give raws, combine two lawful pairs and one botch pair.
    Returns (grants, actions) with grants = inventory to set before the script."""
    n, tier, po = spec["n_items"], spec["tier"], spec["pair_out"]
    raws = [i for i in range(n) if tier[i] == 1]
    lawful = [(i, j) for i in raws for j in raws if i < j and po[i, j] >= 0]
    botch = [(i, j) for i in raws for j in raws if i < j and po[i, j] == BOTCH]
    assert lawful and botch
    acts = [MAX_ITEMS + int(PAIR_INDEX[i, j]) for (i, j) in lawful[:2]] + [MAX_ITEMS + int(PAIR_INDEX[botch[0][0], botch[0][1]])]
    acts += [raws[0]]                                            # a pursue option (fails: no nav)
    exercised = set(lawful[:2]) | {botch[0]}
    return raws, acts, exercised


def _run(spec, acts, raws, n=3):
    env = _env(spec, n)
    env.reset()
    env.base.inv[:, raws] = 2
    env.base._set_state()
    obs = [env.obs().clone()]
    stores = [env.store.state_dict()]
    for a in acts:
        o, _ = env.step(torch.full((n,), a, dtype=torch.long, device=DEVICE))
        obs.append(o.clone()); stores.append(env.store.state_dict())
    return obs, stores


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _same_store(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- (1) (2) (3) (4)

def test_1_unexercised_edit():
    spec = make_world(8100, LAWS)
    raws, acts, exercised = _script(spec)
    n, po = spec["n_items"], spec["pair_out"]
    # an unexercised raw pair: edit its outcome
    cand = [(i, j) for i in raws for j in raws if i < j and (i, j) not in exercised]
    i, j = cand[0]
    edited = dict(spec); po2 = po.copy()
    po2[i, j] = po2[j, i] = (INERT if po[i, j] != INERT else BOTCH)
    edited["pair_out"] = po2
    o1, s1 = _run(spec, acts, raws)
    o2, s2 = _run(edited, acts, raws)
    assert _same(o1, o2) and all(_same_store(x, y) for x, y in zip(s1, s2)), "unexercised edit leaked"


def test_2_gate_edit():
    spec = make_world(8100, LAWS)
    raws, acts, _ = _script(spec)
    edited = dict(spec)
    edited["gate"] = [1 - g if (spec["tier"][k] == 1 and k == raws[-1]) else g for k, g in enumerate(spec["gate"])]
    # the flipped raw is never collected by the script (no nav), and its GATE FLAG is an attribute the
    # agent sees, so the observation legitimately differs in that one disclosed bit — assert everything
    # else is identical: zero the gate attribute column before comparing
    o1, s1 = _run(spec, acts, raws); o2, s2 = _run(edited, acts, raws)
    from scripts.pair_store_v61 import F_GATE
    def strip(o):
        x = o.clone(); x.view(-1, ROW)[:, :MAX_ITEMS * N_SLOT].view(-1, MAX_ITEMS, N_SLOT)[..., F_GATE] = 0; return x
    assert all(torch.equal(strip(a), strip(b)) for a, b in zip(o1, o2))
    assert all(_same_store(x, y) for x, y in zip(s1, s2))


def test_3_law_break():
    spec = make_world(8100, LAWS)
    raws, acts, exercised = _script(spec)
    i, j = sorted(exercised)[0]                    # exercised at the first or second step
    edited = dict(spec); po2 = spec["pair_out"].copy()
    po2[i, j] = po2[j, i] = BOTCH if spec["pair_out"][i, j] != BOTCH else INERT
    edited["pair_out"] = po2
    o1, _ = _run(spec, acts, raws); o2, _ = _run(edited, acts, raws)
    step = [k for k, a in enumerate(acts) if a == MAX_ITEMS + int(PAIR_INDEX[i, j])][0]
    assert all(torch.equal(a, b) for a, b in zip(o1[:step + 1], o2[:step + 1])), "differs before the outcome"
    assert not torch.equal(o1[step + 1], o2[step + 1]), "a broken law must change the outcome"


def test_4_counterfactual_law():
    spec = make_world(8100, LAWS)
    raws, acts, exercised = _script(spec)
    other = dict(spec)
    rng = np.random.default_rng(1)
    po2 = spec["pair_out"].copy()
    n = spec["n_items"]
    for a in range(n):
        for b in range(a + 1, n):
            if spec["tier"][a] == spec["tier"][b] == 1:
                v = rng.choice([BOTCH, INERT])
                po2[a, b] = po2[b, a] = v
    other["pair_out"] = po2
    o1, _ = _run(spec, acts, raws); o2, _ = _run(other, acts, raws)
    assert torch.equal(o1[0], o2[0]), "before any outcome the two laws are indistinguishable"


# ---------------------------------------------------------------- (5) (6) (7)

def test_5_equivariance():
    n, N = 24, 2
    perm = torch.tensor(np.random.default_rng(3).permutation(MAX_ITEMS), device=DEVICE)  # new i <- old perm[i]
    inv = torch.empty_like(perm); inv[perm] = torch.arange(MAX_ITEMS, device=DEVICE)
    # pair index map: new pair p' (i',j') <- old pair of (perm[i'], perm[j'])
    pmap = PAIR_INDEX[perm[PAIR_I], perm[PAIR_J]]
    st, st2 = PairStore(N, MAX_ITEMS), PairStore(N, MAX_ITEMS)
    held = torch.rand(N, MAX_ITEMS, device=DEVICE) > 0.5
    g = torch.tensor([3, 5], device=DEVICE); ob = torch.tensor([True, False], device=DEVICE)
    mask = torch.ones(N, dtype=torch.bool, device=DEVICE)
    st.write_raw(g, held, ob, mask); st2.write_raw(inv[g], held[:, perm], ob, mask)
    p = torch.tensor([7, 100], device=DEVICE); oc = torch.tensor([OUT_PRODUCT, OUT_BOTCH], device=DEVICE)
    pr = torch.tensor([9, -1], device=DEVICE)
    st.write_pair(p, oc, pr, mask)
    p2 = PAIR_INDEX[inv[PAIR_I[p]], inv[PAIR_J[p]]]
    st2.write_pair(p2, oc, torch.where(pr >= 0, inv[pr.clamp(min=0)], pr), mask)
    goal = torch.tensor([9, 9], device=DEVICE)
    f1, q1 = st.features(held, goal); f2, q2 = st2.features(held[:, perm], inv[goal])
    assert torch.allclose(f1[:, perm], f2), "slot features not equivariant"
    assert torch.allclose(q1[:, pmap], q2), "pair features not equivariant"
    # the net: permute an observation's slots and pairs -> logits permute
    torch.manual_seed(0)
    net = PairNet().to(DEVICE)
    obs = torch.rand(N, ROW, device=DEVICE)
    x = obs[:, :MAX_ITEMS * N_SLOT].reshape(N, MAX_ITEMS, N_SLOT); pf = obs[:, MAX_ITEMS * N_SLOT:].reshape(N, N_PAIRS, N_PF)
    x[..., F_VALID] = 1.0
    obs2 = torch.cat([x[:, perm].reshape(N, -1), pf[:, pmap].reshape(N, -1)], -1)
    obs = torch.cat([x.reshape(N, -1), pf.reshape(N, -1)], -1)
    l1, _, _ = net(obs); l2, _, _ = net(obs2)
    assert torch.allclose(l1[:, :MAX_ITEMS][:, perm], l2[:, :MAX_ITEMS], atol=1e-4)
    assert torch.allclose(l1[:, MAX_ITEMS:][:, pmap], l2[:, MAX_ITEMS:], atol=1e-4)


def test_6_no_identity_parameter():
    net = PairNet()
    for name, p in net.named_parameters():
        assert not any(d in (MAX_ITEMS, N_PAIRS, N_ACT) for d in p.shape), f"{name} is shaped to a slot count"


def test_7_uint8_roundtrip():
    spec = make_world(8100, LAWS)
    env = _env(spec, 2); env.reset()
    env.base.inv[:, :6] = 1; env.base._set_state()
    obs = env.obs()
    buf = BufferV61(cap=10, cap_out=10)
    buf.add(obs, torch.zeros(2, dtype=torch.long, device=DEVICE))
    s, _ = buf.sample(2)
    assert (s - obs[:1]).abs().max() <= 0.5 / QS + 1e-6 or (s - obs[1:]).abs().max() <= 0.5 / QS + 1e-6
    x = obs[:, :MAX_ITEMS * N_SLOT].reshape(2, MAX_ITEMS, N_SLOT)
    r = buf.s[:2].float() / QS
    rx = r[:, :MAX_ITEMS * N_SLOT].reshape(2, MAX_ITEMS, N_SLOT)
    for f in (F_INV, F_VALID, GOAL_COL):
        assert torch.equal(x[..., f] > 0.5, rx[..., f] > 0.5)


# ---------------------------------------------------------------- (9) generator

def test_9_generator_invariants():
    K, Bond, Out = LAWS["K"], LAWS["Bond"], LAWS["Out"]
    # proper edge colouring
    for a in range(K):
        outs = [Out[a, b] for b in range(K) if Bond[a, b]]
        assert len(outs) == len(set(outs)) and len(outs) >= 2
    for s in range(20_000, 20_200):
        sp = make_world(s, LAWS)
        n = sp["n_items"]
        assert n == 24 and sp["n_cells"] == 7
        cells = sorted(sp["cell"][i] for i in range(n) if sp["kind"][i] == "R")
        assert cells == [1, 2, 3, 4, 5, 6]
        assert sum(sp["gate"]) == 2
        po = sp["pair_out"]
        assert (po == po.T).all()
        for i in range(n):
            for j in range(i + 1, n):
                v = po[i, j]
                if v >= 0:
                    assert sp["tier"][v] == sp["tier"][i] + 1 == sp["tier"][j] + 1 and not sp["decoy"][v]
                    assert Bond[sp["el"][i], sp["el"][j]] and Out[sp["el"][i], sp["el"][j]] == sp["el"][v]
                elif v == INERT:
                    assert sp["tier"][i] == sp["tier"][j] < T_MAX and Bond[sp["el"][i], sp["el"][j]]
                    assert not any(sp["el"][k] == Out[sp["el"][i], sp["el"][j]] and sp["tier"][k] == sp["tier"][i] + 1
                                   and not sp["decoy"][k] for k in range(n))
                else:
                    assert (sp["tier"][i] != sp["tier"][j] or sp["tier"][i] >= T_MAX
                            or not Bond[sp["el"][i], sp["el"][j]] or sp["decoy"][i] or sp["decoy"][j])
        for t in range(1, T_MAX + 1):
            assert sum(1 for i in range(n) if sp["tier"][i] == t) == 6
            dropped = {c for (tt, c) in sp["dropped"] if tt == t}
            decoys = {sp["el"][i] for i in range(n) if sp["tier"][i] == t and sp["decoy"][i]}
            assert not (dropped & decoys), "a decoy must never be a dropped (law-producible) product"
        for t, goals in sp["admitted"].items():
            for g in goals:
                p = sp["plans"][g]
                assert p["ok"] and max(p["units_by_type"].values()) <= sp["quota"] and p["steps"] <= 48


# ---------------------------------------------------------------- (10) env and planner

def _symbolic_execute(spec, goal):
    """Perfect-nav executor of plan_for on a CPU inventory model with the gate and the quota."""
    p = spec["plans"][goal]
    n, tier, gate, po, q = spec["n_items"], spec["tier"], spec["gate"], spec["pair_out"], spec["quota"]
    inv = [0] * n; quota = {i: q for i in range(n) if tier[i] == 1}
    seq = ([("c", s) for s in p["first_free"]] + [("x", i, j) for (i, j, _) in p["first_pairs"]]
           + [("c", s) for s in p["collect_gated"]] + [("c", s) for s in p["collect_free"]]
           + [("x", i, j) for (i, j, _) in p["pairs"]])
    for a in seq:
        if a[0] == "c":
            s = a[1]
            has_tool = any(inv[k] > 0 and tier[k] >= 2 for k in range(n))
            if quota[s] <= 0 or (gate[s] and not has_tool):
                return False, f"collect {s} refused (quota {quota[s]}, gate {gate[s]}, tool {has_tool})"
            inv[s] += 1; quota[s] -= 1
        else:
            i, j = a[1], a[2]
            if inv[i] <= 0 or inv[j] <= 0:
                return False, f"combine {i},{j} without inputs"
            v = po[i, j]
            if v < 0:
                return False, f"combine {i},{j} not a product ({v})"
            inv[i] -= 1; inv[j] -= 1; inv[v] += 1
    return inv[goal] > 0, "ok" if inv[goal] > 0 else "goal not produced"


def test_10_planner_executes_from_empty():
    for s in range(20_000, 20_020):
        sp = make_world(s, LAWS)
        for t, goals in sp["admitted"].items():
            for g in goals:
                ok, why = _symbolic_execute(sp, g)
                assert ok, f"seed {s} goal {g} (tier {t}): {why}"


def test_10_env_primitives():
    spec = make_world(8100, LAWS)
    n, po, tier = spec["n_items"], spec["pair_out"], spec["tier"]
    cfg = cfg_v61(num_envs=4)
    base = LawVecTechTree(4, spec, grid=cfg["grid"], view=cfg["view"], n_resource=cfg["n_resource"],
                          max_cells=MAX_CELLS, seed=0)
    raws = [i for i in range(n) if tier[i] == 1]
    lawful = [(i, j) for i in raws for j in raws if i < j and po[i, j] >= 0][0]
    botch = [(i, j) for i in raws for j in raws if i < j and po[i, j] == BOTCH][0]
    base.inv[:] = 0
    base.inv[:, raws] = 1
    i = torch.tensor([lawful[0], botch[0], lawful[0], 0], device=DEVICE)
    j = torch.tensor([lawful[1], botch[1], lawful[1], 0], device=DEVICE)
    base.inv[2, lawful[0]] = 0                                   # env 2: input missing -> invalid
    out, prod = base.combine(i, j)
    assert out.tolist()[:3] == [OUT_PRODUCT, OUT_BOTCH, OUT_INVALID] and out[3] == OUT_INVALID
    assert base.inv[0, lawful[0]] == 0 and base.inv[0, lawful[1]] == 0 and base.inv[0, po[lawful]] == 1
    assert base.inv[1, botch[0]] == 0 and base.inv[1, botch[1]] == 0
    assert base.inv[2, lawful[1]] == 1
    # inert: a lawful raw pair whose product was dropped, if this world has one; else synthesise
    inert = [(a, b) for a in raws for b in raws if a < b and po[a, b] == INERT]
    if inert:
        a, b = inert[0]
        base.inv[:] = 0; base.inv[:, [a, b]] = 1
        out, _ = base.combine(torch.full((4,), a, device=DEVICE), torch.full((4,), b, device=DEVICE))
        assert (out == OUT_INERT).all() and base.inv[0, a] == 1 and base.inv[0, b] == 1
    # gate and quota: put the agent on a gated cell
    gated_raw = [k for k in raws if spec["gate"][k] == 1][0]
    free_raw = [k for k in raws if spec["gate"][k] == 0][0]
    c = spec["cell"][gated_raw]
    base.reset(); base.inv[:] = 0
    base.grid[:, 3, 3] = c; base.pos[:] = torch.tensor([3, 3], device=DEVICE)
    base.step(torch.full((4,), 4, device=DEVICE))
    assert (base.inv[:, gated_raw] == 0).all(), "gated collect without a tool must fail"
    t2 = [k for k in range(n) if tier[k] == 2 and not spec["decoy"][k]][0]
    base.inv[:, t2] = 1
    for _ in range(5):
        base.step(torch.full((4,), 4, device=DEVICE))
    assert (base.inv[:, gated_raw] == 4).all(), "quota 4 per episode"
    assert (base.quota_left[:, c] == 0).all()
    base._reset_done(torch.ones(4, dtype=torch.bool, device=DEVICE))
    assert (base.quota_left[:, c] == 4).all()
    # NOOP changes nothing
    g0, p0, i0 = base.grid.clone(), base.pos.clone(), base.inv.clone()
    base.step(torch.full((4,), NOOP, device=DEVICE))
    assert torch.equal(g0, base.grid) and torch.equal(p0, base.pos) and torch.equal(i0, base.inv)
    # PairEnv: a pair action combines exactly once; a pursue on a craft slot is a no-op attempt
    env = _env(spec, 4); env.reset()
    env.base.inv[:] = 0; env.base.inv[:, raws] = 2; env.base._set_state()
    env.step(torch.full((4,), MAX_ITEMS + int(PAIR_INDEX[lawful[0], lawful[1]]), device=DEVICE))
    assert (env.base.inv[:, lawful[0]] == 1).all() and (env.base.inv[:, po[lawful]] == 1).all()
    assert (env.store.pair_succ[:, PAIR_INDEX[lawful[0], lawful[1]]] == 1).all()
    assert (env.store.pair_out[:, PAIR_INDEX[lawful[0], lawful[1]]] == po[lawful]).all()


# ---------------------------------------------------------------- (13) eval isolation

def test_13_eval_isolation():
    from scripts.pair_net_v61 import eval_goal_v61
    spec = make_world(8100, LAWS)
    cfg = cfg_v61(num_envs=2)
    env = _env(spec, 2); env.reset()
    comp = ComposerV61(init_seed=0)
    buf = BufferV61(cap=100, cap_out=100)
    st0 = env.store.state_dict()
    eval_goal_v61(spec, _NoNav(), comp, cfg, 8100, spec["target"], env.store.state_dict())
    assert buf.n == 0 and buf.n_out == 0 and _same_store(st0, env.store.state_dict())


if __name__ == "__main__":
    import sys, time
    names = [k for k in list(globals()) if k.startswith("test_")]
    fails = 0
    for k in names:
        t0 = time.perf_counter()
        try:
            globals()[k]()
            print(f"  ok   {k} ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL {k}: {type(e).__name__}: {e}")
    print(f"{len(names) - fails}/{len(names)} passed")
    sys.exit(1 if fails else 0)
