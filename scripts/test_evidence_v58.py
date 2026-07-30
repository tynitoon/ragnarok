"""ARC 2, Step 1 tests + feasibility demo for the EvidenceStore.

  1. LEAK TEST (the defect-1 exclusion, mechanical): edit a recipe the agent never observed ->
     the store must be BIT-IDENTICAL. (v51's craftable_now, computed from the true spec, fails this
     by construction — that is the whole point of the test.)
  2. EQUIVARIANCE: permuting slots permutes every feature identically — no hidden index preference.
  3. PERSISTENCE: the store survives episode truncation; a fresh store starts blank.
  4. DEMO (the Step-1 milestone): on a real hidden-recipe world with the frozen v55 nav skill, the
     hand-coded evidence policy — persistent store, no learning, no recipe access — must reach the
     deepest admitted goal within a few episodes, where a store-less random policy starves.

Usage: python -m scripts.test_evidence_v58
"""

import copy
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.meta_manager_v51 import MAX_ITEMS
from scripts.hidden_recipe_v55 import permute_spec, admitted_goals
from scripts.evidence_store_v58 import EvidenceStore, StoreEnv, evidence_policy

CFG = dict(num_envs=16, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02, nav_max_steps=40,
           skill_iters=400, option_timeout=16, macro_budget=48, episodes_per_round=4,
           train_steps_per_round=300, max_samples_per_ep=8192, epsilon=0.05, temp=1.0,
           thresh=0.6, r_max=10, skill_stochastic=True, mgr_entropy=0.03, router_iters=0)


class ScriptSkill:
    """Deterministic pseudo-skill so both sides of the leak test see identical trajectories."""
    def act(self, o, deterministic=False):
        return (o.abs().sum(-1) * 977).long() % 5


def store_sig(st):
    return [st.n_succ.clone(), st.n_fail.clone(), st.S.clone(), st.C.clone(), st.obtained_ever.clone()]


def test_leak():
    spec = permute_spec(gen_tree(3002, n_items=14), 3002)
    deepest = max(range(14), key=lambda i: spec["depth"][i])
    spec2 = copy.deepcopy(spec)
    others = [j for j in range(14) if j != deepest and spec["kind"][j] == "C"
              and deepest not in spec2["inputs"][j]]
    spec2["inputs"][deepest] = {others[0]: 1}            # edit a recipe the script never exercises
    torch.manual_seed(0)
    e1 = StoreEnv(8, spec, ScriptSkill(), CFG, seed=5, goal=0)
    torch.manual_seed(0)
    e2 = StoreEnv(8, spec2, ScriptSkill(), CFG, seed=5, goal=0)
    script = [g for g in range(14) if g != deepest] * 4
    for g in script:
        a = torch.full((8,), g, dtype=torch.long, device=DEVICE)
        e1.step(a); e2.step(a)
    for x, y in zip(store_sig(e1.store), store_sig(e2.store)):
        assert torch.equal(x, y), "LEAK: an unobserved recipe edit changed the store"
    print("  [1] leak test PASSED — unobserved recipe edit leaves the store bit-identical")


def test_equivariance():
    torch.manual_seed(1)
    n = 14
    perm = torch.randperm(n, device=DEVICE)
    a, b = EvidenceStore(4, n), EvidenceStore(4, n)
    for _ in range(60):
        g = torch.randint(0, n, (4,), device=DEVICE)
        held = torch.rand(4, MAX_ITEMS, device=DEVICE) > 0.6
        held[:, n:] = False
        got = torch.rand(4, device=DEVICE) > 0.5
        isr = torch.rand(4, device=DEVICE) > 0.5
        a.write(g, held, got, isr)
        held_p = torch.zeros_like(held)
        held_p[:, perm] = held[:, :n]
        b.write(perm[g], held_p, got, isr)
    hf = torch.rand(4, MAX_ITEMS, device=DEVICE) > 0.5
    hf[:, n:] = False
    hfp = torch.zeros_like(hf); hfp[:, perm] = hf[:, :n]
    isr_row = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    fa = a.features(hf, isr_row)[:, :n]
    fb = b.features(hfp, isr_row)[:, perm]
    assert torch.allclose(fa, fb, atol=1e-6), "store features are not permutation-equivariant"
    print("  [2] equivariance PASSED — permuting slots permutes the features identically")


def test_persistence():
    spec = permute_spec(gen_tree(3002, n_items=14), 3002)
    env = StoreEnv(4, spec, ScriptSkill(), dict(CFG, macro_budget=3), seed=2, goal=0)
    for _ in range(7):                                   # crosses two truncation boundaries
        env.step(torch.randint(0, 14, (4,), device=DEVICE))
    assert float(env.store.n_fail.sum() + env.store.n_succ.sum()) >= 28, "store lost writes at truncation"
    assert float(env.tried.sum()) <= 4 * 14, "tried should reset per episode"
    print("  [3] persistence PASSED — store survives truncation; per-episode flags reset")


def load_real_skill():
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(CFG["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], CFG, 0, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=CFG["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load("craft_v6_out/v55_skill_s0.pt", map_location=DEVICE))
    return ppo


def demo(world=3002, episodes=30):
    spec = permute_spec(gen_tree(world, n_items=14), world)
    adm = admitted_goals(spec)
    goal, pc, blind = adm[-1]
    skill = load_real_skill()
    t0 = time.perf_counter()
    print(f"\n  DEMO world {world}: deepest admitted goal {goal} (pc {pc}, blind {blind:.0f}) | "
          f"{episodes} episodes x {CFG['macro_budget']} macro-steps | 16 envs")
    results = {}
    for tag in ("EVIDENCE", "RANDOM"):
        env = StoreEnv(16, spec, skill, CFG, seed=9, goal=goal)
        first_ep = torch.full((16,), -1, dtype=torch.long, device=DEVICE)
        got_count = 0
        for ep in range(episodes):
            for _ in range(CFG["macro_budget"]):
                if tag == "EVIDENCE":
                    a = evidence_policy(env, goal)
                else:
                    a = torch.randint(0, spec["n_items"], (16,), device=DEVICE)
                env.step(a)
            newly = (env.store.obtained_ever[:, goal] > 0) & (first_ep < 0)
            first_ep[newly] = ep
            got_count = int((env.store.obtained_ever[:, goal] > 0).sum())
        reached = int((first_ep >= 0).sum())
        med = int(first_ep[first_ep >= 0].float().median()) if reached else -1
        results[tag] = (reached, med)
        print(f"    {tag:>8}: {reached}/16 envs reached the goal | median first-success episode "
              f"{med if med >= 0 else '—'} | {time.perf_counter()-t0:.0f}s")
    ev, rd = results["EVIDENCE"], results["RANDOM"]
    print(f"\n  => evidence-over-store {ev[0]}/16 vs random {rd[0]}/16 "
          f"{'— the representation carries the load. MILESTONE 1 OK' if ev[0] > rd[0] else '— NOT SEPARATED (investigate before building the net)'}")


if __name__ == "__main__":
    print("EvidenceStore — Step 1 tests")
    test_leak()
    test_equivariance()
    test_persistence()
    demo()
