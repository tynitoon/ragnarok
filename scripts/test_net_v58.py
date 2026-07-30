"""ARC 2, Step 2 tests — the EvidenceNet and the widened pipeline.

  [1] NET EQUIVARIANCE   permuting slots permutes the logits identically (no hidden index preference —
                         the structural guarantee that replaces identity embeddings).
  [2] NET-INPUT LEAK     editing a recipe the agent never observed leaves the NET'S INPUT bit-identical
                         (ARC2_PLAN check 2: the leak test now covers the observation, not just the store).
  [3] QUANTISATION       uint8 write/read round-trips: binary base feats EXACTLY, evidence within 1/255.
  [4] NO EMBEDDINGS      the policy contains no nn.Embedding and no parameter sized to MAX_ITEMS.
  [5] CELL PERMUTATION   measures the nav gate with and without permute_spec_v58 on the real frozen
                         skill. This is the evidence for the documented deviation from ARC2_PLAN
                         section 3 (permuting into 1..22 would feed the skill one-hots it never saw:
                         it trained on cell IDs 1..9 only).

Usage: python -m scripts.test_net_v58
"""

import copy

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.meta_manager_v51 import MAX_ITEMS, N_FEAT
from scripts.hidden_recipe_v55 import permute_spec, nav_gate
from scripts.evidence_store_v58 import N_EVID
from scripts.evidence_net_v58 import (EvidenceNet, ComposerV58, BufferV58, StoreEnvV58,
                                      permute_spec_v58, cfg_v58, N_TOTAL, QS)

CFG = cfg_v58(num_envs=8)


class ScriptSkill:
    def act(self, o, deterministic=False):
        return (o.abs().sum(-1) * 977).long() % 5


def test_equivariance():
    torch.manual_seed(0)
    net = EvidenceNet().to(DEVICE).eval()
    n = 14
    x = torch.rand(6, MAX_ITEMS, N_TOTAL, device=DEVICE)
    x[:, n:, :] = 0.0
    x[:, :n, 6] = 1.0                                    # is_valid
    perm = torch.randperm(n, device=DEVICE)
    xp = x.clone()
    xp[:, :n] = x[:, perm]
    with torch.no_grad():
        l1, _ = net(x.reshape(6, -1))
        l2, _ = net(xp.reshape(6, -1))
    assert torch.allclose(l1[:, :n][:, perm], l2[:, :n], atol=1e-5), "net is not permutation-equivariant"
    print("  [1] net equivariance PASSED — permuting slots permutes the logits identically")


def test_input_leak():
    spec = permute_spec_v58(gen_tree(3002, n_items=14), 3002)
    deepest = max(range(14), key=lambda i: spec["depth"][i])
    spec2 = copy.deepcopy(spec)
    others = [j for j in range(14) if j != deepest and spec["kind"][j] == "C"
              and deepest not in spec2["inputs"][j]]
    spec2["inputs"][deepest] = {others[0]: 1}
    torch.manual_seed(0); e1 = StoreEnvV58(8, spec, ScriptSkill(), CFG, seed=5, goal=0)
    torch.manual_seed(0); e2 = StoreEnvV58(8, spec2, ScriptSkill(), CFG, seed=5, goal=0)
    for g in [j for j in range(14) if j != deepest] * 3:
        a = torch.full((8,), g, dtype=torch.long, device=DEVICE)
        e1.step(a); e2.step(a)
        assert torch.equal(e1.obs(), e2.obs()), "LEAK: unobserved recipe edit changed the net's INPUT"
    print("  [2] net-input leak PASSED — unobserved recipe edit leaves the observation bit-identical")


def test_quantisation():
    buf = BufferV58(cap=4096)
    torch.manual_seed(3)
    x = torch.rand(64, MAX_ITEMS, N_TOTAL, device=DEVICE)
    x[..., :N_FEAT] = (x[..., :N_FEAT] > 0.5).float()    # base half is binary
    flat = x.reshape(64, -1)
    buf.add(flat, torch.zeros(64, dtype=torch.long, device=DEVICE))
    back = (buf.s[:64].float() / QS).reshape(64, MAX_ITEMS, N_TOTAL)
    assert torch.equal(back[..., :N_FEAT], x[..., :N_FEAT]), "binary base features do not round-trip exactly"
    err = (back[..., N_FEAT:] - x[..., N_FEAT:]).abs().max().item()
    assert err <= 1.0 / QS + 1e-6, f"evidence quantisation error {err:.5f} exceeds 1/255"
    print(f"  [3] quantisation PASSED — base feats exact, evidence max error {err:.5f} (<= 1/255)")


def test_no_embeddings():
    net = EvidenceNet()
    assert not any(isinstance(m, torch.nn.Embedding) for m in net.modules()), "policy has an nn.Embedding"
    bad = [n for n, p in net.named_parameters() if MAX_ITEMS in tuple(p.shape)]
    assert not bad, f"policy has item-indexed parameters: {bad}"
    print(f"  [4] no embeddings PASSED — identity-free policy, "
          f"{sum(p.numel() for p in net.parameters()):,} params, none sized to MAX_ITEMS")


def load_skill(seed=0):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(CFG["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], CFG, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=CFG["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(f"craft_v6_out/v55_skill_s{seed}.pt", map_location=DEVICE))
    return ppo


def test_cell_permutation():
    """The deviation evidence: does permuting cell IDs within the used set disturb the frozen skill?"""
    skill = load_skill(0)
    cfg = cfg_v58(num_envs=256)
    print("  [5] cell-permutation nav check (frozen skill trained on cell IDs 1..9 only):")
    worst = []
    for w in (4000, 5000, 6000):
        raw = gen_tree(w, n_items=14)
        a = nav_gate(skill, permute_spec(raw, w), cfg, 0)
        b = nav_gate(skill, permute_spec_v58(raw, w), cfg, 0)
        worst.append((min(a.values()), min(b.values())))
        print(f"        world {w}: min nav v55-permute {min(a.values()):.3f} -> "
              f"v58-permute {min(b.values()):.3f}")
    drop = max(x - y for x, y in worst)
    assert drop < 0.10, f"cell permutation degraded nav by {drop:.3f} — skill pushed out of distribution"
    assert all(y >= 0.85 for _, y in worst), "a world fails the 0.85 nav gate under v58 permutation"
    print(f"        max degradation {drop:+.3f}, all worlds still pass the 0.85 gate — deviation is safe")


if __name__ == "__main__":
    print("EvidenceNet — Step 2 tests")
    test_equivariance()
    test_input_leak()
    test_quantisation()
    test_no_embeddings()
    test_cell_permutation()
    print("\nALL STEP-2 TESTS PASSED")
