"""ARC 3 — law_world: a FAMILY of worlds that share hidden LAWS (ARC3_PLAN.md section 2).

The wall ARC 1 and ARC 2 hit from both sides: reuse pays only when what is shared between worlds is
EXPENSIVE to reacquire. gen_tree worlds share nothing costly. Here every world of a family obeys the same
hidden chemistry, and only the skin changes.

    LAWS (sampled once per family, never observed):
        K elements, T_MAX tiers. An item is (element, tier). Raws are tier 1.
        Bond[K,K]  symmetric bool: which element pairs react.
        Out[K,K]   the product element of a reacting pair, a PROPER EDGE COLOURING of the bond graph
                   (an element's products over its partners are all distinct).
        combine(x, y) is LAWFUL iff tier x == tier y == t < T_MAX and Bond[el x, el y]; it yields the item
        (Out[el x, el y], t+1). Outcomes: PRODUCT (inputs consumed, +1 product) / INERT (lawful but the
        product does not exist in this world: inputs returned) / BOTCH (unlawful: both inputs destroyed).
        GATE: a raw is FREE or GATED; a gated raw can be collected only while holding an item of tier >= 2.
        QUOTA: each raw type yields at most `quota` units per episode.
    SKIN (per world seed): which 6 elements are raws (cells 1..6), which products exist (fixed 6 slots per
        tier, unproducible slots filled with DECOYS), item index permutation, cell-ID permutation.

Everything in the spec that is derived from Bond/Out (pair_out, chains, admission) is EXPERIMENTER-SIDE:
execution and scoring only. The observation is built in scripts/pair_store_v61.py from attributes
(element, tier, gate flag) and the agent's own evidence — never from these tensors (test (1)-(4) of
scripts/test_law_v61.py assert it bit-for-bit).
"""

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import DeviceVecTechTree

T_MAX = 4
INERT, BOTCH = -1, -2            # pair_out codes (>= 0 is the product's item index)
OUT_BOTCH, OUT_INERT, OUT_PRODUCT, OUT_INVALID = 0, 1, 2, 3
NOOP = 5                         # base env: not a move (0-3), not collect (4), no craft actions -> no-op


# ---------------------------------------------------------------- laws

def _edge_colouring(bond, K, rng, tries=200):
    edges = [(a, b) for a in range(K) for b in range(a + 1, K) if bond[a, b]]
    for _ in range(tries):
        order = rng.permutation(len(edges))
        used = [set() for _ in range(K)]
        out = np.full((K, K), -1, dtype=int)
        ok = True
        for e in order:
            a, b = edges[e]
            avail = [c for c in range(K) if c not in used[a] and c not in used[b]]
            if not avail:
                ok = False
                break
            c = int(rng.choice(avail))
            out[a, b] = out[b, a] = c
            used[a].add(c); used[b].add(c)
        if ok:
            return out
    return None


def sample_laws(seed, K=14, density=0.5, min_partners=2):
    """The family's chemistry. Rejection-sampled until every element bonds with >= min_partners others
    and a proper edge colouring exists."""
    rng = np.random.default_rng(seed)
    for attempt in range(10_000):
        u = rng.random((K, K)) < density
        bond = np.triu(u, 1)
        bond = bond | bond.T
        if (bond.sum(1) < min_partners).any():
            continue
        out = _edge_colouring(bond, K, rng)
        if out is None:
            continue
        return dict(K=K, T_MAX=T_MAX, density=density, seed=seed, Bond=bond, Out=out,
                    n_true=int(np.triu(bond, 1).sum()), attempts=attempt + 1)
    raise RuntimeError("no law table found")


# ---------------------------------------------------------------- one world of the family

def gen_law_world(seed, laws, n_raw=6, n_free=4, slots=6, quota=4):
    """Sample the SKIN of one world under `laws`. Returns a spec dict usable by DeviceVecTechTree (kind,
    cell, tool, inputs, tools, craft_actions, n_items, n_cells, depth, target, true_pre) plus the law-world
    fields: el, tier, gate, decoy, pair_out, chains, admitted, dropped, quota, raw_el."""
    rng = np.random.default_rng(seed)
    K, Bond, Out = laws["K"], laws["Bond"], laws["Out"]
    raw_el = [int(x) for x in rng.choice(K, size=n_raw, replace=False)]
    el, tier, gate, decoy = list(raw_el), [1] * n_raw, [0] * n_free + [1] * (n_raw - n_free), [False] * n_raw
    present = {1: set(raw_el)}
    dropped = []
    for t in range(2, T_MAX + 1):
        prev = sorted(present[t - 1])
        cand = sorted({int(Out[a, b]) for a in prev for b in prev if a < b and Bond[a, b]})
        cand = [cand[i] for i in rng.permutation(len(cand))]
        real, drop = cand[:slots], cand[slots:]
        dropped += [(t, c) for c in drop]
        nondec = [c for c in range(K) if c not in set(cand)]
        n_dec = slots - len(real)
        dec = [int(x) for x in rng.choice(nondec, size=n_dec, replace=False)] if n_dec > 0 else []
        for c in real:
            el.append(c); tier.append(t); gate.append(0); decoy.append(False)
        for c in dec:
            el.append(c); tier.append(t); gate.append(0); decoy.append(True)
        present[t] = set(real)
    n = len(el)
    assert n == n_raw + (T_MAX - 1) * slots
    # ---- experimenter-side: the outcome of every pair -------------------------------------------
    pair_out = np.full((n, n), BOTCH, dtype=int)
    slot_of = {(el[i], tier[i]): i for i in range(n) if not decoy[i]}
    for i in range(n):
        for j in range(i + 1, n):
            if decoy[i] or decoy[j] or tier[i] != tier[j] or tier[i] >= T_MAX:
                continue
            if Bond[el[i], el[j]]:
                k = slot_of.get((int(Out[el[i], el[j]]), tier[i] + 1))
                pair_out[i, j] = pair_out[j, i] = (k if k is not None else INERT)
    spec = dict(
        n_items=n, n_cells=n_raw + 1,
        kind=["R" if t == 1 else "C" for t in tier],
        cell=[(i + 1) if i < n_raw else -1 for i in range(n)],
        tool=[-1] * n, inputs=[{} for _ in range(n)], tools=[[] for _ in range(n)],
        craft_actions=[],                                   # combine is the only production primitive
        depth=[t - 1 for t in tier],
        el=el, tier=tier, gate=gate, decoy=decoy, raw_el=raw_el,
        pair_out=pair_out, dropped=dropped, quota=quota, n_free=n_free,
        laws_seed=laws["seed"], seed=seed,
    )
    _annotate_chains(spec)
    spec["target"] = spec["admitted"][T_MAX][0] if spec["admitted"][T_MAX] else 0
    return spec


# ---------------------------------------------------------------- chains, the planner, admission

def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def cheapest_chains(spec):
    """For every real item: the cheapest recipe tree (fewest raw units, ties by max units of one type).
    chain[k] = dict(units={raw slot: count}, pairs=[(i, j, k), ...] bottom-up, steps=int)."""
    n, tier, decoy, po = spec["n_items"], spec["tier"], spec["decoy"], spec["pair_out"]
    chain = {}
    for i in range(n):
        if tier[i] == 1:
            chain[i] = dict(units={i: 1}, pairs=[], steps=1)
    for t in range(2, T_MAX + 1):
        for k in range(n):
            if tier[k] != t or decoy[k]:
                continue
            best = None
            for i in range(n):
                for j in range(i + 1, n):
                    if po[i, j] == k and i in chain and j in chain:
                        units = _merge(chain[i]["units"], chain[j]["units"])
                        cand = dict(units=units, pairs=chain[i]["pairs"] + chain[j]["pairs"] + [(i, j, k)],
                                    steps=chain[i]["steps"] + chain[j]["steps"] + 1)
                        key = (sum(units.values()), max(units.values()), cand["steps"])
                        if best is None or key < best[0]:
                            best = (key, cand)
            if best is not None:
                chain[k] = best[1]
    return chain


def plan_for(spec, chain, goal):
    """The scripted LAW-KNOWER's plan for `goal` (experimenter-side; ARC3_PLAN 2.7 L): gather-then-craft.
    If the chain needs a gated raw, a tier-2 TOOL buildable from FREE raws must be held while collecting
    gated units: the chain's own free-only tier-2 item if it has one, else an extra one (cheapest).
    Returns dict(collect_free=[slots...], collect_gated=[...], pairs=[(i,j,k)...], tool=None|slot,
    tool_pairs=[...], units_by_type={slot: n}, steps=int, ok=bool, why=str)."""
    tier, gate, decoy = spec["tier"], spec["gate"], spec["decoy"]
    c = chain.get(goal)
    if c is None:
        return dict(ok=False, why="unproducible", steps=10 ** 9, units_by_type={})
    units = dict(c["units"])
    gated_units = {s: k for s, k in units.items() if gate[s] == 1}
    tool, tool_pairs, extra = None, [], {}
    if gated_units:
        # a tier-2 item of the chain made from two FREE raws, produced before any gated collect
        for (i, j, k) in c["pairs"]:
            if tier[k] == 2 and gate[i] == 0 and gate[j] == 0:
                tool = k
                break
        if tool is None:
            # extra tool: the cheapest free-only tier-2 item of this world
            for k in range(spec["n_items"]):
                if tier[k] == 2 and not decoy[k] and k in chain:
                    (i, j, _), = chain[k]["pairs"][-1:]
                    if gate[i] == 0 and gate[j] == 0:
                        tool = k; tool_pairs = [(i, j, k)]; extra = _merge(extra, {i: 1, j: 1})
                        break
        if tool is None:
            return dict(ok=False, why="gated chain, no free tool", steps=10 ** 9, units_by_type=units)
    total = _merge(units, extra)
    q = spec["quota"]
    ok = max(total.values()) <= q
    # order: tool's free units -> tool combine -> gated units -> remaining free units -> chain combines
    if tool is not None and not tool_pairs:
        # the chain's own tool: pull its FIRST production (two free units + one combine) to the front.
        # Only that one production moves — an intermediate reused later in the chain is produced again
        # where the chain needs it (the "reused intermediate" defect the planner test guards).
        idx = [k for k, p in enumerate(c["pairs"]) if p[2] == tool][0]
        ti, tj, _ = c["pairs"][idx]
        first_free = [ti, tj]
        first_pairs = [(ti, tj, tool)]
        rest_pairs = c["pairs"][:idx] + c["pairs"][idx + 1:]
        units_left = _merge(units, {ti: -1, tj: -1})
    else:
        first_free = [s for s, k in extra.items() for _ in range(k)]
        first_pairs = list(tool_pairs)
        rest_pairs = list(c["pairs"])
        units_left = dict(units)
    collect_gated = [s for s, k in units_left.items() if gate[s] == 1 for _ in range(k)]
    collect_free = [s for s, k in units_left.items() if gate[s] == 0 for _ in range(k)]
    steps = len(first_free) + len(first_pairs) + len(collect_gated) + len(collect_free) + len(rest_pairs)
    return dict(ok=ok, why="" if ok else "quota", tool=tool, first_free=first_free, first_pairs=first_pairs,
                collect_gated=collect_gated, collect_free=collect_free, pairs=rest_pairs,
                units_by_type=total, steps=steps)


def _annotate_chains(spec):
    chain = cheapest_chains(spec)
    plans = {k: plan_for(spec, chain, k) for k in chain if spec["tier"][k] >= 2}
    admitted = {t: [] for t in range(2, T_MAX + 1)}
    for k, p in plans.items():
        if p["ok"] and p["steps"] <= 48:
            admitted[spec["tier"][k]].append(k)
    for t in admitted:
        admitted[t].sort()
    spec["chains"], spec["plans"], spec["admitted"] = chain, plans, admitted
    spec["true_pre"] = [set(chain[k]["pairs"][-1][:2]) if k in chain and chain[k]["pairs"] else set()
                        for k in range(spec["n_items"])]
    free_t2 = any(spec["tier"][k] == 2 and not spec["decoy"][k] and k in chain
                  and all(spec["gate"][s] == 0 for s in chain[k]["units"]) for k in range(spec["n_items"]))
    spec["world_ok"] = bool(free_t2 and all(admitted[t] for t in admitted))
    return spec


def goal_stream(spec, n_t4=1):
    """The CHAIN-BLIND stream of ARC3_PLAN 2.8: the admitted item of tier 2, 3, 4 with the LOWEST index
    (indices are a per-world permutation, so this is a coin flip the experimenter never touches)."""
    out = [spec["admitted"][2][0], spec["admitted"][3][0]] + spec["admitted"][4][:n_t4]
    return out


# ---------------------------------------------------------------- skin: permutation

def permute_law_spec(spec, seed):
    """Item-index permutation (v55 rule) + cell-ID permutation WITHIN the used set (v58 rule: the frozen
    nav skill only ever saw cell IDs 1..9). Every law-world tensor is permuted consistently."""
    n = spec["n_items"]
    perm = np.random.default_rng(seed).permutation(n)              # new i <- old perm[i]
    inv = np.empty(n, dtype=int); inv[perm] = np.arange(n)          # old j -> new inv[j]
    m = lambda j: int(inv[j])                                        # noqa: E731
    po = np.full((n, n), BOTCH, dtype=int)
    for i in range(n):
        for j in range(n):
            v = spec["pair_out"][perm[i], perm[j]]
            po[i, j] = m(v) if v >= 0 else v
    out = dict(spec)
    out.update(
        kind=[spec["kind"][perm[i]] for i in range(n)],
        cell=[spec["cell"][perm[i]] for i in range(n)],
        depth=[spec["depth"][perm[i]] for i in range(n)],
        el=[spec["el"][perm[i]] for i in range(n)], tier=[spec["tier"][perm[i]] for i in range(n)],
        gate=[spec["gate"][perm[i]] for i in range(n)], decoy=[spec["decoy"][perm[i]] for i in range(n)],
        pair_out=po, _perm=perm.tolist(),
    )
    used = sorted({out["cell"][i] for i in range(n) if out["kind"][i] == "R"})
    if len(used) > 1:
        cperm = np.random.default_rng(seed + 7717).permutation(len(used))
        cm = {c: used[cperm[k]] for k, c in enumerate(used)}
        out["cell"] = [cm[c] if out["kind"][i] == "R" else c for i, c in enumerate(out["cell"])]
    _annotate_chains(out)
    out["target"] = out["admitted"][T_MAX][0] if out["admitted"][T_MAX] else 0
    out["_permuted"] = True
    return out


def make_world(seed, laws, perm_seed=None, **kw):
    """gen + permute, the way every run builds a world."""
    return permute_law_spec(gen_law_world(seed, laws, **kw), seed if perm_seed is None else perm_seed)


def admitted_worlds(laws, seeds, need=None, **kw):
    """Mechanical admission over a seed window: the first `need` seeds whose world is world_ok."""
    out = []
    for s in seeds:
        sp = make_world(s, laws, **kw)
        if sp["world_ok"]:
            out.append(s)
            if need is not None and len(out) >= need:
                break
    return out


# ---------------------------------------------------------------- the env

class LawVecTechTree(DeviceVecTechTree):
    """DeviceVecTechTree with the law world's primitives: gated + quota'd collects and combine(i, j).
    No craft actions. The grid, the egocentric view, the wall sentinel and nav mode are untouched, so the
    frozen childhood nav skill sees exactly the grids it was calibrated on."""

    def __init__(self, num_envs, spec, grid=7, view=13, n_resource=4, max_cells=None, seed=0):
        self.quota = int(spec["quota"])
        n = spec["n_items"]
        self.tier_t = torch.tensor(spec["tier"], dtype=torch.long, device=DEVICE)
        self.el_t = torch.tensor(spec["el"], dtype=torch.long, device=DEVICE)
        self.gate_t = torch.tensor(spec["gate"], dtype=torch.bool, device=DEVICE)
        self.decoy_t = torch.tensor(spec["decoy"], dtype=torch.bool, device=DEVICE)
        self.is_tool = self.tier_t >= 2
        self.pair_out_t = torch.tensor(spec["pair_out"], dtype=torch.long, device=DEVICE)
        gc = torch.zeros(spec["n_cells"], dtype=torch.bool, device=DEVICE)
        for i in range(n):
            if spec["kind"][i] == "R":
                gc[spec["cell"][i]] = bool(spec["gate"][i])
        self.gate_cell = gc
        self.quota_left = torch.full((num_envs, spec["n_cells"]), self.quota, dtype=torch.long, device=DEVICE)
        super().__init__(num_envs, spec, grid=grid, view=view, max_steps=10 ** 9, n_resource=n_resource,
                         goal=None, grant=None, seed=seed, max_cells=max_cells)
        self.action_dim = 6                                    # 0-3 move, 4 collect, 5 NOOP

    def reset(self):
        self.quota_left[:] = self.quota
        return super().reset()

    def _reset_done(self, done):
        super()._reset_done(done)
        self.quota_left[done] = self.quota

    def step(self, action):
        a = action.reshape(self.num_envs).long()
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=DEVICE)
        mv = a < 4
        if bool(mv.any()):
            np_ = (self.pos + deltas[a.clamp(max=3)]).clamp(0, self.G - 1)
            self.pos = torch.where(mv.unsqueeze(-1), np_, self.pos)
        coll = a == 4
        cur = self.grid[ar, self.pos[:, 0], self.pos[:, 1]]
        valid_cell = (cur >= 1) & (cur < self.n_cells)
        cc = cur.clamp(max=self.n_cells - 1)
        item_here = torch.where(valid_cell, self.cell2item[cc], torch.full_like(cur, -1))
        gated = valid_cell & self.gate_cell[cc]
        has_tool = ~gated | (self.inv[:, self.is_tool] > 0).any(-1)
        has_quota = self.quota_left[ar, cc] > 0
        do = coll & (item_here >= 0) & has_tool & has_quota
        if bool(do.any()):
            idx = do.nonzero(as_tuple=True)[0]
            items = item_here[idx]
            self.inv[idx, items] += 1
            self.unlocked[idx, items] = True
            self.quota_left[idx, cc[idx]] -= 1
        self.steps += 1
        self._set_state()
        z = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        return self.state, torch.zeros(N, device=DEVICE), z, z, z

    def combine(self, i, j):
        """One production attempt per env: slots i, j (N,) long. Returns (outcome code (N,), product (N,))
        with outcome in {OUT_BOTCH, OUT_INERT, OUT_PRODUCT, OUT_INVALID}; OUT_INVALID = an input was not
        held (nothing happens, nothing is written by the caller)."""
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        i = i.clamp(0, self.n_items - 1); j = j.clamp(0, self.n_items - 1)
        held = (self.inv[ar, i] > 0) & (self.inv[ar, j] > 0) & (i != j)
        po = self.pair_out_t[i, j]
        out = torch.full((N,), OUT_INVALID, dtype=torch.long, device=DEVICE)
        out = torch.where(held & (po >= 0), torch.full_like(out, OUT_PRODUCT), out)
        out = torch.where(held & (po == INERT), torch.full_like(out, OUT_INERT), out)
        out = torch.where(held & (po == BOTCH), torch.full_like(out, OUT_BOTCH), out)
        consume = held & (po != INERT)
        if bool(consume.any()):
            idx = consume.nonzero(as_tuple=True)[0]
            self.inv[idx, i[idx]] -= 1
            self.inv[idx, j[idx]] -= 1
        prod = held & (po >= 0)
        if bool(prod.any()):
            idx = prod.nonzero(as_tuple=True)[0]
            self.inv[idx, po[idx]] += 1
            self.unlocked[idx, po[idx]] = True
        self.steps += 1
        self._set_state()
        return out, torch.where(prod, po, torch.full_like(po, -1))
