"""ARC 3 — the STORE SWEEPER's decision rule, shared by the CPU model (scripts/mc_v61.py, perfect nav) and
the GPU reference G' (scripts/hand_v61.py, real nav). One function, one policy, so the gate's CONSISTENT
check compares the same rule under two executors.

G' knows the tier law and the gate law and its own store; it never reads Bond/Out. Rule, per env:
    1. if the goal's recipe is KNOWN and buildable this episode -> build it (obtain inputs, combine)
    2. else pick the CHEAPEST untried equal-tier pair among PRODUCIBLE items (raws with quota, or items
       with a known recipe whose inputs are producible), random ties -> build toward it
    3. else, if a gated raw is unexplored and no tool is held -> build a known tier-2 item from free raws
    4. else collect any raw with quota (refill)
"obtain(item)" is a one-step recursive planner over inventory counts: collect a raw (building a tool
first if the raw is gated and none is held), or obtain the inputs of a known recipe, then combine.
"""

import numpy as np

T_MAX = 4


def sweeper_action(inv, quota, known, nonprod, tried_ep, tier, gate, goal, rng):
    """inv (n,) int | quota (n,) units left per RAW slot (any value for crafts) | known: dict (i,j)->k with
    i<j | nonprod: set of (i,j) botched or inert | tried_ep: set of (i,j) tried this episode | tier (n,)
    | gate (n,) | goal int | rng. Returns ("collect", s) or ("combine", i, j) or None."""
    n = len(inv)
    held = inv > 0
    has_tool = any(held[k] and tier[k] >= 2 for k in range(n))
    recipes = {}
    for (i, j), k in known.items():
        if k >= 0:
            recipes.setdefault(k, []).append((i, j))

    def cost(k, depth=0):
        """raw units by slot to make one k from scratch this episode, None if impossible."""
        if depth > 8:
            return None
        if tier[k] == 1:
            return {k: 1} if quota[k] > 0 else None
        best = None
        for (i, j) in recipes.get(k, []):
            ci, cj = cost(i, depth + 1), cost(j, depth + 1)
            if ci is None or cj is None:
                continue
            c = dict(ci)
            for s, v in cj.items():
                c[s] = c.get(s, 0) + v
            if any(c[s] > quota[s] for s in c):
                continue
            if best is None or sum(c.values()) < sum(best.values()):
                best = c
        return best

    def free_tool():
        for k in range(n):
            if tier[k] == 2 and k in recipes:
                for (i, j) in recipes[k]:
                    if tier[i] == 1 and tier[j] == 1 and not gate[i] and not gate[j] and quota[i] > 0 and quota[j] > 0:
                        return k, (i, j)
        return None

    def obtain(k, depth=0):
        if depth > 12:
            return None
        if tier[k] == 1:
            if gate[k] and not has_tool:
                ft = free_tool()
                return obtain(ft[0], depth + 1) if ft else None
            return ("collect", k) if quota[k] > 0 else None
        opts = [(i, j) for (i, j) in recipes.get(k, []) if cost(i) is not None or held[i]]
        if not opts:
            return None
        i, j = opts[0]
        if not held[i]:
            return obtain(i, depth + 1)
        if not held[j] or (i == j and inv[i] < 2):
            return obtain(j, depth + 1)
        return ("combine", i, j)

    # 1. the goal
    if goal in recipes and (cost(goal) is not None or any(held[i] and held[j] for (i, j) in recipes[goal])):
        a = obtain(goal)
        if a is not None:
            return a
    # 2. the cheapest untried producible equal-tier pair
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if tier[i] != tier[j] or tier[i] >= T_MAX:
                continue
            if (i, j) in known or (i, j) in nonprod or (i, j) in tried_ep:
                continue
            ci = {i: 0} if held[i] else cost(i)
            cj = {j: 0} if held[j] else cost(j)
            if ci is None or cj is None:
                continue
            tot = sum(ci.values()) + sum(cj.values())
            cands.append((tot, tier[i], rng.random(), i, j))
    if cands:
        cands.sort()
        _, _, _, i, j = cands[0]
        if held[i] and held[j]:
            return ("combine", i, j)
        a = obtain(i) if not held[i] else obtain(j)
        if a is not None:
            return a
    # 3. unlock gated raws
    if not has_tool and any(tier[k] == 1 and gate[k] and quota[k] > 0 for k in range(n)):
        ft = free_tool()
        if ft:
            a = obtain(ft[0])
            if a is not None:
                return a
    # 4. refill
    opts = [k for k in range(n) if tier[k] == 1 and quota[k] > 0 and (not gate[k] or has_tool)]
    if opts:
        return ("collect", int(opts[int(rng.integers(len(opts)))]))
    return None


def simulate_sweep(spec, goal, rng, n_mc=50, cap=576, macro=48):
    """CPU model: perfect nav, the quota, three outcomes; the store persists across episodes within the
    goal (per-goal store), EMPTY at start. Returns (median over runs of attempts to first obtain, censored
    at `cap`; censored fraction) — the SAME statistic run_goal_v61 records per env (first_demo_attempt =
    median over envs, censored at the budget), so the gate's CONSISTENT check compares like with like."""
    n, tier, gate, po, q = spec["n_items"], spec["tier"], spec["gate"], spec["pair_out"], spec["quota"]
    costs = []
    for _ in range(n_mc):
        known, nonprod = {}, set()
        steps, got = 0, False
        while steps < cap and not got:
            inv = np.zeros(n, dtype=int)
            quota = np.array([q if tier[k] == 1 else 99 for k in range(n)])
            tried_ep = set()
            for _ep in range(macro):
                if steps >= cap or got:
                    break
                a = sweeper_action(inv, quota, known, nonprod, tried_ep, tier, gate, goal, rng)
                steps += 1
                if a is None:
                    continue
                if a[0] == "collect":
                    s = a[1]
                    if quota[s] > 0 and (not gate[s] or any(inv[k] > 0 and tier[k] >= 2 for k in range(n))):
                        inv[s] += 1; quota[s] -= 1
                else:
                    i, j = a[1], a[2]
                    if inv[i] <= 0 or inv[j] <= 0:
                        continue
                    p = po[i, j]
                    tried_ep.add((i, j))
                    if p == -2:                       # BOTCH
                        inv[i] -= 1; inv[j] -= 1; nonprod.add((i, j))
                    elif p == -1:                     # INERT
                        nonprod.add((i, j))
                    else:
                        inv[i] -= 1; inv[j] -= 1; inv[p] += 1; known[(i, j)] = int(p)
                        if p == goal:
                            got = True
        costs.append(steps if got else cap)
    return int(np.median(costs)), float(np.mean([c >= cap for c in costs]))
