"""v22 — INTEGRATION M2: recognise-OR-LEARN (novelty detection + library growth).

M1 showed recognise-and-reuse. M2 adds the developmental core: faced with a
stream of tasks, the agent REUSES a known concept when one fits, but DETECTS
when nothing fits (a genuinely novel physics) and LEARNS a new concept, growing
its library. = vision points 2 (reuse), 3 (recognise), 4 (learn anew).

Library starts with 3 known landing-rules; a 4th NOVEL rule (max) appears in the
stream. The agent should: reuse on known rules, detect the novel one (all models
fit poorly), learn it, add it, and reuse it thereafter.

Usage: python -m scripts.integrate_v22
"""

import argparse
import itertools
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE

W, H, NCOL = 8, 14, 5
ALL_SHAPES = torch.tensor(list(itertools.product(range(4), repeat=4)),
                          dtype=torch.float32, device=DEVICE)
RULE_NAMES = ["gravity(min)", "edge(first)", "soft(mean)", "NOVEL(max)"]
NOVEL = 3
THRESH = 0.5            # min-error above this on ALL known models => novel


def landing(surface, bp, rule):
    out = torch.empty(surface.shape[0], NCOL, device=DEVICE)
    for col in range(NCOL):
        cand = surface[:, col:col + 4] - 1 - bp
        if rule == 0:
            out[:, col] = cand.min(dim=1).values
        elif rule == 1:
            out[:, col] = cand[:, 0]
        elif rule == 2:
            out[:, col] = cand.mean(dim=1)
        else:
            out[:, col] = cand.max(dim=1).values        # the NOVEL rule
    return out


def gen_batch(n, pool, rule):
    surface = H - torch.randint(0, H - 2, (n, W), device=DEVICE).float()
    bp = pool[torch.randint(0, pool.shape[0], (n,), device=DEVICE)]
    return surface, bp, landing(surface, bp, rule)


class RuleNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W + 4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, NCOL))

    def forward(self, surface, bp):
        return self.net(torch.cat([surface / H, bp / 3.0], -1)) * H


def train_on(rule, shapes, iters, batch=2048):
    m = RuleNet().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(iters):
        s, bp, land = gen_batch(batch, shapes, rule)
        loss = (m(s, bp) - land).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def mae(m, s, bp, land):
    return float((m(s, bp) - land).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--learn-iters", type=int, default=2500)
    p.add_argument("--n-tasks", type=int, default=200)
    p.add_argument("--obs", type=int, default=128)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.learn_iters, args.n_tasks = 300, 300, 40

    perm = torch.randperm(256, device=DEVICE)
    train_shapes, test_shapes = ALL_SHAPES[perm[:200]], ALL_SHAPES[perm[200:]]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v22] device={DEVICE} | INTEGRATION M2 | recognise-OR-learn | "
          f"library starts with rules 0,1,2; NOVEL rule {NOVEL} (max) appears",
          flush=True)
    t0 = time.perf_counter()
    library = [train_on(r, train_shapes, args.iters) for r in range(3)]
    lib_rules = [0, 1, 2]                                 # which rule each model is
    print(f"  initial library: {len(library)} models | {time.perf_counter()-t0:.0f}s",
          flush=True)

    # stream of tasks; reuse if a model fits, else detect novelty + learn
    detect_ok, reuse_err, learned_novel, size_curve = 0, [], 0, []
    for t in range(args.n_tasks):
        rule = int(torch.randint(0, 4, (1,)))            # hidden true rule
        s, bp, land = gen_batch(args.obs, test_shapes, rule)
        maes = [mae(m, s, bp, land) for m in library]
        best = int(torch.tensor(maes).argmin()); best_mae = maes[best]
        is_novel_pred = best_mae > THRESH
        is_novel_true = lib_rules[best] != rule if not is_novel_pred else (rule not in lib_rules)
        # detection correctness: novel-predicted iff the true rule isn't in the library
        true_known = rule in lib_rules
        detect_ok += int(is_novel_pred != true_known)
        if is_novel_pred:                                # NOVEL -> learn it, add
            newm = train_on(rule, train_shapes, args.learn_iters)
            library.append(newm); lib_rules.append(rule)
            learned_novel += 1
            reuse_err.append(mae(newm, *gen_batch(args.obs, test_shapes, rule)))
        else:                                            # KNOWN -> reuse
            reuse_err.append(best_mae)
        size_curve.append(len(library))

    acc = detect_ok / args.n_tasks
    re_ = sum(reuse_err) / len(reuse_err)
    grew = len(library) > 3 and NOVEL in lib_rules
    ok = acc >= 0.9 and grew and re_ <= 0.3
    verdict = (f"RECOGNISE-OR-LEARN WORKS — novelty detection accuracy {acc:.0%}; "
               f"the agent reused known concepts AND detected the novel physics, "
               f"learned it, and grew its library 3 -> {len(library)} (final reuse "
               f"error {re_:.3f}). The developmental core — reuse what fits, learn "
               f"what's new — runs end to end."
               if ok else
               f"PARTIAL — detect acc {acc:.0%}, library {len(library)}, reuse err "
               f"{re_:.3f}, novel learned {learned_novel}x.")
    print(f"\n  -> {verdict}\n  final library size {len(library)} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v22_integration_m2.json"), "w") as f:
        json.dump(dict(detect_acc=acc, final_library=len(library),
                       reuse_err=re_, learned_novel=learned_novel,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
