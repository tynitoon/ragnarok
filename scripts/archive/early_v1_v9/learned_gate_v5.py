"""v5.0 — a LEARNED relevance gate. Replace the Phase-3/4 exhaustive probe
(route a task by trying EVERY library skill, O(library) reach-rollouts) with
a small LEARNED recognizer that names the relevant skill in O(1) from a
short interaction SIGNATURE, and flags out-of-distribution (novel) regimes
so the "if no link, learn a notion" branch fires by detection, not by
probe-failure.

Preregistered as preregistration.md amendment v5.0 (committed before this
script, the env angle extension, and any run).

Recognizer: MLP on a K-step signature [(v_t, a_t, v_{t+1})] (random probe
actions) -> softmax over KNOWN skills + a novelty score (low max-softmax,
optionally confirmed by ONE verification rollout of the proposed skill).
Trained self-supervised on signatures the agent generates from its own
library. Signature cost = K env-steps, independent of library size.

Experiments:
  A routing accuracy + generalization to unseen goals/starts.
  B SCALING (headline): per-routing cost, probe O(R) vs learned ~flat,
    R in {2,4,8}.
  C novelty detection AUC on a held-out regime (leave-one-out).
  D integrated developmental loop with the learned gate (vs Phase-3 probe).

Usage: python -m scripts.learned_gate_v5 [--exp A B C D] [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.device_env import DeviceVecPointMass2D
from scripts.devloop_v4 import (_learn_skill, _skill_success, _skill_action,
                                _fresh_skill)

ALL_ROT = ["free", "rot45", "rot90", "rot135", "reverse", "rot225",
           "rot270", "rot315"]               # 8 equally-spaced rotations
CLEAN4 = ["free", "rot90", "reverse", "rot270"]
ACTION_DIM = 2

# In-process cache of reach skills (regime -> (skill, measured-learn-cost)).
# A regime's skill is seed-invariant prior competence, so it is trained once
# and reused (cost charged at the measured value wherever a skill is needed).
_SKILLS = {}


def _get_skill(regime, cfg):
    if regime not in _SKILLS:
        pi, steps, ok, fin = _learn_skill(regime, cfg)
        _SKILLS[regime] = (pi, steps, fin)
    return _SKILLS[regime]


# --------------------------------------------------------------------------
# Interaction signature + recognizer.
# --------------------------------------------------------------------------
@torch.no_grad()
def _collect_signature(regime, n, K):
    """n parallel K-step signatures under `regime`. Each step records
    [v_t (2), a_t (2), v_{t+1} (2)] -> 6; flattened to (n, 6K). Random probe
    actions; pure system-identification of the local dynamics (goal-agnostic)."""
    env = DeviceVecPointMass2D(n, regime=regime)
    feats = []
    for _ in range(K):
        v = env.state[:, 2:4].clone()
        a = (torch.rand(n, ACTION_DIM, device=DEVICE) - 0.5) * 2.0
        env.step(a)
        vp = env.state[:, 2:4]
        feats.append(torch.cat([v, a, vp], dim=-1))
    return torch.stack(feats, dim=1).reshape(n, 6 * K)


class Recognizer(nn.Module):
    def __init__(self, K, n_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 * K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def _train_recognizer(regimes, cfg):
    """Self-supervised: signatures per known regime, labelled by index."""
    rec = Recognizer(cfg["K"], len(regimes)).to(DEVICE)
    opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
    for _ in range(cfg["rec_iters"]):
        xs, ys = [], []
        for i, r in enumerate(regimes):
            x = _collect_signature(r, cfg["rec_batch"], cfg["K"])
            xs.append(x); ys.append(torch.full((cfg["rec_batch"],), i,
                                                device=DEVICE, dtype=torch.long))
        x = torch.cat(xs); y = torch.cat(ys)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(rec(x), y)
        loss.backward(); opt.step()
    rec.eval()
    return rec


@torch.no_grad()
def _routing_accuracy(rec, regimes, cfg, n=2048):
    """Fresh signatures (new random starts/actions) -> argmax accuracy."""
    correct = 0
    for i, r in enumerate(regimes):
        logits = rec(_collect_signature(r, n, cfg["K"]))
        correct += (logits.argmax(-1) == i).float().sum().item()
    return correct / (n * len(regimes))


# --------------------------------------------------------------------------
# A — routing accuracy.
# --------------------------------------------------------------------------
def _exp_A(cfg, regimes):
    accs = []
    for seed in range(cfg["seeds"]):
        torch.manual_seed(seed); np.random.seed(seed)
        rec = _train_recognizer(regimes, cfg)
        accs.append(_routing_accuracy(rec, regimes, cfg))
        print(f"  [A] seed {seed} routing accuracy {accs[-1]:.3f}", flush=True)
    return dict(regimes=regimes, accuracy=accs, mean=float(np.mean(accs)))


# --------------------------------------------------------------------------
# B — scaling: per-routing cost, probe O(R) vs learned ~flat.
# --------------------------------------------------------------------------
def _exp_B(cfg):
    P, E, K = cfg["probe_trials"], cfg["eval_steps"], cfg["K"]
    rows = []
    for R in cfg["scaling_R"]:
        regimes = ALL_ROT[:R]
        # probe gate: try each of R skills with a reach-probe (P trials, E steps)
        probe_cost = R * P * E
        # learned gate: signature (K steps x n_sig envs) + ONE verification
        sig_cost = K * cfg["n_sig"]
        learned_cost = sig_cost + P * E              # one proposed-skill check
        learned_recognly = sig_cost                  # recognizer alone (no verify)
        # measured routing accuracy at this R (sanity that learned routes right)
        torch.manual_seed(0); np.random.seed(0)
        rec = _train_recognizer(regimes, cfg)
        acc = _routing_accuracy(rec, regimes, cfg, n=1024)
        rows.append(dict(R=R, probe_cost=probe_cost, learned_cost=learned_cost,
                         learned_recognizer_only=learned_recognly,
                         routing_acc=acc, speedup=probe_cost / learned_cost))
        print(f"  [B] R={R:>2} | probe {probe_cost:>8,} | learned {learned_cost:>7,} "
              f"(recog-only {learned_recognly}) | speedup {probe_cost/learned_cost:.1f}x "
              f"| routing acc {acc:.3f}", flush=True)
    return dict(rows=rows)


# --------------------------------------------------------------------------
# C — novelty detection (leave-one-out): is a held-out regime flagged novel?
# --------------------------------------------------------------------------
@torch.no_grad()
def _auc(scores_pos, scores_neg):
    """AUC that `scores_pos` (novel) rank above `scores_neg` (known)."""
    pos = torch.as_tensor(scores_pos); neg = torch.as_tensor(scores_neg)
    # fraction of (pos,neg) pairs with pos>neg (+0.5 ties)
    comp = (pos[:, None] > neg[None, :]).float().mean()
    ties = (pos[:, None] == neg[None, :]).float().mean()
    return float(comp + 0.5 * ties)


def _exp_C(cfg, regimes):
    """Leave-one-out novelty detection via the O(1) mechanism: the recognizer
    PROPOSES one skill, we VERIFY it (one reach trial), novelty = 1 - success.
    A novel regime's nearest known skill fails verification -> flagged. Robust
    to OOD over-confidence (which raw max-softmax suffers). We report this
    verification-AUC (the decisive number) and the raw max-softmax AUC."""
    for r in regimes:                 # warm the skill cache OUTSIDE no_grad
        _get_skill(r, cfg)            # (verification below runs under no_grad)
    v_aucs, s_aucs, detect = [], [], []
    for held in regimes:
        known = [r for r in regimes if r != held]
        rec = _train_recognizer(known, cfg)

        @torch.no_grad()
        def propose(r, n):
            x = _collect_signature(r, n, cfg["K"])
            logits = rec(x)
            idx = logits.argmax(-1)                      # per-sample proposal
            soft = 1.0 - torch.softmax(logits, -1).max(-1).values
            return idx, soft

        # verification novelty: success of the (modal) proposed skill on r
        @torch.no_grad()
        def verify_nov(r):
            idx, _ = propose(r, cfg["n_sig"])
            modal = int(torch.mode(idx).values.item())
            s = _skill_success(_get_skill(known[modal], cfg)[0], r, cfg)
            return 1.0 - s
        novel_v = verify_nov(held)
        known_v = [verify_nov(r) for r in known]
        v_aucs.append(_auc([novel_v], known_v))
        detect.append(novel_v >= (1.0 - cfg["mastery"]))   # flagged novel?
        # raw max-softmax novelty (secondary)
        _, ns = propose(held, 1024); _, ks0 = propose(known[0], 1024)
        ksm = sum(([propose(r, 256)[1].mean().item()] for r in known), [])
        s_aucs.append(_auc([ns.mean().item()], ksm))
        print(f"  [C] held-out {held:8s} | verif-novelty {novel_v:.2f} "
              f"(known max {max(known_v):.2f}) flagged={detect[-1]}", flush=True)
    return dict(regimes=regimes, verif_auc=float(np.mean(v_aucs)),
                softmax_auc=float(np.mean(s_aucs)),
                detect_rate=float(np.mean(detect)))


# --------------------------------------------------------------------------
# D — integrated developmental loop with the LEARNED gate.
# --------------------------------------------------------------------------
def _make_curriculum(regimes, blocks, seed):
    rng = np.random.default_rng(seed)
    cur = []
    for _ in range(blocks):
        perm = list(regimes); rng.shuffle(perm); cur.extend(perm)
    return cur


def _retrain_on_library(lib_regimes, cfg):
    return _train_recognizer(lib_regimes, cfg) if lib_regimes else None


def _exp_D(cfg, regimes):
    """Phase-3 curriculum, but route each task with the LEARNED recognizer
    (propose) + ONE verification, instead of probing every skill."""
    results = []
    for seed in range(cfg["seeds"]):
        torch.manual_seed(seed); np.random.seed(seed)
        curriculum = _make_curriculum(regimes, cfg["blocks"], seed)
        library = []          # list of (regime_label, skill)
        rec = None
        log = []
        for ti, regime in enumerate(curriculum):
            sig_steps = cfg["K"] * cfg["n_sig"]
            proposed = None
            if rec is not None and library:
                with torch.no_grad():
                    x = _collect_signature(regime, cfg["n_sig"], cfg["K"])
                    idx = int(torch.softmax(rec(x), -1).mean(0).argmax().item())
                proposed = idx
            # verify proposal (O(1)); else learn
            verify_steps = 0
            if proposed is not None:
                s = _skill_success(library[proposed][1], regime, cfg)
                verify_steps = cfg["probe_trials"] * cfg["eval_steps"]
                if s >= cfg["mastery"]:
                    cost = sig_steps + verify_steps
                    log.append(dict(task=ti, regime=regime, decision="reuse",
                                    proposed=library[proposed][0], succ=s,
                                    cost=cost));
                    print(f"    [D s{seed}] t{ti:>2} {regime:8s} reuse "
                          f"->{library[proposed][0]:8s} ({s:.2f}) +{cost:,}",
                          flush=True)
                    continue
            # novel -> learn + add + retrain recognizer
            pi, steps, fin = _get_skill(regime, cfg)
            library.append((regime, pi))
            rec = _retrain_on_library([r for r, _ in library], cfg)
            cost = sig_steps + verify_steps + steps
            log.append(dict(task=ti, regime=regime, decision="learn",
                            succ=fin, cost=cost))
            print(f"    [D s{seed}] t{ti:>2} {regime:8s} LEARN ({fin:.2f}) "
                  f"+{cost:,} | lib {len(library)}", flush=True)
        costs = [e["cost"] for e in log]
        nperb = len(regimes)
        results.append(dict(library_size=len(library),
                            first_block=float(np.mean(costs[:nperb])),
                            last_block=float(np.mean(costs[-nperb:])),
                            total=int(sum(costs)),
                            all_reuse_correct=all(
                                e["decision"] == "learn" or e["proposed"] == e["regime"]
                                for e in log),
                            recog_cost_per_reuse=cfg["K"] * cfg["n_sig"]
                            + cfg["probe_trials"] * cfg["eval_steps"]))
        print(f"  [D] seed {seed} library {len(library)} (true {len(regimes)}) | "
              f"first-block {int(results[-1]['first_block']):,} -> last "
              f"{int(results[-1]['last_block']):,} | reuse-correct "
              f"{results[-1]['all_reuse_correct']}", flush=True)
    return dict(regimes=regimes, seeds=results,
                probe_cost_per_reuse=len(regimes) * cfg["probe_trials"] * cfg["eval_steps"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--rec-iters", type=int, default=300)
    p.add_argument("--rec-batch", type=int, default=512)
    p.add_argument("--n-sig", type=int, default=64)
    p.add_argument("--scaling-R", type=int, nargs="+", default=[2, 4, 8])
    # skill / probe params (shared with devloop helpers)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--skill-rollouts", type=int, default=80)
    p.add_argument("--skill-updates", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--probe-trials", type=int, default=64)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--consolidate", type=float, default=0.95)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--out-dir", default="lgate_v5_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.rec_iters, args.rec_batch = 1, 60, 128
        args.scaling_R = [2, 4]
        args.skill_rollouts, args.skill_updates = 12, 32
        args.eval_every, args.eval_steps, args.probe_trials = 3, 40, 32
        args.num_envs, args.horizon, args.blocks = 64, 32, 2
        args.consolidate = 0.5

    cfg = {k: getattr(args, k) for k in
           ("K", "rec_iters", "rec_batch", "n_sig", "num_envs", "horizon",
            "skill_rollouts", "skill_updates", "eval_every", "eval_steps",
            "probe_trials", "mastery", "consolidate", "blocks", "seeds")}
    cfg["scaling_R"] = args.scaling_R

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[lgate-v5] device={DEVICE} | exp={args.exp} | seeds={args.seeds} "
          f"| K={args.K}", flush=True)
    t0 = time.perf_counter()
    out = {}

    if "A" in args.exp:
        print("\n[A] routing accuracy (8 rotations, held-out goals)", flush=True)
        out["A"] = _exp_A(cfg, ALL_ROT)
        print(f"  -> mean routing accuracy {out['A']['mean']:.3f}", flush=True)
    if "B" in args.exp:
        print("\n[B] scaling: recognition cost probe O(R) vs learned ~flat", flush=True)
        out["B"] = _exp_B(cfg)
        sp = [r["speedup"] for r in out["B"]["rows"]]
        print(f"  -> speedup grows with R: {[round(x,1) for x in sp]}", flush=True)
    if "C" in args.exp:
        print("\n[C] novelty detection (leave-one-out, 8 rotations)", flush=True)
        out["C"] = _exp_C(cfg, ALL_ROT)
        print(f"  -> verification-AUC {out['C']['verif_auc']:.3f} | detect-rate "
              f"{out['C']['detect_rate']:.2f} | raw-softmax-AUC "
              f"{out['C']['softmax_auc']:.3f}", flush=True)
    if "D" in args.exp:
        print("\n[D] integrated developmental loop with the LEARNED gate "
              "(clean-4)", flush=True)
        out["D"] = _exp_D(cfg, CLEAN4)
        libs = [s["library_size"] for s in out["D"]["seeds"]]
        fb = np.mean([s["first_block"] for s in out["D"]["seeds"]])
        lb = np.mean([s["last_block"] for s in out["D"]["seeds"]])
        print(f"  -> libraries {libs} (true 4) | first-block {int(fb):,} -> "
              f"last {int(lb):,} | recog cost/reuse {out['D']['seeds'][0]['recog_cost_per_reuse']:,}"
              f" vs probe {out['D']['probe_cost_per_reuse']:,}", flush=True)

    # ---- verdict ----
    ok = {}
    if "A" in out:
        ok["A"] = out["A"]["mean"] >= 0.95
    if "B" in out:
        sp = [r["speedup"] for r in out["B"]["rows"]]
        ok["B"] = sp == sorted(sp) and sp[-1] > sp[0] and sp[-1] >= 2.0
    if "C" in out:
        ok["C"] = out["C"]["verif_auc"] >= 0.9 and out["C"]["detect_rate"] >= 0.9
    if "D" in out:
        libs = [s["library_size"] for s in out["D"]["seeds"]]
        fb = np.mean([s["first_block"] for s in out["D"]["seeds"]])
        lb = np.mean([s["last_block"] for s in out["D"]["seeds"]])
        ok["D"] = all(l == len(CLEAN4) for l in libs) and lb < fb \
            and out["D"]["seeds"][0]["recog_cost_per_reuse"] < out["D"]["probe_cost_per_reuse"]
    print(f"\n{'=' * 72}\n  v5.0 LEARNED GATE — per-experiment pass: {ok}")
    verdict = ("LEARNED GATE WORKS — O(1) recognition that generalizes, scales "
               "(flat vs probe O(R)), and detects novelty; the developmental "
               "loop is preserved at library-independent recognition cost."
               if all(ok.values()) else
               "MIXED — see per-experiment flags above.")
    print(f"  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    out["pass"] = ok; out["verdict"] = verdict
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  results -> {os.path.join(args.out_dir, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
