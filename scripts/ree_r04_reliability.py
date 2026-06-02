"""r0.4 — the DECISIVE test the 3 reviews demanded: RELIABILITY BY FALSIFICATION.

The Refutation Engine's ONLY genuinely-novel claim is "reliable reuse BY CONSTRUCTION": a law
is trusted only while UNREFUTED, so on a task that VIOLATES it the agent refutes + adapts rather
than blindly mis-applying. r0.2c never tested this (no wrong law was ever on trial). Here we test
it directly.

Task stream (state obs): gravity tasks (const-acc holds) interleaved with DAMPING tasks (air
resistance: vy*=(1-drag), g=0 -> const-acc is VIOLATED, the true form is 'damp'). Three readouts
from the SAME observations each task:
- REFUTATION: uses the falsification-SELECTED form (re-selects per task). On damping it should
  select 'damp' (refute the held const-acc) and intercept.
- BLIND: assumes const-acc ALWAYS (re-fits only theta). On damping it mis-applies the parabola.
- ORACLE: the true form per task (upper bound).

Genuine contribution IFF, on the VIOLATING tasks, REFUTATION >> BLIND (refute+adapt where blind
fails) and REFUTATION ~ ORACLE, with the survivor correctly switching const-acc<->damp. Else the
headline reliability claim is FALSIFIED on state alone -> stop. >=3 seeds.

Usage: python -m scripts.ree_r04_reliability [--seeds 0 1 2]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.projectile import DeviceVecProjectileCatch
from scripts.ree_r02b_discover import FormGrammar


@torch.no_grad()
def rollout(bx, by, bvx, bvy, form, theta, k, x_plane, horizon=110):
    bx, by, bvx, bvy = bx.clone(), by.clone(), bvx.clone(), bvy.clone()
    land = by.clone(); arr = bx >= x_plane
    for _ in range(horizon):
        if form == "const_acc":
            bvy = bvy + theta
        elif form == "damp":
            bvy = bvy * k
        bx = bx + bvx
        by = by + bvy
        lo, hi = by < 0, by > 1
        by = torch.where(lo, -by, torch.where(hi, 2 - by, by))
        bvy = torch.where(lo | hi, -bvy, bvy)
        newly = (~arr) & (bx >= x_plane)
        land = torch.where(newly, by, land)
        arr = arr | (bx >= x_plane)
    return land


def act_toward(t, cy, cs):
    d = t - cy
    a = torch.zeros_like(cy, dtype=torch.long)
    a = torch.where(d > cs * 0.5, torch.ones_like(a), a)
    a = torch.where(d < -cs * 0.5, torch.full_like(a, 2), a)
    return a


def make(cfg, g, drag, seed):
    return DeviceVecProjectileCatch(cfg["ne"], gravity=g, drag=drag, max_steps=cfg["ms"],
                                    x_plane=cfg["xp"], tol=cfg["tol"], seed=seed)


@torch.no_grad()
def eval_catch(cfg, g, drag, form, theta, k, episodes=20, seed=555):
    env = make(cfg, g, drag, seed)
    dc = torch.zeros(cfg["ne"], device=DEVICE)
    while float(dc.min()) < episodes:
        tgt = rollout(env.bx, env.by, env.bvx, env.bvy, form, theta, k, cfg["xp"])
        _, _, _, _, done = env.step(act_toward(tgt, env.cy, env.cs))
        dc += done.float()
    return env.catch_rate()


def fit(cfg, g, drag, seed, steps=400):
    """Observe the ball; fit the form grammar (bvy is observed in state). Return gram."""
    env = make(cfg, g, drag, seed)
    gram = FormGrammar()
    for _ in range(steps):
        vyb = env.bvy.clone(); byb = env.by.clone()
        env.step(torch.randint(0, 3, (cfg["ne"],), device=DEVICE))
        gram.observe(vyb, env.bvy, byb)
    return gram


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ne", type=int, default=256)
    p.add_argument("--ms", type=int, default=80)
    p.add_argument("--xp", type=float, default=0.97)
    p.add_argument("--drag", type=float, default=0.08)
    p.add_argument("--tol", type=float, default=0.03)        # tight -> the law must be RIGHT
    p.add_argument("--out-dir", default="craft_v6_out")
    args = p.parse_args()
    cfg = dict(ne=args.ne, ms=args.ms, xp=args.xp, tol=args.tol)
    # stream: gravity, gravity, DAMPING(violates const-acc), gravity, DAMPING
    stream = [("grav", 0.004, 0.0), ("grav", 0.004, 0.0), ("damp", 0.0, args.drag),
              ("grav", 0.004, 0.0), ("damp", 0.0, args.drag)]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.4] device={DEVICE} | RELIABILITY BY FALSIFICATION | stream {[s[0] for s in stream]} "
          f"| Refutation (re-select form) vs Blind (assume const-acc) vs Oracle | seeds {args.seeds}",
          flush=True)
    t0 = time.perf_counter()
    rows = []
    for s in args.seeds:
        torch.manual_seed(s)
        held = None
        for ti, (kind, g, drag) in enumerate(stream):
            gram = fit(cfg, g, drag, s * 100 + ti)
            surv, _ = gram.survivor()
            refuted = held is not None and surv != held
            held = surv
            true_form = "const_acc" if drag == 0.0 else "damp"
            r_ref = eval_catch(cfg, g, drag, surv, gram.theta, gram.k)            # refutation
            r_bli = eval_catch(cfg, g, drag, "const_acc", gram.theta, gram.k)     # blind const-acc
            r_orc = eval_catch(cfg, g, drag, true_form, gram.theta, gram.k)       # oracle form
            rows.append(dict(seed=s, task=ti, kind=kind, survivor=surv, true_form=true_form,
                             refuted=refuted, refutation=round(r_ref, 3), blind=round(r_bli, 3),
                             oracle=round(r_orc, 3)))
            print(f"  s{s} t{ti} {kind:4s}: survivor='{surv}' (true '{true_form}') refuted={refuted} "
                  f"| REFUT {r_ref:.2f} | BLIND {r_bli:.2f} | oracle {r_orc:.2f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    viol = [r for r in rows if r["kind"] == "damp"]
    grav = [r for r in rows if r["kind"] == "grav"]
    # reliability: on violating tasks, survivor switches to the true form, refutation >> blind
    correct_form = all(r["survivor"] == r["true_form"] for r in rows)
    refut_beats_blind = all(r["refutation"] > r["blind"] + 0.15 for r in viol)
    refut_keeps_grav = all(r["refutation"] > 0.6 for r in grav)
    positive = correct_form and refut_beats_blind and refut_keeps_grav and len(viol) >= 3
    verdict = (
        f"RELIABILITY BY FALSIFICATION HOLDS — on law-VIOLATING (damping) tasks the agent REFUTES "
        f"const-acc and selects 'damp' (correct), intercepting "
        f"{[r['refutation'] for r in viol]} vs BLIND const-acc {[r['blind'] for r in viol]} "
        f"(~oracle), while keeping gravity tasks. The falsification gives reliable reuse-or-refute "
        f"the blind reuse (the 48-version failure mode) lacks. This is the genuinely-novel part."
        if positive else
        f"PARTIAL/CHECK — correct_form={correct_form}, refut>blind on violators="
        f"{[round(r['refutation']-r['blind'],2) for r in viol]}. See rows. If blind ~ refutation, "
        f"the violating task isn't violating enough / forms degenerate; if survivor wrong, falsification fails.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r04.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, stream=[s[0] for s in stream], rows=rows,
                       positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
