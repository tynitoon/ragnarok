"""r0.2b of the REFUTATION ENGINE — DISCOVER THE LAW FORM (not just a parameter).

The genuine "little scientist": the agent does NOT know the world is gravity. It holds a
small GRAMMAR of candidate forms for how vertical velocity evolves and FALSIFIES them:
  - constant-velocity:     vy_next = vy           (residual high under gravity)
  - constant-acceleration: vy_next = vy + theta   (gravity lives here; theta fit)
  - damping:               vy_next = k * vy        (residual high under gravity)
It keeps the SURVIVOR (lowest residual). Separately, where the survivor BREAKS (large
residual), it checks the boundary and discovers the REFLECTIVE-BOUNCE conditional
(near a wall, vy_next = -vy). It then rolls out the DISCOVERED law (survivor form +
discovered params + discovered bounce) to predict the landing and intercept.

Honest bound: the grammar (3 forms + the reflective hypothesis) is hand-provided — full
form-invention is program induction (open problem). But the SELECTION + the bounce
discovery are the agent's, by falsification = genuine discovery within the grammar (more
than r0.2a's single-parameter fit). Success: the agent picks constant-ACCELERATION (not
the alternatives) by residual, discovers the bounce, and intercepts near the oracle.

Usage: python -m scripts.ree_r02b_discover [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.projectile import DeviceVecProjectileCatch


class FormGrammar:
    """Falsify 3 candidate vy-update forms + discover reflective bounce."""
    def __init__(self):
        # constant-acceleration: theta = mean(dvy); residual sum
        self.Sdvy = 0.0; self.n = 0.0
        self.r_acc = 0.0          # sum (dvy - theta)^2
        self.r_vel = 0.0          # sum (dvy - 0)^2   (constant velocity)
        # damping vy_next = k vy: k = S(vyn*vy)/S(vy*vy)
        self.Svv = 0.0; self.Svyn = 0.0
        self.r_damp_pairs = []    # accumulate for k then residual in two passes via sums
        self.Svy2 = 0.0; self.Svynvy = 0.0; self.Svyn2 = 0.0
        # bounce discovery
        self.bounce_seen = 0.0; self.bounce_reflect = 0.0

    def observe(self, vy, vy_next, by):
        dvy = vy_next - vy
        nb = dvy.abs() < 0.03                                  # non-bounce (no sign flip)
        d = dvy[nb]; v = vy[nb]; vn = vy_next[nb]
        if d.numel():
            self.Sdvy += float(d.sum()); self.n += d.numel()
            self.Svy2 += float((v * v).sum()); self.Svynvy += float((vn * v).sum())
            self.Svyn2 += float((vn * vn).sum())
        # bounce regime: large |dvy| near a boundary -> is it reflective (vy_next ~ -vy)?
        bb = (~nb) & ((by < 0.08) | (by > 0.92))
        if bool(bb.any()):
            self.bounce_seen += float(bb.sum())
            self.bounce_reflect += float(((vy_next[bb] + vy[bb]).abs() < 0.02).sum())

    @property
    def theta(self):
        return self.Sdvy / self.n if self.n > 0 else 0.0

    @property
    def k(self):
        return self.Svynvy / self.Svy2 if self.Svy2 > 1e-9 else 1.0

    def residuals(self):
        if self.n == 0:
            return dict(const_acc=1.0, const_vel=1.0, damp=1.0)
        th, k = self.theta, self.k
        # E[(dvy-th)^2] = E[dvy^2] - th^2 ; E[dvy^2] = (Svyn2 - 2 Svynvy + Svy2)/n
        Sdvy2 = self.Svyn2 - 2 * self.Svynvy + self.Svy2
        r_acc = (Sdvy2 / self.n) - th * th * (self.n / self.n)   # var of dvy around theta
        r_vel = Sdvy2 / self.n
        # damp residual E[(vyn - k vy)^2] = (Svyn2 - 2k Svynvy + k^2 Svy2)/n
        r_damp = (self.Svyn2 - 2 * k * self.Svynvy + k * k * self.Svy2) / self.n
        return dict(const_acc=max(r_acc, 0.0), const_vel=r_vel, damp=max(r_damp, 0.0))

    def survivor(self):
        r = self.residuals()
        return min(r, key=r.get), r

    @property
    def bounce_ok(self):
        return self.bounce_seen > 0 and (self.bounce_reflect / self.bounce_seen) > 0.5


@torch.no_grad()
def predict_landing_discovered(bx, by, bvx, bvy, gram, x_plane, horizon=90):
    """Roll out the DISCOVERED law (survivor form + discovered bounce)."""
    form, _ = gram.survivor()
    th, k = gram.theta, gram.k
    bounce = gram.bounce_ok
    bx, by, bvx, bvy = bx.clone(), by.clone(), bvx.clone(), bvy.clone()
    landing = by.clone(); arrived = bx >= x_plane
    for _ in range(horizon):
        if form == "const_acc":
            bvy = bvy + th
        elif form == "damp":
            bvy = bvy * k
        # const_vel: bvy unchanged
        bx = bx + bvx
        by = by + bvy
        if bounce:
            lo, hi = by < 0, by > 1
            by = torch.where(lo, -by, torch.where(hi, 2 - by, by))
            bvy = torch.where(lo | hi, -bvy, bvy)
        newly = (~arrived) & (bx >= x_plane)
        landing = torch.where(newly, by, landing)
        arrived = arrived | (bx >= x_plane)
    return landing


@torch.no_grad()
def act_toward(target_y, cy, cs):
    diff = target_y - cy
    a = torch.zeros_like(cy, dtype=torch.long)
    a = torch.where(diff > cs * 0.5, torch.ones_like(a), a)
    a = torch.where(diff < -cs * 0.5, torch.full_like(a, 2), a)
    return a


@torch.no_grad()
def eval_catch(gram, cfg, episodes, mode, seed=999):
    env = DeviceVecProjectileCatch(cfg["num_envs"], gravity=cfg["g"], max_steps=cfg["max_steps"],
                                   x_plane=cfg["x_plane"], seed=seed)
    N = cfg["num_envs"]; dc = torch.zeros(N, device=DEVICE)
    while float(dc.min()) < episodes:
        if mode == "random":
            a = torch.randint(0, 3, (N,), device=DEVICE)
        elif mode == "oracle":
            a = act_toward(env._landing(), env.cy, env.cs)
        else:
            a = act_toward(predict_landing_discovered(env.bx, env.by, env.bvx, env.bvy, gram,
                                                      cfg["x_plane"]), env.cy, env.cs)
        _, _, _, _, done = env.step(a); dc += done.float()
    return env.catch_rate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--gravity", type=float, default=0.004)
    p.add_argument("--max-steps", type=int, default=70)
    p.add_argument("--x-plane", type=float, default=0.97)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_episodes = 1800, 128, 600, 12

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = dict(num_envs=args.num_envs, g=args.gravity, max_steps=args.max_steps, x_plane=args.x_plane)
    gram = FormGrammar()
    env = DeviceVecProjectileCatch(args.num_envs, gravity=args.gravity, max_steps=args.max_steps,
                                   x_plane=args.x_plane, seed=args.seed)
    N = args.num_envs
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.2b] device={DEVICE} | DISCOVER the law FORM by falsification (const-vel / "
          f"const-acc / damp) + bounce | g_true={args.gravity} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    rnd = eval_catch(gram, cfg, args.eval_episodes, "random")
    orc = eval_catch(gram, cfg, args.eval_episodes, "oracle")
    print(f"  random {rnd:.2f} | oracle {orc:.2f}", flush=True)

    curve = []
    for step in range(1, args.steps + 1):
        tgt = predict_landing_discovered(env.bx, env.by, env.bvx, env.bvy, gram, args.x_plane)
        a = act_toward(tgt, env.cy, env.cs)
        expl = torch.rand(N, device=DEVICE) < args.epsilon
        a = torch.where(expl, torch.randint(0, 3, (N,), device=DEVICE), a)
        vy_before = env.bvy.clone(); by_before = env.by.clone()
        env.step(a)
        gram.observe(vy_before, env.bvy, by_before)
        if step % args.eval_every == 0 or step == args.steps:
            form, res = gram.survivor()
            c = eval_catch(gram, cfg, args.eval_episodes, "law")
            curve.append(dict(step=step, catch=round(c, 3), form=form,
                              residuals={k: round(v, 6) for k, v in res.items()},
                              theta=round(gram.theta, 5), bounce=gram.bounce_ok))
            print(f"  step {step:>5} | DISCOVERED form='{form}' theta={gram.theta:.5f} "
                  f"bounce={gram.bounce_ok} | catch {c:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}) | "
                  f"res={ {k: round(v,5) for k,v in res.items()} } | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    last = curve[-1]
    discovered_gravity = (last["form"] == "const_acc" and last["theta"] < 0 and last["bounce"])
    ok = discovered_gravity and last["catch"] > 1.5 * rnd and last["catch"] > 0.7 * orc
    verdict = (
        f"LITTLE SCIENTIST (r0.2b) — the agent DISCOVERED the law form by falsification: "
        f"'{last['form']}' (theta={last['theta']}, bounce={last['bounce']}) won over const-vel/damp, "
        f"and intercepts {last['catch']:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}). The gravity+bounce "
        f"INVARIANT was discovered, not given. Next: REUSE it across gravities vs fair PPO."
        if ok else
        f"PARTIAL/CHECK — form='{last['form']}' theta={last['theta']} bounce={last['bounce']} "
        f"catch {last['catch']:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}). residuals {last['residuals']}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r02b.json"), "w") as f:
        json.dump(dict(random=rnd, oracle=orc, curve=curve, discovered_gravity=discovered_gravity,
                       ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
