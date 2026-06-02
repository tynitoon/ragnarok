"""r0.2a of the REFUTATION ENGINE — discover & ACT THROUGH a gravity LAW (projectile).

The EXPENSIVE-invariant regime our v44 boundary says reuse should pay: a ball arcs to the
catcher plane under gravity (with wall bounces). The optimal action is to be at the LANDING
y, which a model-free agent must laboriously learn from sparse catch reward. The Refutation
agent instead DISCOVERS the gravity law  Delta(vy) = -g  (fit g online from the observed
velocity, refutation = residual), ROLLS OUT that law to predict the landing, and moves the
catcher there. Control flows entirely through the discovered law.

r0.2a (mechanism): does the law-agent intercept near the ORACLE (true-landing) and far above
random, with a tiny law residual? (State obs here to isolate law discovery+falsification+
control; pixels later.) r0.2b will REUSE the law across launch distributions vs from-scratch
and a fair model-free PPO at matched compute.

Usage: python -m scripts.ree_r02_gravity [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.projectile import DeviceVecProjectileCatch


class GravityLaw:
    """Delta(vy) = -g, fit online from observed vy on NON-bounce steps (a bounce flips
    vy -> large |Delta|, excluded). Residual = refutation signal."""
    def __init__(self):
        self.Sg = 0.0
        self.n = 0.0
        self.Sres = 0.0
        self.Svar = 0.0

    @property
    def g(self):
        return self.Sg / self.n if self.n > 0 else 0.0

    def update(self, dvy):
        keep = dvy.abs() < 0.03                       # exclude bounce steps (vy sign-flip)
        d = dvy[keep]
        if d.numel() == 0:
            return
        g = self.g
        self.Sg += float((-d).sum())
        self.n += d.numel()
        self.Sres += float(((d - (-g)) ** 2).sum())
        self.Svar += float((d * d).sum())

    @property
    def residual(self):
        return self.Sres / self.Svar if self.Svar > 1e-9 else 1.0


@torch.no_grad()
def predict_landing(bx, by, bvx, bvy, g, x_plane, horizon=160):
    """Roll out the DISCOVERED law to the catcher plane -> landing y (with wall bounces)."""
    bx, by, bvx, bvy = bx.clone(), by.clone(), bvx.clone(), bvy.clone()
    landing = by.clone()
    arrived = bx >= x_plane
    for _ in range(horizon):
        bvy = bvy - g
        bx = bx + bvx
        by = by + bvy
        lo, hi = by < 0, by > 1
        by = torch.where(lo, -by, torch.where(hi, 2 - by, by))
        bvy = torch.where(lo | hi, -bvy, bvy)
        newly = (~arrived) & (bx >= x_plane)
        landing = torch.where(newly, by, landing)
        arrived = arrived | (bx >= x_plane)
    return landing


@torch.no_grad()
def act_toward(target_y, cy, cs):
    """Move catcher toward target_y: 1 up / 2 down / 0 stay (deadzone ~ a step)."""
    diff = target_y - cy
    a = torch.zeros_like(cy, dtype=torch.long)
    a = torch.where(diff > cs * 0.5, torch.ones_like(a), a)
    a = torch.where(diff < -cs * 0.5, torch.full_like(a, 2), a)
    return a


@torch.no_grad()
def eval_catch(law, cfg, episodes, mode, seed=999):
    """mode: 'law' (discovered g rollout), 'oracle' (env analytic landing), 'random'."""
    env = DeviceVecProjectileCatch(cfg["num_envs"], gravity=cfg["g"], max_steps=cfg["max_steps"],
                                   x_plane=cfg["x_plane"], seed=seed)
    N = cfg["num_envs"]
    done_count = torch.zeros(N, device=DEVICE)
    while float(done_count.min()) < episodes:
        if mode == "random":
            a = torch.randint(0, 3, (N,), device=DEVICE)
        else:
            tgt = (env._landing() if mode == "oracle"
                   else predict_landing(env.bx, env.by, env.bvx, env.bvy, law.g, cfg["x_plane"]))
            a = act_toward(tgt, env.cy, env.cs)
        _, _, term, _, done = env.step(a)
        done_count += done.float()
    return env.catch_rate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--gravity", type=float, default=0.004)
    p.add_argument("--max-steps", type=int, default=70)
    p.add_argument("--x-plane", type=float, default=0.97)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_episodes = 1500, 128, 300, 12

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = dict(num_envs=args.num_envs, g=args.gravity, max_steps=args.max_steps,
               x_plane=args.x_plane)
    law = GravityLaw()
    env = DeviceVecProjectileCatch(args.num_envs, gravity=args.gravity, max_steps=args.max_steps,
                                   x_plane=args.x_plane, seed=args.seed)
    N = args.num_envs
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.2a] device={DEVICE} | discover & ACT THROUGH a gravity LAW (projectile "
          f"interception) | g_true={args.gravity} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    rnd = eval_catch(law, cfg, args.eval_episodes, "random")
    orc = eval_catch(law, cfg, args.eval_episodes, "oracle")
    print(f"  random catch {rnd:.2f} | ORACLE (true landing) catch {orc:.2f}", flush=True)

    prev_bvy = env.bvy.clone()
    curve = []
    for step in range(1, args.steps + 1):
        tgt = predict_landing(env.bx, env.by, env.bvx, env.bvy, law.g, args.x_plane)
        a = act_toward(tgt, env.cy, env.cs)
        expl = torch.rand(N, device=DEVICE) < args.epsilon
        a = torch.where(expl, torch.randint(0, 3, (N,), device=DEVICE), a)
        prev_bvy = env.bvy.clone()
        env.step(a)
        law.update(env.bvy - prev_bvy)                # fit/refute the gravity law online
        if step % args.eval_every == 0 or step == args.steps:
            c = eval_catch(law, cfg, args.eval_episodes, "law")
            curve.append(dict(step=step, catch=round(c, 3), g=round(law.g, 5),
                              residual=round(law.residual, 4)))
            print(f"  step {step:>5} | law catch {c:.2f} (random {rnd:.2f}, oracle {orc:.2f}) | "
                  f"g_est={law.g:.5f} (true {args.gravity}) residual={law.residual:.3f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]["catch"]
    finals = [r["catch"] for r in curve[-3:]]
    stable = len(finals) >= 2 and (max(finals) - min(finals)) <= 0.15
    ok = final > 1.5 * rnd and final > 0.7 * orc and stable and curve[-1]["residual"] < 0.2
    verdict = (
        f"GRAVITY-LAW CONTROL (r0.2a) — the agent DISCOVERS g={curve[-1]['g']} (true {args.gravity}, "
        f"residual {curve[-1]['residual']}) and intercepts by rolling out the law: catch {final:.2f} "
        f"vs random {rnd:.2f}, approaching ORACLE {orc:.2f}. Stable. The expensive landing-invariant "
        f"is captured as ONE refutable law. Next r0.2b: REUSE it vs from-scratch + fair PPO."
        if ok else
        f"PARTIAL/CHECK — law catch {final:.2f} (random {rnd:.2f}, oracle {orc:.2f}), stable={stable}, "
        f"g_est {curve[-1]['g']} residual {curve[-1]['residual']}. Tune.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r02a.json"), "w") as f:
        json.dump(dict(random=rnd, oracle=orc, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
