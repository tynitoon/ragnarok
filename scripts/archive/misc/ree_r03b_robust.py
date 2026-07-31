"""r0.3b — noise-ROBUST gravity discovery from pixels (trajectory parabola fit).

r0.3 showed control-from-pixels works but per-step falsification picks the WRONG form
(damp) because pixel quantisation noise (~0.02) swamps the true per-step Delta(vy) (~0.004).
Fix: discover the law over a TRAJECTORY WINDOW. Fit a parabola y=a+b*t+c*t^2 to the last W
perceived ball-y's; the cumulative curvature (0.5*g*W^2) is LARGE vs the noise, so it robustly
(a) detects acceleration (quadratic residual << linear residual = const-acc, not const-vel) and
(b) recovers g=-2c. The agent then rolls out the const-acc law (+reflective bounce) and intercepts.

Success: from raw pixels, quadratic beats linear (acceleration discovered), g_est ~ true g, and
catch ~ oracle. The little scientist correctly identifies gravity FROM PIXELS, noise and all.

Usage: python -m scripts.ree_r03b_robust [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.projectile import DeviceVecProjectileCatch
from scripts.ree_r03_pixels import perceive, act_toward


class TrajGravity:
    """Windowed parabola fit: robustly detect acceleration + recover g from pixel trajectory."""
    def __init__(self, W=12):
        self.W = W
        t = torch.arange(W, device=DEVICE).float()
        Mq = torch.stack([torch.ones(W, device=DEVICE), t, t * t], 1)     # (W,3)
        Ml = torch.stack([torch.ones(W, device=DEVICE), t], 1)            # (W,2)
        self.pinv_q = torch.linalg.pinv(Mq)                               # (3,W)
        self.Pq = Mq @ self.pinv_q                                        # (W,W) projector
        self.Pl = Ml @ torch.linalg.pinv(Ml)
        self.crow = self.pinv_q[2]                                        # curvature extractor
        self.buf = None; self.fill = 0
        self.Sc = 0.0; self.n = 0.0; self.Rq = 0.0; self.Rl = 0.0

    def push(self, by):
        if self.buf is None:
            self.buf = by.unsqueeze(1).repeat(1, self.W)
            self.fill = 1
            return
        self.buf = torch.cat([self.buf[:, 1:], by.unsqueeze(1)], 1)
        self.fill = min(self.fill + 1, self.W)
        if self.fill < self.W:
            return
        rng = self.buf.max(1).values - self.buf.min(1).values
        valid = rng < 0.5                                                  # exclude reset/bounce windows
        if not bool(valid.any()):
            return
        y = self.buf[valid]                                                # (M,W)
        c = (y * self.crow).sum(1)                                         # curvature
        rq = ((y - y @ self.Pq.T) ** 2).sum(1)                            # quadratic residual
        rl = ((y - y @ self.Pl.T) ** 2).sum(1)                            # linear residual
        self.Sc += float(c.sum()); self.n += c.numel()
        self.Rq += float(rq.sum()); self.Rl += float(rl.sum())

    @property
    def g(self):
        return -2.0 * (self.Sc / self.n) if self.n > 0 else 0.0           # c = -g/2

    @property
    def accel_discovered(self):                                           # quad << linear residual
        return self.n > 0 and (self.Rq / max(self.Rl, 1e-12)) < 0.5


@torch.no_grad()
def rollout_landing(bx, by, bvx, bvy, theta, x_plane, horizon=80):
    bx, by, bvx, bvy = bx.clone(), by.clone(), bvx.clone(), bvy.clone()
    land = by.clone(); arr = bx >= x_plane
    for _ in range(horizon):
        bvy = bvy + theta
        bx = bx + bvx
        by = by + bvy
        lo, hi = by < 0, by > 1
        by = torch.where(lo, -by, torch.where(hi, 2 - by, by))
        bvy = torch.where(lo | hi, -bvy, bvy)
        newly = (~arr) & (bx >= x_plane)
        land = torch.where(newly, by, land)
        arr = arr | (bx >= x_plane)
    return land


@torch.no_grad()
def eval_catch(tg, cfg, episodes, mode, seed=999):
    env = DeviceVecProjectileCatch(cfg["ne"], gravity=cfg["g"], max_steps=cfg["ms"],
                                   x_plane=cfg["xp_env"], img=cfg["img"], seed=seed)
    N, img = cfg["ne"], cfg["img"]
    dc = torch.zeros(N, device=DEVICE)
    bxm, bym, _ = perceive(env.state.view(N, 3, img, img), img)
    env.step(torch.randint(0, 3, (N,), device=DEVICE))
    bx0, by0, cy = perceive(env.state.view(N, 3, img, img), img)
    while float(dc.min()) < episodes:
        if mode == "random":
            a = torch.randint(0, 3, (N,), device=DEVICE)
        elif mode == "oracle":
            a = act_toward(env._landing(), env.cy)
        else:
            tgt = rollout_landing(bx0, by0, bx0 - bxm, by0 - bym, tg.g, cfg["xp"])
            a = act_toward(tgt, cy)
        _, _, _, _, done = env.step(a)
        dc += done.float()
        bxm, bym = bx0, by0
        bx0, by0, cy = perceive(env.state.view(N, 3, img, img), img)
    return env.catch_rate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--gravity", type=float, default=0.004)
    p.add_argument("--max-steps", type=int, default=70)
    p.add_argument("--xp-env", type=float, default=0.97)
    p.add_argument("--xp", type=float, default=0.92)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_episodes = 2200, 128, 700, 10

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = dict(ne=args.num_envs, g=args.gravity, ms=args.max_steps, xp_env=args.xp_env,
               xp=args.xp, img=args.img)
    tg = TrajGravity(W=args.window)
    env = DeviceVecProjectileCatch(args.num_envs, gravity=args.gravity, max_steps=args.max_steps,
                                   x_plane=args.xp_env, img=args.img, seed=args.seed)
    N, img = args.num_envs, args.img
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.3b] device={DEVICE} | noise-ROBUST gravity from PIXELS (trajectory parabola "
          f"fit, W={args.window}) | g_true={args.gravity} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    rnd = eval_catch(tg, cfg, args.eval_episodes, "random")
    orc = eval_catch(tg, cfg, args.eval_episodes, "oracle")
    print(f"  random {rnd:.2f} | oracle {orc:.2f}", flush=True)

    bxm, bym, _ = perceive(env.state.view(N, 3, img, img), img)
    env.step(torch.randint(0, 3, (N,), device=DEVICE))
    bx0, by0, cy = perceive(env.state.view(N, 3, img, img), img)
    curve = []
    for step in range(1, args.steps + 1):
        tgt = rollout_landing(bx0, by0, bx0 - bxm, by0 - bym, tg.g, args.xp)
        a = act_toward(tgt, cy)
        expl = torch.rand(N, device=DEVICE) < args.epsilon
        a = torch.where(expl, torch.randint(0, 3, (N,), device=DEVICE), a)
        env.step(a)
        bx1, by1, cy = perceive(env.state.view(N, 3, img, img), img)
        tg.push(by0)
        bxm, bym, bx0, by0 = bx0, by0, bx1, by1
        if step % args.eval_every == 0 or step == args.steps:
            c = eval_catch(tg, cfg, args.eval_episodes, "law")
            curve.append(dict(step=step, catch=round(c, 3), g_est=round(tg.g, 5),
                              accel=tg.accel_discovered, res_ratio=round(tg.Rq / max(tg.Rl, 1e-12), 3)))
            print(f"  step {step:>5} | accel_discovered={tg.accel_discovered} g_est={tg.g:.5f} "
                  f"(true {args.gravity}) quad/lin_res={tg.Rq/max(tg.Rl,1e-12):.3f} | catch {c:.2f} "
                  f"(rnd {rnd:.2f}, oracle {orc:.2f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    last = curve[-1]
    ok = (last["accel"] and abs(last["g_est"] - args.gravity) < 0.0015
          and last["catch"] > 1.5 * rnd and last["catch"] > 0.7 * orc)
    verdict = (
        f"GRAVITY FROM PIXELS, ROBUST (r0.3b) — via trajectory parabola fit the agent DISCOVERS "
        f"acceleration (quad/lin residual {last['res_ratio']}) and recovers g={last['g_est']} "
        f"(true {args.gravity}) FROM RAW PIXELS, intercepting {last['catch']:.2f} (rnd {rnd:.2f}, "
        f"oracle {orc:.2f}). The little scientist correctly identifies gravity from pixels."
        if ok else
        f"PARTIAL/CHECK — accel={last['accel']} g_est={last['g_est']} (true {args.gravity}) catch "
        f"{last['catch']:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}) res_ratio={last['res_ratio']}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r03b.json"), "w") as f:
        json.dump(dict(random=rnd, oracle=orc, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
