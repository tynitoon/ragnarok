"""r0.3 of the REFUTATION ENGINE — discover the gravity LAW FROM PIXELS.

Removes the state-perception hand-engineering of r0.2: the agent now perceives the ball
(x,y) and the catcher (y) from PIXELS via colour-blob centroids, estimates velocity from
frame deltas, and runs the SAME falsification form-discovery (const-vel / const-acc / damp
+ reflective bounce) on the pixel-extracted trajectory. It then rolls out the discovered
law to predict the landing and intercepts. The North-Star-relevant test: does the little
scientist still discover gravity + control when its quantities come from raw pixels (with
quantisation noise)?

Usage: python -m scripts.ree_r03_pixels [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.projectile import DeviceVecProjectileCatch
from scripts.ree_r02b_discover import FormGrammar, predict_landing_discovered


def perceive(frame, img):
    """(N,3,img,img) -> ball (bx,by) [red blob] and catcher cy [green blob], in [0,1]."""
    xs = torch.arange(img, device=DEVICE).float()
    ys = torch.arange(img, device=DEVICE).float()
    red = (frame[:, 0] - torch.maximum(frame[:, 1], frame[:, 2])).clamp(min=0)   # (N,img,img)
    rm = red.sum((-1, -2)).clamp(min=1e-3)
    bx = (red.sum(-2) * xs).sum(-1) / rm / (img - 1)
    by = (red.sum(-1) * ys).sum(-1) / rm / (img - 1)
    green = (frame[:, 1] - torch.maximum(frame[:, 0], frame[:, 2])).clamp(min=0)
    gm = green.sum((-1, -2)).clamp(min=1e-3)
    cy = (green.sum(-1) * ys).sum(-1) / gm / (img - 1)
    return bx, by, cy


def act_toward(t, cy, dead=0.02):
    d = t - cy
    a = torch.zeros_like(cy, dtype=torch.long)
    a = torch.where(d > dead, torch.ones_like(a), a)
    a = torch.where(d < -dead, torch.full_like(a, 2), a)
    return a


@torch.no_grad()
def eval_catch(gram, cfg, episodes, mode, seed=999):
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
            bvx, bvy = bx0 - bxm, by0 - bym
            tgt = predict_landing_discovered(bx0, by0, bvx, bvy, gram, cfg["xp"])
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
    p.add_argument("--xp", type=float, default=0.92)         # agent's landing-trigger plane (pixel)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_episodes = 2000, 128, 600, 10

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = dict(ne=args.num_envs, g=args.gravity, ms=args.max_steps, xp_env=args.xp_env,
               xp=args.xp, img=args.img)
    gram = FormGrammar()
    env = DeviceVecProjectileCatch(args.num_envs, gravity=args.gravity, max_steps=args.max_steps,
                                   x_plane=args.xp_env, img=args.img, seed=args.seed)
    N, img = args.num_envs, args.img
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.3] device={DEVICE} | DISCOVER gravity law FROM PIXELS ({img}px, colour-blob "
          f"perception) + intercept | g_true={args.gravity} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    rnd = eval_catch(gram, cfg, args.eval_episodes, "random")
    orc = eval_catch(gram, cfg, args.eval_episodes, "oracle")
    print(f"  random {rnd:.2f} | oracle {orc:.2f}", flush=True)

    bxm, bym, _ = perceive(env.state.view(N, 3, img, img), img)
    env.step(torch.randint(0, 3, (N,), device=DEVICE))
    bx0, by0, cy = perceive(env.state.view(N, 3, img, img), img)
    curve = []
    for step in range(1, args.steps + 1):
        bvx, bvy = bx0 - bxm, by0 - bym
        tgt = predict_landing_discovered(bx0, by0, bvx, bvy, gram, args.xp)
        a = act_toward(tgt, cy)
        expl = torch.rand(N, device=DEVICE) < args.epsilon
        a = torch.where(expl, torch.randint(0, 3, (N,), device=DEVICE), a)
        env.step(a)
        bx1, by1, cy = perceive(env.state.view(N, 3, img, img), img)
        gram.observe(by0 - bym, by1 - by0, by0)               # falsify forms on PIXEL trajectory
        bxm, bym, bx0, by0 = bx0, by0, bx1, by1
        if step % args.eval_every == 0 or step == args.steps:
            form, res = gram.survivor()
            c = eval_catch(gram, cfg, args.eval_episodes, "law")
            curve.append(dict(step=step, catch=round(c, 3), form=form, theta=round(gram.theta, 5),
                              bounce=gram.bounce_ok))
            print(f"  step {step:>5} | form='{form}' theta={gram.theta:.5f} bounce={gram.bounce_ok} "
                  f"| catch {c:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}) | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    last = curve[-1]
    discovered = last["form"] == "const_acc" and last["theta"] < 0
    ok = discovered and last["catch"] > 1.5 * rnd and last["catch"] > 0.6 * orc
    verdict = (
        f"GRAVITY LAW FROM PIXELS (r0.3) — from raw {img}px frames (colour-blob perception), the "
        f"agent DISCOVERED form='{last['form']}' theta={last['theta']} bounce={last['bounce']} by "
        f"falsification and intercepts {last['catch']:.2f} (rnd {rnd:.2f}, oracle {orc:.2f}). The "
        f"little scientist discovers + uses a physical law FROM PIXELS. Honest: grammar still given."
        if ok else
        f"PARTIAL/CHECK — form='{last['form']}' theta={last['theta']} catch {last['catch']:.2f} "
        f"(rnd {rnd:.2f}, oracle {orc:.2f}). Pixel noise may blur the form; tune perception/xp.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r03.json"), "w") as f:
        json.dump(dict(random=rnd, oracle=orc, curve=curve, discovered=discovered, ok=ok,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
