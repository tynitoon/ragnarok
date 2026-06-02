"""r0.1 of the REFUTATION ENGINE (REFUTATION_ENGINE_DESIGN.md) — OPTION 2.

The agent's knowledge is a REFUTABLE LAW, not a neural net. On Catcher it perceives
two quantities (paddle_x, fruit_x) from colour blobs, and holds one controllability
law:  Delta(paddle_x) = theta * action_sign  (action_sign: +1 right, -1 left, 0 stay).
It FITS theta online by least squares, tracks the law's RESIDUAL (its refutation
signal), and ACTS by applying the law: predict each action's paddle_x and pick the one
that brings the paddle under the fruit. No policy network, no gradient descent, no
growing mixture — so it cannot "collapse" like NG v0.3a; a law is fit-or-refuted, period.

Success (r0.1): catch rate rises to high AND is STABLE (the contrast with NG's collapse),
and the law's residual is tiny (the invariant holds). Sets up r0.2 = REUSE the law on
Pong/Breakout (same paddle controllability) -> immediate control.

Usage: python -m scripts.ree_r01_catcher [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.catcher import DeviceVecCatcher


def perceive(frame, H, W):
    """frame (N,3,H,W) -> (paddle_x, fruit_x) in [0,1] via colour-blob centroids."""
    xs = torch.arange(W, device=DEVICE).float()
    pad_row = int(0.88 * (H - 1))
    band = frame[:, :, max(0, pad_row - 2):pad_row + 3, :]
    white = band.min(1).values.sum(1)                              # (N,W) whiteness
    paddle_x = (white * xs).sum(-1) / white.sum(-1).clamp(min=1e-3) / (W - 1)
    red = (frame[:, 0] - torch.maximum(frame[:, 1], frame[:, 2])).clamp(min=0).sum(1)
    fruit_x = (red * xs).sum(-1) / red.sum(-1).clamp(min=1e-3) / (W - 1)
    return paddle_x, fruit_x


class ControllabilityLaw:
    """Delta(paddle_x) = theta * action_sign. theta fit by least-squares-through-origin;
    residual tracked as the refutation signal."""
    def __init__(self):
        self.Sxy = 0.0      # sum dpx * a_s
        self.Sxx = 0.0      # sum a_s^2
        self.Sres = 0.0     # sum residual^2
        self.Svar = 0.0     # sum dpx^2
        self.n = 0.0

    @property
    def theta(self):
        return self.Sxy / self.Sxx if self.Sxx > 1e-6 else 0.0

    def update(self, dpx, a_s):
        th = self.theta
        self.Sxy += float((dpx * a_s).sum())
        self.Sxx += float((a_s * a_s).sum())
        self.Sres += float(((dpx - th * a_s) ** 2).sum())
        self.Svar += float((dpx * dpx).sum())
        self.n += a_s.numel()

    @property
    def residual(self):                 # fraction of variance UNexplained (refutation signal)
        return self.Sres / self.Svar if self.Svar > 1e-6 else 1.0


@torch.no_grad()
def eval_catch(law, cfg, steps, random_act=False, seed=999):
    env = DeviceVecCatcher(cfg["num_envs"], img=cfg["img"], seed=seed)
    N, H, W = cfg["num_envs"], cfg["img"], cfg["img"]
    sign = torch.tensor([0.0, -1.0, 1.0], device=DEVICE)           # stay,left,right
    for _ in range(steps):
        if random_act:
            act = torch.randint(0, 3, (N,), device=DEVICE)
        else:
            px, fx = perceive(env.state.view(N, 3, H, W), H, W)
            th = law.theta
            pred = (px.unsqueeze(1) + th * sign.unsqueeze(0)).clamp(0, 1)   # (N,3)
            act = (-(pred - fx.unsqueeze(1)).abs()).argmax(1)
        env.step(act)
    return float(env.cum_catch.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_steps = 1500, 64, 300, 300

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    N, H, W = args.num_envs, args.img, args.img
    cfg = dict(num_envs=N, img=args.img)
    law = ControllabilityLaw()
    env = DeviceVecCatcher(N, img=args.img, seed=args.seed)
    sign = torch.tensor([0.0, -1.0, 1.0], device=DEVICE)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[REE r0.1] device={DEVICE} | knowledge = ONE refutable controllability LAW | "
          f"acts via the law, fit-or-refuted (no net, no collapse) | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    rnd = eval_catch(law, cfg, args.eval_steps, random_act=True)
    print(f"  random-action baseline catch: {rnd:.2f}", flush=True)

    px_prev, _ = perceive(env.state.view(N, 3, H, W), H, W)
    curve = []
    for step in range(1, args.steps + 1):
        px, fx = perceive(env.state.view(N, 3, H, W), H, W)
        th = law.theta
        pred = (px.unsqueeze(1) + th * sign.unsqueeze(0)).clamp(0, 1)
        act = (-(pred - fx.unsqueeze(1)).abs()).argmax(1)
        expl = torch.rand(N, device=DEVICE) < args.epsilon
        act = torch.where(expl, torch.randint(0, 3, (N,), device=DEVICE), act)
        env.step(act)
        px_new, _ = perceive(env.state.view(N, 3, H, W), H, W)
        law.update(px_new - px, sign[act])                         # fit the law online
        if step % args.eval_every == 0 or step == args.steps:
            c = eval_catch(law, cfg, args.eval_steps)
            curve.append(dict(step=step, catch=round(c, 2), theta=round(law.theta, 4),
                              residual=round(law.residual, 4)))
            print(f"  step {step:>5} | catch {c:.2f} (random {rnd:.2f}) | law theta="
                  f"{law.theta:.4f} residual={law.residual:.3f} | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    finals = [r["catch"] for r in curve[-3:]]
    final = curve[-1]["catch"]
    stable = len(finals) >= 2 and (max(finals) - min(finals)) <= 0.6 * max(final, 1)
    ok = final > 1.5 * rnd and stable and curve[-1]["residual"] < 0.3
    verdict = (
        f"REFUTATION-LAW CONTROL (r0.1) — the agent acts via ONE refutable controllability law "
        f"(theta={curve[-1]['theta']}, residual={curve[-1]['residual']}); it catches {final:.2f} vs "
        f"random {rnd:.2f} ({final/max(rnd,0.1):.1f}x) and is STABLE (no collapse, unlike NG v0.3a). "
        f"Next r0.2: REUSE the law on Pong/Breakout."
        if ok else
        f"PARTIAL/CHECK — catch {final:.2f} vs random {rnd:.2f}, stable={stable}, residual="
        f"{curve[-1]['residual']}. Tune perception/exploration.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ree_r01.json"), "w") as f:
        json.dump(dict(random_catch=rnd, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
