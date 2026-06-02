"""v0.3a of the NOTION GRAPH — the agent ACTS THROUGH its notions (control).

On Catcher (move a paddle to catch falling fruit). The agent has NO policy network:
at each step it PLANS by predicting, for each candidate action, the next frame via
its notion library (context-bound), decodes paddle-x vs fruit-x from the predicted
frame, and picks the action that best brings the paddle under the fruit. Control
quality is therefore ENTIRELY a function of the notions' predictive quality. It
learns the notions online from its own transitions (mint on surprise, prune on disuse).

Mechanism success (v0.3a): the notion-agent CATCHES far more than random actions, and
its catch rate RISES as the notions are learned -> the agent genuinely acts through its
notions. (Cross-task control reuse, fair converged baseline = v0.3b.)

Usage: python -m scripts.ng_v03_control [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.catcher import DeviceVecCatcher
from ragnarok.learning.notion_graph import NotionLibrary, patchify, unpatchify


def decode(frame, pad_row):
    """frame (N,3,H,W) -> (paddle_x, fruit_x) in pixel columns. Paddle=white
    (min over channels) in the bottom band; fruit=red (R minus max(G,B))."""
    H, W = frame.shape[-2:]
    xs = torch.arange(W, device=DEVICE).float()
    band = frame[:, :, max(0, pad_row - 2):pad_row + 3, :]
    white = band.min(1).values.sum(1)                              # (N,W)
    paddle_x = (white * xs).sum(-1) / white.sum(-1).clamp(min=1e-3)
    red = (frame[:, 0] - torch.maximum(frame[:, 1], frame[:, 2])).clamp(min=0).sum(1)
    fruit_x = (red * xs).sum(-1) / red.sum(-1).clamp(min=1e-3)
    return paddle_x, fruit_x


@torch.no_grad()
def plan(lib, prev, cur, cfg):
    """Greedy action per env: predict next frame for each action via notions,
    pick the one minimising predicted |paddle_x - fruit_x|."""
    N, img, patch, A = cfg["num_envs"], cfg["img"], cfg["patch"], 3
    pv, gh, gw = patchify(prev, patch)
    cv, _, _ = patchify(cur, patch)
    G = gh * gw
    ar = torch.arange(N, device=DEVICE)
    scores = []
    for a in range(A):
        aoh = torch.zeros(N, A, device=DEVICE)
        aoh[:, a] = 1.0
        ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, cfg["ctx_dim"])
        pred, _ = lib.predict(ctx)                                 # (P, patch_dim)
        frame = unpatchify(pred.reshape(N, G, cfg["P_dim"]), gh, gw, patch)
        px, fx = decode(frame, cfg["pad_row"])
        scores.append(-((px - fx) ** 2))
    return torch.stack(scores, -1).argmax(-1)                      # (N,)


def learn_step(lib, prev, cur, act, nxt, cfg):
    N, patch, A = cfg["num_envs"], cfg["patch"], 3
    pv, gh, gw = patchify(prev, patch)
    cv, _, _ = patchify(cur, patch)
    nx, _, _ = patchify(nxt, patch)
    G = gh * gw
    ar = torch.arange(N, device=DEVICE)
    aoh = torch.zeros(N, A, device=DEVICE)
    aoh[ar, act] = 1.0
    ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, cfg["ctx_dim"])
    target = nx.reshape(-1, cfg["P_dim"])
    return lib.learn(ctx, target)


@torch.no_grad()
def eval_catch(lib, cfg, steps, random_act=False, seed=999):
    env = DeviceVecCatcher(cfg["num_envs"], img=cfg["img"], seed=seed)
    N, A = cfg["num_envs"], 3
    def frame():
        return env.state.view(N, 3, cfg["img"], cfg["img"])
    prev = frame()
    env.step(torch.randint(0, A, (N,), device=DEVICE))
    cur = frame()
    for _ in range(steps):
        if random_act:
            act = torch.randint(0, A, (N,), device=DEVICE)
        else:
            act = plan(lib, prev, cur, cfg)
        env.step(act)
        prev, cur = cur, frame()
    return float(env.cum_catch.mean())                             # catches per env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--k-max", type=int, default=24)
    p.add_argument("--epsilon", type=float, default=0.25)          # explore to learn dynamics
    p.add_argument("--mint-every", type=int, default=40)
    p.add_argument("--mint-tol", type=float, default=0.010)
    p.add_argument("--prune-every", type=int, default=200)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.eval_every, args.eval_steps = 1200, 64, 400, 300

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    P_dim = 3 * args.patch * args.patch
    cfg = dict(num_envs=args.num_envs, img=args.img, patch=args.patch, P_dim=P_dim,
               ctx_dim=2 * P_dim + 3, pad_row=int(0.88 * (args.img - 1)),
               mint_every=args.mint_every, mint_tol=args.mint_tol, prune_every=args.prune_every)
    lib = NotionLibrary(cfg["ctx_dim"], P_dim, hidden=args.hidden, k_init=1, k_max=args.k_max)
    env = DeviceVecCatcher(args.num_envs, img=args.img, seed=args.seed)
    A = 3
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[NG v0.3a] device={DEVICE} | ACT THROUGH NOTIONS on Catcher | the agent has NO "
          f"policy: it plans 1-step through its notion library | steps {args.steps}", flush=True)
    t0 = time.perf_counter()

    rnd_catch = eval_catch(lib, cfg, args.eval_steps, random_act=True)   # random baseline
    print(f"  random-action baseline catch: {rnd_catch:.2f} per env / {args.eval_steps} steps", flush=True)

    def frame():
        return env.state.view(args.num_envs, 3, args.img, args.img)
    prev = frame()
    env.step(torch.randint(0, A, (args.num_envs,), device=DEVICE))
    cur = frame()
    topema, curve = 0.02, []
    for step in range(1, args.steps + 1):
        act = plan(lib, prev, cur, cfg)
        rnd = torch.rand(args.num_envs, device=DEVICE) < args.epsilon
        act = torch.where(rnd, torch.randint(0, A, (args.num_envs,), device=DEVICE), act)
        env.step(act)
        nxt = frame()
        loss, min_err, assign = learn_step(lib, prev, cur, act, nxt, cfg)
        with torch.no_grad():
            topk = max(1, min_err.numel() // 10)
            topema = 0.9 * topema + 0.1 * float(min_err.topk(topk).values.mean())
        if step > args.warmup and step % cfg["mint_every"] == 0 and topema > cfg["mint_tol"] \
                and lib.K < lib.k_max:
            pv, gh, gw = patchify(prev, args.patch)
            cv, _, _ = patchify(cur, args.patch)
            G = gh * gw
            aoh = torch.zeros(args.num_envs, A, device=DEVICE)
            aoh[torch.arange(args.num_envs, device=DEVICE), act] = 1.0
            ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, cfg["ctx_dim"])
            ti = min_err.topk(topk).indices
            lib.mint(ctx_seed=ctx[ti], target_seed=patchify(nxt, args.patch)[0].reshape(-1, P_dim)[ti])
        if step > args.warmup and step % cfg["prune_every"] == 0:
            lib.prune(1e-3)
        prev, cur = cur, nxt
        if step % args.eval_every == 0 or step == args.steps:
            c = eval_catch(lib, cfg, args.eval_steps, random_act=False)
            curve.append(dict(step=step, catch=round(c, 2), K=lib.K))
            print(f"  step {step:>5} | notion-agent catch {c:.2f} (random {rnd_catch:.2f}) | "
                  f"K={lib.K} | {time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]["catch"]
    rose = len(curve) >= 2 and curve[-1]["catch"] > curve[0]["catch"]
    ok = final > 1.5 * rnd_catch and rose
    verdict = (
        f"ACTS THROUGH NOTIONS (v0.3a) — the notion-agent catches {final:.2f} vs random "
        f"{rnd_catch:.2f} ({final/max(rnd_catch,0.1):.1f}x), rising as notions are learned. "
        f"Control flows entirely through the notion library, from pixels. Next: v0.3b control reuse."
        if ok else
        f"PARTIAL/CHECK — final catch {final:.2f} vs random {rnd_catch:.2f}, rose={rose}, K={final and curve[-1]['K']}. "
        f"Tune planning/decoder/explore.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ng_v03_control.json"), "w") as f:
        json.dump(dict(random_catch=rnd_catch, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
