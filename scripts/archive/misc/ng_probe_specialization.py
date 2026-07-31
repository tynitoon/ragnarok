"""ADVERSARIAL PROBE: is reuse GENUINE (frozen Pong notions already predict the new
world's dynamics) or is it RETRAINING (the slot exists, gets re-fit on world2)?

Train a library on Pong. Then, on world2 (breakout/snake), with the library FROZEN
(no gradient steps, no mint), measure dyn on changed patches = TRUE ZERO-SHOT reuse.
Compare to:
  * a FROZEN random library (same K, random weights)  -> floor
  * persist baseline (copy current frame)             -> trivial floor
  * the trained-on-world2 number from the main exp    -> ceiling
If frozen-Pong dyn << frozen-random and << persist, the Pong notions genuinely
transfer. If frozen-Pong ~ frozen-random ~ persist, 'reuse' is just retraining a
pre-sized library and the knowledge does NOT transfer.

Also reports the notion-binding histogram on world2: how many DISTINCT notions carry
>5% of changed-patch assignments (is the library actually composing, or 1 notion?).

Usage: python -m scripts.ng_probe_specialization --world2 breakout
"""
import argparse
import torch
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake
from ragnarok.learning.notion_graph import NotionLibrary, patchify

AMAX = 4
N, IMG, P = 128, 48, 8


def make(name, seed):
    if name == "pong":
        return DeviceVecPong(N, img=IMG, seed=seed), 3
    if name == "breakout":
        return DeviceVecBreakout(N, img=IMG, seed=seed), 3
    if name == "snake":
        return DeviceVecSnake(N, img=IMG, seed=seed), 4


def new_lib(P_dim, hidden, k_max):
    return NotionLibrary(P_dim * 2 + AMAX, P_dim, hidden=hidden, k_init=1, k_max=k_max)


def train(lib, env, A, steps, mint_tol=0.010, mint_every=40, prune_every=200, kmax=16):
    ar = torch.arange(N, device=DEVICE)
    P_dim = lib.patch_dim

    def frame():
        return env.state.view(N, 3, IMG, IMG)
    prev = frame(); env.step(torch.randint(0, A, (N,), device=DEVICE)); cur = frame()
    topema = 0.02
    for step in range(1, steps + 1):
        act = torch.randint(0, A, (N,), device=DEVICE)
        env.step(act); nxt = frame()
        pv, gh, gw = patchify(prev, P); cv, _, _ = patchify(cur, P); nx, _, _ = patchify(nxt, P)
        G = gh * gw
        aoh = torch.zeros(N, AMAX, device=DEVICE); aoh[ar, act] = 1.0
        ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, lib.ctx_dim)
        target = nx.reshape(-1, P_dim)
        loss, min_err, assign = lib.learn(ctx, target)
        with torch.no_grad():
            topk = max(1, min_err.numel() // 10)
            topema = 0.9 * topema + 0.1 * float(min_err.topk(topk).values.mean())
        if step % mint_every == 0 and topema > mint_tol and lib.K < kmax:
            ti = min_err.topk(topk).indices
            lib.mint(target_seed=target[ti])
        if step % prune_every == 0:
            lib.prune(1e-3)
        prev, cur = cur, nxt
    return lib


@torch.no_grad()
def frozen_eval(lib, env, A, steps=600):
    """No learning, no mint. Return dyn on changed patches + binding histogram."""
    ar = torch.arange(N, device=DEVICE)
    P_dim = lib.patch_dim

    def frame():
        return env.state.view(N, 3, IMG, IMG)
    prev = frame(); env.step(torch.randint(0, A, (N,), device=DEVICE)); cur = frame()
    sum_err, n = 0.0, 0
    hist = torch.zeros(lib.K, device=DEVICE)
    for step in range(steps):
        act = torch.randint(0, A, (N,), device=DEVICE)
        env.step(act); nxt = frame()
        pv, gh, gw = patchify(prev, P); cv, _, _ = patchify(cur, P); nx, _, _ = patchify(nxt, P)
        G = gh * gw
        aoh = torch.zeros(N, AMAX, device=DEVICE); aoh[ar, act] = 1.0
        ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, lib.ctx_dim)
        target = nx.reshape(-1, P_dim)
        preds = lib.predict_all(ctx)
        err = ((preds - target.unsqueeze(1)) ** 2).mean(-1)
        min_err, assign = err.min(1)
        changed = ((nx - cv) ** 2).mean(-1).reshape(-1) > 1e-5
        if bool(changed.any()):
            sum_err += float(min_err[changed].sum()); n += int(changed.sum())
            hist.index_add_(0, assign[changed], torch.ones_like(assign[changed], dtype=torch.float))
        prev, cur = cur, nxt
    hist = hist / hist.sum().clamp(min=1)
    return sum_err / max(1, n), hist.cpu()


@torch.no_grad()
def persist_dyn(env, A, steps=600):
    def frame():
        return env.state.view(N, 3, IMG, IMG)
    prev = frame(); env.step(torch.randint(0, A, (N,), device=DEVICE)); cur = frame()
    s, n = 0.0, 0
    for _ in range(steps):
        env.step(torch.randint(0, A, (N,), device=DEVICE)); nxt = frame()
        cv, gh, gw = patchify(cur, P); nx, _, _ = patchify(nxt, P)
        changed = ((nx - cv) ** 2).mean(-1).reshape(-1) > 1e-5
        nxf = nx.reshape(-1, nx.shape[-1]); cvf = cv.reshape(-1, cv.shape[-1])
        if bool(changed.any()):
            s += float(((nxf - cvf) ** 2).mean(-1)[changed].sum()); n += int(changed.sum())
        cur = nxt
    return s / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world2", default="breakout", choices=["breakout", "snake"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pong-steps", type=int, default=3000)
    args = ap.parse_args()
    P_dim = 3 * P * P
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) train on Pong
    lib = new_lib(P_dim, 64, 16)
    pong, Ap = make("pong", args.seed)
    train(lib, pong, Ap, args.pong_steps)
    K = lib.K
    print(f"[spec] world2={args.world2} seed={args.seed}  K_after_pong={K}")

    # 2) FROZEN Pong library on world2 (true zero-shot)
    w2a, A2 = make(args.world2, args.seed + 100)
    frozen_pong_dyn, hist_pong = frozen_eval(lib, w2a, A2)

    # 3) FROZEN random library, same K
    rlib = new_lib(P_dim, 64, 16)
    while rlib.K < K:
        rlib.mint()
    w2b, _ = make(args.world2, args.seed + 100)
    frozen_rand_dyn, _ = frozen_eval(rlib, w2b, A2)

    # 4) persist floor + frozen Pong library on PONG itself (transfer-from ceiling)
    w2c, _ = make(args.world2, args.seed + 100)
    persist = persist_dyn(w2c, A2)
    pong_self, _ = frozen_eval(lib, make("pong", args.seed + 100)[0], Ap)

    print(f"  ZERO-SHOT dyn on {args.world2} (changed patches):")
    print(f"    frozen Pong-lib   : {frozen_pong_dyn:.5f}   <- if << others, real transfer")
    print(f"    frozen RANDOM-lib : {frozen_rand_dyn:.5f}   (same K, no knowledge)")
    print(f"    persist (copy)    : {persist:.5f}   (trivial)")
    print(f"    [ref] frozen Pong-lib ON PONG : {pong_self:.5f}  (its home turf)")
    top = (hist_pong > 0.05).sum().item()
    print(f"  binding on {args.world2}: {top}/{K} notions carry >5% of changed patches; "
          f"top-3 shares {sorted(hist_pong.tolist(), reverse=True)[:3]}")


if __name__ == "__main__":
    main()
