"""ADVERSARIAL PROBE of the NG v0.2 reuse claim.

Answers, with numbers from the actual envs:
 (A) What fraction of 8x8 patches actually CHANGE per step (Pong, Breakout, Snake)?
 (B) Is `dyn_mse` (error on changed patches) beatable by TRIVIAL baselines that have
     no understanding? Baselines:
        - persist  : predict next = current frame (copy; zero motion)
        - persist_prev: predict next = prev frame
     If the notion library's dyn (~0.004-0.006) is near or worse than 'persist' on
     changed patches, the metric is hollow.
 (C) How visually similar are Pong and Breakout vs Pong and Snake, at the patch level?
     Report mean per-pixel L2 between a Pong frame-distribution and each.

Usage: python -m scripts.ng_probe
"""
import torch
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake
from ragnarok.learning.notion_graph import patchify

torch.manual_seed(0)
N, IMG, P = 128, 48, 8


def make(env_name):
    if env_name == "pong":
        return DeviceVecPong(N, img=IMG, seed=0)
    if env_name == "breakout":
        return DeviceVecBreakout(N, img=IMG, seed=0)
    if env_name == "snake":
        return DeviceVecSnake(N, img=IMG, seed=0)


def frame(env):
    return env.state.view(N, 3, IMG, IMG)


def step(env, A):
    env.step(torch.randint(0, A, (N,), device=DEVICE))


@torch.no_grad()
def probe(env_name, A, steps=600):
    env = make(env_name)
    prev = frame(env).clone()
    step(env, A)
    cur = frame(env).clone()
    tot_patch = 0
    changed_patch = 0
    # accumulators for dyn-on-changed under trivial baselines
    sum_persist, sum_persist_prev, sum_zero, n_changed = 0.0, 0.0, 0.0, 0
    # foreground (non-black) pixel fraction
    fg_frac = 0.0
    nfr = 0
    for _ in range(steps):
        step(env, A)
        nxt = frame(env).clone()
        pv, gh, gw = patchify(prev, P)
        cv, _, _ = patchify(cur, P)
        nx, _, _ = patchify(nxt, P)
        G = gh * gw
        changed = ((nx - cv) ** 2).mean(-1).reshape(-1) > 1e-5      # same crit as exp
        tot_patch += changed.numel()
        changed_patch += int(changed.sum())
        # trivial baselines' error ON CHANGED patches (what dyn measures)
        nxf = nx.reshape(-1, nx.shape[-1])
        cvf = cv.reshape(-1, cv.shape[-1])
        pvf = pv.reshape(-1, pv.shape[-1])
        if bool(changed.any()):
            err_persist = ((nxf - cvf) ** 2).mean(-1)[changed]
            err_persist_prev = ((nxf - pvf) ** 2).mean(-1)[changed]
            err_zero = (nxf ** 2).mean(-1)[changed]
            sum_persist += float(err_persist.sum())
            sum_persist_prev += float(err_persist_prev.sum())
            sum_zero += float(err_zero.sum())
            n_changed += int(changed.sum())
        fg_frac += float((cur > 0.05).float().mean())
        nfr += 1
        prev, cur = cur, nxt
    return dict(
        env=env_name,
        changed_frac=changed_patch / max(1, tot_patch),
        fg_pixel_frac=fg_frac / nfr,
        dyn_persist=sum_persist / max(1, n_changed),
        dyn_persist_prev=sum_persist_prev / max(1, n_changed),
        dyn_zero=sum_zero / max(1, n_changed),
    )


@torch.no_grad()
def frame_similarity():
    """Mean per-pixel L2 between the time-averaged frame of each game (a crude
    'how alike do they look' measure), plus the average single-frame energy."""
    out = {}
    means = {}
    for name, A in [("pong", 3), ("breakout", 3), ("snake", 4)]:
        env = make(name)
        acc = torch.zeros(3 * IMG * IMG, device=DEVICE)
        for _ in range(300):
            step(env, A)
            acc += env.state.mean(0)
        means[name] = acc / 300
    def l2(a, b):
        return float(((means[a] - means[b]) ** 2).mean().sqrt())
    out["pong_vs_breakout"] = l2("pong", "breakout")
    out["pong_vs_snake"] = l2("pong", "snake")
    out["breakout_vs_snake"] = l2("breakout", "snake")
    return out


if __name__ == "__main__":
    print(f"device={DEVICE}  N={N} img={IMG} patch={P}\n")
    print("=== (A)(B) changed-patch fraction + TRIVIAL-baseline dyn (error on changed) ===")
    print(f"{'env':10} {'changed%':>9} {'fg_pix%':>8} {'dyn_persist':>12} "
          f"{'dyn_persistPrev':>15} {'dyn_zero':>10}")
    for name, A in [("pong", 3), ("breakout", 3), ("snake", 4)]:
        r = probe(name, A)
        print(f"{r['env']:10} {r['changed_frac']*100:8.2f}% {r['fg_pixel_frac']*100:7.2f}% "
              f"{r['dyn_persist']:12.5f} {r['dyn_persist_prev']:15.5f} {r['dyn_zero']:10.5f}")
    print("\n=== (C) time-averaged frame L2 distance (visual similarity) ===")
    for k, v in frame_similarity().items():
        print(f"  {k:22} {v:.5f}")
    print("\nNG library reported dyn (Breakout, final): ~0.0040-0.0061 (warm), ~0.0037-0.0049 (scratch)")
