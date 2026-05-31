"""v42 — does a notion that AMORTISES the search make a search-heavy task cheaper?
(FAIR: both model-free; the v17b mechanism, generalised; frozen design.)

Diagnosis from 4 nulls: reuse pays iff the notion short-circuits an expensive
SEARCH the task requires. Tetris IS search-heavy: choosing a placement needs
simulating the drop (gravity+collision+line-clears). The NOTION = that drop
outcome (lines/holes/height/dead per placement) — the env computes it analytically
(evaluate_placements), and v17b showed it is LEARNABLE from the board. Here we
ISOLATE the reuse effect with the oracle notion (learned-notion is the follow-up):

- WARM: model-free PPO whose obs = the per-placement drop-outcome metrics (the
  amortised search). It must still learn WHICH outcome is good (clear lines, stay
  low) and pick that placement.
- SCRATCH: model-free PPO from the raw BOARD IMAGE (must learn the drop-outcome
  implicitly to choose well).
Both model-free PPO -> FAIR (addresses the v38 strawman critique). Metric: lines
cleared per eval vs iters, >=3 seeds. If WARM>>SCRATCH -> reuse pays when it
amortises a real search (the positive); if null even here, the theory is wrong.

Usage: python -m scripts.notion_amortizes_search_v42 [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet, PPONet
from ragnarok.environments.tetris import DeviceVecTetris

SCALE = torch.tensor([0.25, 1 / 30.0, 1 / 60.0, 1.0], device=DEVICE)   # normalise metrics


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class WarmTetris:
    """obs = per-placement drop-outcome metrics (the amortised-search notion)."""
    def __init__(self, n, seed=0):
        self.env = DeviceVecTetris(n, seed=seed)
        self.num_envs = n
        self.action_dim = self.env.action_dim
        self.obs_dim = self.action_dim * 4

    @property
    def state(self):
        m = self.env.evaluate_placements() * SCALE          # (N, A, 4)
        return m.reshape(self.num_envs, -1)

    def step(self, a):
        _, r, term, trunc, done = self.env.step(a)
        return self.state, r, term, trunc, done

    @property
    def cum_lines(self):
        return self.env.cum_lines


@torch.no_grad()
def eval_lines(ppo, make_env, n=128, steps=200, seed=9):
    env = make_env(n, seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return float(env.cum_lines.mean())


def run_arm(mode, iters, eval_every, num_envs, seed):
    seed_all(seed + (1 if mode == "warm" else 2))
    if mode == "warm":
        make = lambda n, s: WarmTetris(n, seed=s)
        probe = make(num_envs, seed)
        net = PPONet(probe.obs_dim, probe.action_dim, hidden=256)
    else:
        make = lambda n, s: DeviceVecTetris(n, seed=s)
        probe = make(num_envs, seed)
        net = ConvPPONet(probe.img_hw, probe.action_dim, hidden=256)
    ppo = DiscretePPO(probe.obs_dim, probe.action_dim, entropy=0.01, net=net)
    env = make(num_envs, seed)
    curve = [(0, round(eval_lines(ppo, make), 2))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 16)
        if it % eval_every == 0:
            curve.append((it, round(eval_lines(ppo, make), 2)))
    return curve


def iters_to(curve, thr):
    for it, v in curve:
        if v >= thr:
            return it
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.iters, args.num_envs = [0], 30, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v42] device={DEVICE} | notion AMORTISES SEARCH (Tetris) | WARM(drop-metrics) "
          f"vs SCRATCH(board), both model-free | lines->iters>={args.threshold}, "
          f"seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        warm = run_arm("warm", args.iters, args.eval_every, args.num_envs, s)
        scratch = run_arm("scratch", args.iters, args.eval_every, args.num_envs, s)
        wi, si = iters_to(warm, args.threshold), iters_to(scratch, args.threshold)
        rows.append(dict(seed=s, warm_iters=wi, warm_final=warm[-1][1],
                         scratch_iters=si, scratch_final=scratch[-1][1]))
        print(f"  seed {s}: WARM ->{args.threshold} lines in {wi} (final {warm[-1][1]}) | "
              f"SCRATCH in {si} (final {scratch[-1][1]}) | {time.perf_counter()-t0:.0f}s",
              flush=True)

    warm_ok = all(r["warm_iters"] is not None for r in rows)
    faster = all(r["warm_iters"] is not None and
                 (r["scratch_iters"] is None or r["warm_iters"] * 2 <= r["scratch_iters"])
                 for r in rows)
    better_final = all(r["warm_final"] >= r["scratch_final"] + 2 for r in rows)
    ok = warm_ok and (faster or better_final)
    verdict = (
        f"POSITIVE — a notion that AMORTISES the placement-search makes Tetris reliably "
        f"cheaper for a model-free agent: every seed WARM (drop-outcome feature) reaches "
        f"the line-threshold in <= half the iters of (or clears far more than) SCRATCH "
        f"(board) (per seed warm,scratch iters {[(r['warm_iters'], r['scratch_iters']) for r in rows]}; "
        f"finals {[(r['warm_final'], r['scratch_final']) for r in rows]}). FAIR (both "
        f"model-free). This confirms the diagnosis: reuse pays when the notion short-circuits "
        f"a real search. (Oracle notion here; v17b showed it is learnable -> next: learned "
        f"notion to close the loop.) REVIEW before reporting."
        if ok else
        f"PARTIAL/NEG — warm_ok={warm_ok}, faster={faster}, better_final={better_final}; "
        f"rows={rows}. If null, the amortise-search theory is wrong -> rethink.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v42_amortize_search.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, rows=rows, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
