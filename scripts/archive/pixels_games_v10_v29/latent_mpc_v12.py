"""v12 Phase C fallback — PLANNING in the learned pixel world model.

Robust alternative to dreaming-a-policy: train the RSSM world model (from
pixels) on dense-reward rollouts, then ACT by random-shooting MPC — at each
step encode the image to a latent, sample K action sequences, roll each
through the world model, and execute the first action of the highest
predicted-reward sequence (receding horizon). No actor training.

Decisive: the MPC agent collects wood / unlocks achievements far above random
-> control via a world model learned from pixels works (model-based control,
v4-Phase-1 idea, now perceptual). Else: honest negative for acting-via-
learned-pixel-model in this budget.

Usage: python -m scripts.latent_mpc_v12 [--wm-rollouts 150] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.rollout import RolloutBatch
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, N_ACH, WOOD, STONE_I, COAL_I, IRON_I)
from scripts.worldmodel_v12 import build_rssm, HID, STOCH, ACTION_DIM

RES = [WOOD, STONE_I, COAL_I, IRON_I]


@torch.no_grad()
def collect_dense(env, horizon):
    """Random rollout with a DENSE collect reward (resources gathered/step)."""
    N = env.num_envs
    O, A, R, D = [], [], [], []
    obs = env.state
    for _ in range(horizon):
        a = torch.randint(0, ACTION_DIM, (N,), device=DEVICE)
        before = env.inv[:, RES].sum(-1).clone()
        nobs, _, _t, _tr, done = env.step(a)
        r = (env.inv[:, RES].sum(-1) - before).clamp(min=0).float()
        O.append(obs); A.append(torch.nn.functional.one_hot(a, ACTION_DIM).float())
        R.append(r); D.append(done.float())
        obs = nobs
    zc = torch.zeros(N, horizon, device=DEVICE)
    return RolloutBatch(obs=torch.stack(O, 1), raw_obs=torch.stack(O, 1),
                        actions=torch.stack(A, 1), rewards=torch.stack(R, 1),
                        dones=torch.stack(D, 1), logp=zc, values=zc,
                        last_obs=obs, last_value=torch.zeros(N, device=DEVICE))


@torch.no_grad()
def plan_action(rssm, h, z, K, H):
    """Random-shooting MPC: K action seqs of length H per env, rolled through
    the world model; return the first action of the best-predicted sequence."""
    N = h.shape[0]
    he = h.repeat_interleave(K, 0)
    ze = z.repeat_interleave(K, 0)
    seqs = torch.randint(0, ACTION_DIM, (N * K, H), device=DEVICE)
    total = torch.zeros(N * K, device=DEVICE)
    for t in range(H):
        a_oh = torch.nn.functional.one_hot(seqs[:, t], ACTION_DIM).float()
        he = rssm.core.step(he, ze, a_oh)
        pm, pls = rssm.core.forward_prior(he)
        ze = rssm.core.sample(pm, pls)
        total += rssm.reward_predictor(he, ze)
    best = total.view(N, K).argmax(1)
    return seqs.view(N, K, H)[torch.arange(N, device=DEVICE), best, 0]


@torch.no_grad()
def deploy_mpc(env, rssm, steps, K, H):
    N = env.num_envs
    env.reset()
    unlocked = torch.zeros(N, N_ACH, dtype=torch.bool, device=DEVICE)
    obs = env.state
    h, z = rssm.initial_state(N, DEVICE)
    prev_a = torch.zeros(N, ACTION_DIM, device=DEVICE)
    for _ in range(steps):
        h, z = rssm.encode_observation(obs, h, z, prev_a, deterministic=True)
        a = plan_action(rssm, h, z, K, H)
        prev_a = torch.nn.functional.one_hot(a, ACTION_DIM).float()
        obs, _, _, _, _ = env.step(a)
        unlocked |= env.unlocked
    return unlocked.float().mean(0).cpu()


@torch.no_grad()
def random_profile(env, steps):
    N = env.num_envs
    env.reset()
    unlocked = torch.zeros(N, N_ACH, dtype=torch.bool, device=DEVICE)
    for _ in range(steps):
        env.step(torch.randint(0, ACTION_DIM, (N,), device=DEVICE))
        unlocked |= env.unlocked
    return unlocked.float().mean(0).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm-rollouts", type=int, default=150)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--deploy-envs", type=int, default=64)
    p.add_argument("--deploy-steps", type=int, default=60)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.wm_rollouts, args.num_envs, args.horizon = 6, 32, 24
        args.K, args.deploy_envs, args.deploy_steps = 64, 16, 30

    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecCraftWorld(args.num_envs, grid=args.grid, view=args.view,
                              max_steps=10 ** 9, pixel=True, tile=args.tile)
    rssm = build_rssm(env.img_hw)
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=3e-4)
    print(f"[v12-C/MPC] device={DEVICE} | training dense-reward pixel WM "
          f"({args.wm_rollouts} rollouts)...", flush=True)
    t0 = time.perf_counter()
    for it in range(1, args.wm_rollouts + 1):
        wm.train_world_model_on_rollout(collect_dense(env, args.horizon),
                                        epochs=args.epochs)
        if it % 30 == 0:
            print(f"  [wm] rollout {it}/{args.wm_rollouts} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    deval = DeviceVecCraftWorld(args.deploy_envs, grid=args.grid, view=args.view,
                                max_steps=10 ** 9, pixel=True, tile=args.tile)
    rand = random_profile(deval, args.deploy_steps)
    mpc = deploy_mpc(deval, rssm, args.deploy_steps, args.K, args.H)
    print(f"\n  {'achievement':20s} {'MPC':>8} {'random':>8}")
    for i, nm in enumerate(ACH_NAMES):
        print(f"  {nm:20s} {mpc[i]:>8.2f} {rand[i]:>8.2f}", flush=True)
    ok = mpc[0] > rand[0] + 0.1 and mpc.sum() > rand.sum() + 0.3
    verdict = ("MODEL-BASED CONTROL FROM PIXELS WORKS — random-shooting MPC in "
               "the learned pixel world model collects/achieves above random "
               "-> control via a world model learned from pixels (planning, not "
               "dreaming). Phase C salvaged."
               if ok else
               f"NEGATIVE — MPC total {mpc.sum():.2f} vs random {rand.sum():.2f}; "
               "acting-via-learned-pixel-model not cracked in this budget. A "
               "(perception) + B (world model predicts) stand; control is future work.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v12c_mpc.json"), "w") as f:
        json.dump(dict(mpc=mpc.tolist(), random=rand.tolist(), verdict=verdict,
                       ach_names=ACH_NAMES, K=args.K, H=args.H), f, indent=2)


if __name__ == "__main__":
    main()
