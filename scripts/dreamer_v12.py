"""v12 Phase C — learn to ACT by DREAMING, from pixels.

Dreamer-style loop on the craft world FROM PIXELS:
  1. collect a real pixel rollout, tracking latent state (encode obs -> latent;
     actor acts on the latent);
  2. train the RSSM world model on it (Phase-B machinery);
  3. train ACTOR + CRITIC purely in IMAGINATION — roll the frozen world model
     H steps from real latents with the actor, compute lambda-returns on the
     model's PREDICTED reward + critic, update actor (REINFORCE) + critic (MSE).
Deploy the dreamed actor in the REAL env; compare achievements unlocked to a
random baseline.

Usage: python -m scripts.dreamer_v12 [--iters 120] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.rollout import RolloutBatch
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, N_ACH, WOOD, STONE_I, COAL_I, IRON_I)
from scripts.worldmodel_v12 import build_rssm, HID, STOCH, ACTION_DIM

FEAT = HID + STOCH
RES = [WOOD, STONE_I, COAL_I, IRON_I]      # for the dense collect reward


class MLP(nn.Module):
    def __init__(self, out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def collect_real(env, rssm, actor, horizon, explore=0.3, dense=False):
    """Real pixel rollout; track latent (posterior), act with the actor
    (epsilon-explore). Returns a RolloutBatch for WM training. If dense, the
    reward is the number of resource units gathered this step (a dense,
    learnable signal) instead of the env's sparse achievement reward."""
    N = env.num_envs
    O, A, R, D = [], [], [], []
    obs = env.state
    h, z = rssm.initial_state(N, DEVICE)
    prev_a = torch.zeros(N, ACTION_DIM, device=DEVICE)
    for _ in range(horizon):
        h, z = rssm.encode_observation(obs, h, z, prev_a, deterministic=True)
        logits = actor(torch.cat([h, z], -1))
        a = torch.distributions.Categorical(logits=logits).sample()
        rand = torch.rand(N, device=DEVICE) < explore
        a = torch.where(rand, torch.randint(0, ACTION_DIM, (N,), device=DEVICE), a)
        a_oh = torch.nn.functional.one_hot(a, ACTION_DIM).float()
        res_before = env.inv[:, RES].sum(-1).clone() if dense else None
        nobs, r, _t, _tr, done = env.step(a)
        if dense:
            r = (env.inv[:, RES].sum(-1) - res_before).clamp(min=0).float()
        O.append(obs); A.append(a_oh); R.append(r); D.append(done.float())
        prev_a = a_oh
        if bool(done.any()):
            m = (~done).float().unsqueeze(-1)
            h = h * m; z = z * m; prev_a = prev_a * m
        obs = nobs
    zc = torch.zeros(N, horizon, device=DEVICE)
    return RolloutBatch(obs=torch.stack(O, 1), raw_obs=torch.stack(O, 1),
                        actions=torch.stack(A, 1), rewards=torch.stack(R, 1),
                        dones=torch.stack(D, 1), logp=zc, values=zc,
                        last_obs=obs, last_value=torch.zeros(N, device=DEVICE))


def train_actor_critic(rssm, actor, critic, opt, starts, H=15, gamma=0.99,
                       lam=0.95, ent_coef=0.01):
    """One actor-critic update on imagined rollouts from `starts` latents."""
    h, z = starts[:, :HID].contiguous(), starts[:, HID:].contiguous()
    feats, acts, rews = [], [], []
    with torch.no_grad():                       # roll the FROZEN world model
        for _ in range(H):
            feat = torch.cat([h, z], -1)
            a = torch.distributions.Categorical(logits=actor(feat)).sample()
            feats.append(feat); acts.append(a)
            a_oh = torch.nn.functional.one_hot(a, ACTION_DIM).float()
            h = rssm.core.step(h, z, a_oh)
            pm, pls = rssm.core.forward_prior(h)
            z = rssm.core.sample(pm, pls)
            rews.append(rssm.reward_predictor(h, z))
        last_feat = torch.cat([h, z], -1)
        feats_t = torch.stack(feats, 1)          # (B,H,FEAT)
        acts_t = torch.stack(acts, 1)            # (B,H)
        rews_t = torch.stack(rews, 1)            # (B,H)
        B = feats_t.shape[0]
        v_all = critic(feats_t.reshape(B * H, FEAT)).reshape(B, H)
        v_last = critic(last_feat).squeeze(-1)
        R = torch.zeros(B, H, device=DEVICE)
        nextR = v_last
        for t in reversed(range(H)):
            nextv = v_all[:, t + 1] if t < H - 1 else v_last
            R[:, t] = rews_t[:, t] + gamma * ((1 - lam) * nextv + lam * nextR)
            nextR = R[:, t]
    # grad update on detached latents
    ff = feats_t.reshape(B * H, FEAT)
    dist = torch.distributions.Categorical(logits=actor(ff))
    logp = dist.log_prob(acts_t.reshape(B * H))
    ent = dist.entropy().mean()
    v = critic(ff).squeeze(-1)
    Rf = R.reshape(B * H)
    adv = (Rf - v).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    actor_loss = -(logp * adv).mean() - ent_coef * ent
    critic_loss = (v - Rf.detach()).pow(2).mean()
    opt.zero_grad()
    (actor_loss + 0.5 * critic_loss).backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
    opt.step()
    return float(rews_t.mean().item()), float(ent.item())


@torch.no_grad()
def deploy(env, rssm, actor, steps, greedy=True):
    """Run the actor in the real env from pixels; return mean achievements."""
    N = env.num_envs
    env.reset()
    unlocked = torch.zeros(N, N_ACH, dtype=torch.bool, device=DEVICE)
    obs = env.state
    h, z = rssm.initial_state(N, DEVICE)
    prev_a = torch.zeros(N, ACTION_DIM, device=DEVICE)
    for _ in range(steps):
        h, z = rssm.encode_observation(obs, h, z, prev_a, deterministic=True)
        logits = actor(torch.cat([h, z], -1))
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
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
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--wm-epochs", type=int, default=3)
    p.add_argument("--ac-updates", type=int, default=8)
    p.add_argument("--imag-h", type=int, default=15)
    p.add_argument("--imag-batch", type=int, default=512)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--deploy-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--dense", action="store_true",
                   help="dense collect reward (fair test: isolates the actor "
                        "from the sparse-reward confound)")
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.horizon, args.eval_every = 6, 32, 24, 3
        args.ac_updates, args.imag_batch = 3, 128

    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecCraftWorld(args.num_envs, grid=args.grid, view=args.view,
                              max_steps=10 ** 9, pixel=True, tile=args.tile)
    rssm = build_rssm(env.img_hw)
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=3e-4)
    actor, critic = MLP(ACTION_DIM).to(DEVICE), MLP(1).to(DEVICE)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    rand_prof = random_profile(env, args.deploy_steps)
    print(f"[v12-C] device={DEVICE} | Dreamer-from-pixels | random baseline "
          f"achievements: wood {rand_prof[0]:.2f} table {rand_prof[1]:.2f} "
          f"wpick {rand_prof[2]:.2f} | total {rand_prof.sum():.2f}", flush=True)

    t0 = time.perf_counter()
    curve = []
    for it in range(1, args.iters + 1):
        batch = collect_real(env, rssm, actor, args.horizon,
                             explore=max(0.1, 0.5 - it / args.iters * 0.4),
                             dense=args.dense)
        wm.train_world_model_on_rollout(batch, epochs=args.wm_epochs)
        with torch.no_grad():
            out = rssm.observe(batch.obs, batch.actions)
            feats = torch.cat([out["h"], out["z"]], -1).reshape(-1, FEAT)
        for _ in range(args.ac_updates):
            idx = torch.randint(0, feats.shape[0], (args.imag_batch,), device=DEVICE)
            imag_r, ent = train_actor_critic(rssm, actor, critic, opt, feats[idx],
                                             H=args.imag_h)
        if it % args.eval_every == 0:
            prof = deploy(env, rssm, actor, args.deploy_steps)
            depth = max((i for i in range(N_ACH) if prof[i] >= 0.5), default=-1)
            curve.append(dict(it=it, profile=prof.tolist(), imag_r=imag_r))
            print(f"  it {it:>3} | dreamed-actor: wood {prof[0]:.2f} table "
                  f"{prof[1]:.2f} wpick {prof[2]:.2f} stone {prof[3]:.2f} | "
                  f"total {prof.sum():.2f} | deepest>=.5 {depth} | "
                  f"imagR {imag_r:.3f} ent {ent:.2f} | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    prof = deploy(env, rssm, actor, args.deploy_steps)
    beat = sum(1 for i in range(N_ACH) if prof[i] > rand_prof[i] + 0.05)
    ok = prof.sum() > rand_prof.sum() + 0.5 and prof[0] >= 0.5
    verdict = ("DREAMING-TO-ACT WORKS — an actor trained ONLY in imagination "
               "(in the learned pixel world model) unlocks more/deeper "
               "achievements than random, acting from pixels."
               if ok else
               f"CHECK/NEGATIVE — dreamed actor total {prof.sum():.2f} vs random "
               f"{rand_prof.sum():.2f} (beat on {beat}/{N_ACH} achievements). "
               "Dreamer is finicky; CEM-in-latent is the fallback.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v12c.json"), "w") as f:
        json.dump(dict(random=rand_prof.tolist(), final=prof.tolist(),
                       curve=curve, verdict=verdict, ach_names=ACH_NAMES), f, indent=2)


if __name__ == "__main__":
    main()
