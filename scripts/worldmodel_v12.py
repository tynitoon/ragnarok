"""v12 Phase B — RSSM WORLD MODEL from PIXELS.

Plug a CNN encoder + transposed-conv decoder into the existing pluggable RSSM
(reuse the GRU core, prior/posterior, reward/continue predictors, and
WorldModelTrainer). Train on pixel rollouts of the craft world to PREDICT the
world (reconstruction + KL + reward). Then evaluate:
  - one-step reconstruction error (the model encodes/decodes what it sees);
  - OPEN-LOOP imagination: from a real latent, roll the prior k steps with the
    TRUE actions, decode each, compare to the actual frames -> multi-step
    prediction of the scrolling egocentric view (beats a persistence baseline
    if it learned dynamics, not just autoencoding).

Usage: python -m scripts.worldmodel_v12 [--rollouts 120] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.rollout import RolloutBatch
from ragnarok.environments.craft_world import DeviceVecCraftWorld

ENC_DIM, HID, STOCH = 128, 128, 32
ACTION_DIM = 10


class PixelEncoder(nn.Module):
    def __init__(self, img_hw, out_dim=ENC_DIM):
        super().__init__()
        self.img_hw = img_hw
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img_hw, img_hw)).reshape(1, -1).shape[1]
        self.fc = nn.Linear(d, out_dim)

    def forward(self, obs_flat):
        B = obs_flat.shape[0]
        x = obs_flat.view(B, 3, self.img_hw, self.img_hw)
        return self.fc(self.conv(x).reshape(B, -1))


class PixelDecoder(nn.Module):
    def __init__(self, in_dim, img_hw):
        super().__init__()
        self.img_hw = img_hw
        self.fc = nn.Linear(in_dim, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),   # 4 -> 6
            nn.ConvTranspose2d(32, 16, 3, stride=2), nn.ReLU(),   # 6 -> 13
            nn.ConvTranspose2d(16, 3, 4, stride=2), nn.Sigmoid())  # 13 -> 28

    def forward(self, hz):
        B = hz.shape[0]
        x = self.fc(hz).view(B, 32, 4, 4)
        return self.deconv(x).reshape(B, -1)


def build_rssm(img_hw):
    return RSSM(obs_dim=3 * img_hw * img_hw, action_dim=ACTION_DIM,
                stoch_dim=STOCH, hidden_dim=HID, encoder_hidden=ENC_DIM,
                encoder=PixelEncoder(img_hw), decoder=PixelDecoder(HID + STOCH, img_hw)
                ).to(DEVICE)


@torch.no_grad()
def collect_wm(env, horizon, ppo=None):
    """Random (or skill-driven) pixel rollout -> RolloutBatch with one-hot acts."""
    N = env.num_envs
    O, A, R, D = [], [], [], []
    obs = env.state
    for _ in range(horizon):
        a = (ppo.act(obs, deterministic=False) if ppo is not None
             else torch.randint(0, ACTION_DIM, (N,), device=DEVICE))
        a_oh = torch.nn.functional.one_hot(a, ACTION_DIM).float()
        obs2, r, _t, _tr, done = env.step(a)
        O.append(obs); A.append(a_oh); R.append(r); D.append(done.float())
        obs = obs2
    z = torch.zeros(N, horizon, device=DEVICE)
    return RolloutBatch(obs=torch.stack(O, 1), raw_obs=torch.stack(O, 1),
                        actions=torch.stack(A, 1), rewards=torch.stack(R, 1),
                        dones=torch.stack(D, 1), logp=z, values=z,
                        last_obs=obs, last_value=torch.zeros(N, device=DEVICE))


@torch.no_grad()
def evaluate(rssm, env, horizon, k=10, t0=6):
    """One-step recon MSE + open-loop k-step recon MSE vs a persistence baseline."""
    batch = collect_wm(env, horizon)
    out = rssm.observe(batch.obs, batch.actions)              # posterior pass
    recon = rssm.decoder(torch.cat([out["h"], out["z"]], -1).reshape(-1, HID + STOCH))
    recon = recon.reshape(batch.obs.shape)
    one_step_mse = float(((recon - batch.obs) ** 2).mean().item())

    # open-loop: from posterior latent at t0, roll prior k steps with true acts
    h = out["h"][:, t0]; z = out["z"][:, t0]
    ol_mse, pers_mse = [], []
    persist = batch.obs[:, t0]                                # naive: frame stays
    for j in range(k):
        a = batch.actions[:, t0 + j]
        h = rssm.core.step(h, z, a)
        pm, pls = rssm.core.forward_prior(h)
        z = rssm.core.sample(pm, pls)
        pred = rssm.decoder(torch.cat([h, z], -1))
        actual = batch.obs[:, t0 + 1 + j]
        ol_mse.append(float(((pred - actual) ** 2).mean().item()))
        pers_mse.append(float(((persist - actual) ** 2).mean().item()))
    return one_step_mse, ol_mse, pers_mse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=int, default=120)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.rollouts, args.num_envs, args.horizon, args.eval_every = 6, 32, 24, 3

    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecCraftWorld(args.num_envs, grid=args.grid, view=args.view,
                              max_steps=10 ** 9, pixel=True, tile=args.tile)
    img_hw = env.img_hw
    print(f"[v12-B] device={DEVICE} | pixel RSSM | obs 3x{img_hw}x{img_hw} "
          f"(dim {env.obs_dim})", flush=True)
    rssm = build_rssm(img_hw)
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=args.lr)

    t0 = time.perf_counter()
    curve = []
    for it in range(1, args.rollouts + 1):
        batch = collect_wm(env, args.horizon)
        wm.train_world_model_on_rollout(batch, epochs=args.epochs)
        if it % args.eval_every == 0:
            one_step, ol, pers = evaluate(rssm, env, args.horizon)
            curve.append(dict(it=it, one_step_mse=one_step,
                              open_loop_mse=ol, persistence_mse=pers))
            print(f"  it {it:>4} | 1-step recon MSE {one_step:.4f} | "
                  f"open-loop[1,{len(ol)//2},{len(ol)}] "
                  f"{ol[0]:.4f}/{ol[len(ol)//2]:.4f}/{ol[-1]:.4f} | "
                  f"persistence {pers[-1]:.4f} | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    one_step, ol, pers = evaluate(rssm, env, args.horizon)
    beats_persist = sum(1 for a, b in zip(ol, pers) if a < b)
    ok = one_step < 0.01 and beats_persist >= len(ol) * 0.7
    verdict = ("WORLD MODEL WORKS — low pixel reconstruction and open-loop "
               "rollouts that beat the persistence baseline -> the model learned "
               "to SEE and PREDICT the world (dynamics, not just autoencoding)."
               if ok else
               f"CHECK — 1-step MSE {one_step:.4f}, beats-persistence "
               f"{beats_persist}/{len(ol)} steps.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v12b.json"), "w") as f:
        json.dump(dict(curve=curve, final_one_step=one_step, final_open_loop=ol,
                       final_persistence=pers, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
