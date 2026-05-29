"""Device-batched discrete-action PPO — the on-policy learner for
DeviceVecCraftWorld (v6.0). MLP actor-critic over the flat observation,
categorical policy, batched rollout across N parallel envs, batched GAE.

Mirrors the device-resident paradigm of the rest of the codebase: the env
exposes `.state` (N, obs_dim) and `.step(action:(N,))`, everything stays on
the accelerator, no host sync in the hot loop.
"""

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.advantages import compute_gae_batched


class PPONet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)   # near-uniform start

    def forward(self, obs):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


class ConvPPONet(nn.Module):
    """CNN actor-critic for flattened RGB image obs (B, 3*H*W). The agent
    learns features from pixels (no hand-given symbols)."""
    def __init__(self, img_hw, action_dim, hidden=256):
        super().__init__()
        self.img_hw = img_hw
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img_hw, img_hw)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.view(B, 3, self.img_hw, self.img_hw)
        h = self.fc(self.conv(x).reshape(B, -1))
        return self.actor(h), self.critic(h).squeeze(-1)


class DiscretePPO:
    def __init__(self, obs_dim, action_dim, hidden=256, lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, entropy=0.01, value_coeff=0.5,
                 epochs=4, minibatches=4, grad_clip=0.5, net=None):
        self.net = (net if net is not None
                    else PPONet(obs_dim, action_dim, hidden)).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.entropy, self.value_coeff = entropy, value_coeff
        self.epochs, self.minibatches, self.grad_clip = epochs, minibatches, grad_clip
        self.action_dim = action_dim
        self.total_steps = 0

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, _ = self.net(obs)
        if deterministic:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()

    def collect(self, env, n_steps):
        O, A, LP, V, R, D = [], [], [], [], [], []
        obs = env.state
        for _ in range(n_steps):
            with torch.no_grad():
                logits, val = self.net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            O.append(obs); A.append(a); LP.append(lp); V.append(val)
            obs, r, term, trunc, done = env.step(a)
            R.append(r); D.append(done.float())
        with torch.no_grad():
            last_val = self.net(obs)[1]
        self.total_steps += env.num_envs * n_steps
        return dict(obs=torch.stack(O, 1), act=torch.stack(A, 1),
                    logp=torch.stack(LP, 1), val=torch.stack(V, 1),
                    rew=torch.stack(R, 1), done=torch.stack(D, 1),
                    last_val=last_val)

    def update(self, roll):
        adv, ret = compute_gae_batched(roll["rew"], roll["val"], roll["done"],
                                       roll["last_val"], self.gamma, self.lam)
        N, T = roll["rew"].shape
        B = N * T
        obs = roll["obs"].reshape(B, -1)
        act = roll["act"].reshape(B)
        old_lp = roll["logp"].reshape(B)
        adv = adv.reshape(B); ret = ret.reshape(B)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = max(1, B // self.minibatches)
        last = {}
        for _ in range(self.epochs):
            idx = torch.randperm(B, device=DEVICE)
            for i in range(self.minibatches):
                j = idx[i * mb:(i + 1) * mb]
                logits, val = self.net(obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act[j])
                ent = dist.entropy().mean()
                ratio = (lp - old_lp[j]).exp()
                aj = adv[j]
                pg = -torch.min(ratio * aj,
                                ratio.clamp(1 - self.clip, 1 + self.clip) * aj).mean()
                vl = (val - ret[j]).pow(2).mean()
                loss = pg + self.value_coeff * vl - self.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()
                last = dict(pg=float(pg.item()), v=float(vl.item()),
                            ent=float(ent.item()))
        return last

    def train_iter(self, env, n_steps):
        roll = self.collect(env, n_steps)
        m = self.update(roll)
        m["mean_rew"] = float(roll["rew"].sum(1).mean().item())
        return m
