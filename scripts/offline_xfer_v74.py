"""v74 offline transfer test: does an encoder learnt on OTHER games make a new game's policy
improvement learnable from fewer measured roots (fewer emulator frames)?

Data: Q datasets from scripts/collect_q_v74.py (roots + value of every slot, agent-side values).
Target game T, sources S (never T). For a grid of training sizes N (start-state groups of T),
folds by start state, several seeds:
  fresh   : PolicyNet from scratch on T's N roots
  carried : conv trunk pretrained on the sources (labels per source game, one head per game),
            then the whole net fine-tuned on T's N roots with a fresh head
  placebo : same pretraining with each source's labels permuted across roots (same pixels,
            no action knowledge), then fine-tuned identically
Metric: held-out gain over the base slot (agent's own events per hold+horizon steps).
'Learns faster' reads as: carried reaches a gain level with fewer roots than fresh AND placebo.
Usage: C:/Python314/python.exe -m scripts.offline_xfer_v74 TARGET.npz SOURCE1.npz [SOURCE2.npz ...]
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.agent_v71.policy import PolicyNet
from scripts.offline_q_v74 import conv_inputs, label_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = (60, 120, 240, 480, 960)
SEEDS = (0, 1, 2)


def load(p):
    Dz = np.load(p, allow_pickle=True)
    obs, prev, me, Q = Dz["obs"], Dz["prev"], Dz["me"], Dz["q"]
    mask, base, bg = Dz["mask"].astype(bool), int(Dz["base"]), Dz["bg"]
    pme = Dz["pme"] if "pme" in Dz.files else me
    grp = Dz["sid"] if "sid" in Dz.files else Dz["rid"]
    CI = []
    for i in range(len(Q)):
        m = None if me[i][0] < 0 else (float(me[i][0]), float(me[i][1]))
        pm = None if pme[i][0] < 0 else (float(pme[i][0]), float(pme[i][1]))
        from ragnarok.agent_v71.policy import ego_view
        _, _, pf = ego_view(prev[i], bg, pm, None)
        img, pos, _ = ego_view(obs[i], bg, m, pf)
        CI.append((img.astype(np.uint8), pos))
    return dict(game=str(Dz["game"]), I=np.stack([c[0] for c in CI]), P=np.stack([c[1] for c in CI]),
                Q=Q, M=mask, base=base, grp=np.asarray(grp), lab=label_of(Q, np.broadcast_to(mask, Q.shape), base))


class Multi(nn.Module):
    """The PolicyNet trunk shared across games, one action head per source game."""

    def __init__(self, heads):
        super().__init__()
        self.body = PolicyNet(2)
        self.body.head = nn.Sequential(nn.Linear(32 * 4 * 4 + 2, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, k) for k in heads])

    def forward(self, img, pos, h):
        z = self.body.conv(img).flatten(1)
        return self.heads[h](self.body.head(torch.cat([z, pos], 1)))


def pretrain(sources, permute, seed, steps=4000, bs=128):
    torch.manual_seed(seed)
    net = Multi([len(s["M"]) for s in sources]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    T = []
    for s in sources:
        lab = s["lab"].copy()
        if permute:
            lab = lab[rng.permutation(len(lab))]
        T.append((torch.tensor(s["I"], device=DEV).float(), torch.tensor(s["P"], device=DEV),
                  torch.tensor(lab, device=DEV), torch.tensor(s["M"], device=DEV)))
    for t in range(steps):
        h = t % len(T)
        I, P, L, M = T[h]
        idx = torch.randint(0, len(L), (min(bs, len(L)),), device=DEV)
        lo = net(I[idx], P[idx], h).masked_fill(~M, -1e9)
        loss = F.cross_entropy(lo, L[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def finetune(target, idx_tr, init, seed, steps=2000, bs=128):
    torch.manual_seed(seed)
    net = PolicyNet(len(target["M"])).to(DEV)
    if init is not None:                      # carry the trunk: conv + first head layer
        net.conv.load_state_dict(init.body.conv.state_dict())
        net.head[0].load_state_dict(init.body.head[0].state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    I = torch.tensor(target["I"][idx_tr], device=DEV).float()
    P = torch.tensor(target["P"][idx_tr], device=DEV)
    L = torch.tensor(target["lab"][idx_tr], device=DEV)
    M = torch.tensor(target["M"], device=DEV)
    for _ in range(steps):
        i = torch.randint(0, len(L), (min(bs, len(L)),), device=DEV)
        loss = F.cross_entropy(net(I[i], P[i]).masked_fill(~M, -1e9), L[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def gain(net, target, idx_te):
    with torch.no_grad():
        lo = net(torch.tensor(target["I"][idx_te], device=DEV).float(), torch.tensor(target["P"][idx_te], device=DEV))
        lo = lo.masked_fill(~torch.tensor(target["M"], device=DEV), -1e9)
        a = lo.argmax(1).cpu().numpy()
    Qn = np.nan_to_num(target["Q"][idx_te], nan=-1e9)
    return float(np.mean(Qn[np.arange(len(a)), a] - Qn[:, target["base"]]))


def main(tp, sps):
    T = load(tp)
    S = [load(p) for p in sps]
    assert all(s["game"] != T["game"] for s in S), "the target must never be a source"
    ug = np.unique(T["grp"])
    fold_of = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 5))
    fid = np.array([fold_of[g] for g in T["grp"]])
    Qn = np.nan_to_num(T["Q"], nan=-1e9)
    oracle = float(np.mean(np.where(T["M"], Qn, -1e9).max(1) - Qn[:, T["base"]]))
    res = {a: {n: [] for n in NS} for a in ("fresh", "carried", "placebo")}
    for seed in SEEDS:
        pre = {"carried": pretrain(S, False, seed), "placebo": pretrain(S, True, seed)}
        for f in range(5):
            te = np.nonzero(fid == f)[0]
            tr_all = np.nonzero(fid != f)[0]
            rng = np.random.default_rng(100 * seed + f)
            order = tr_all[rng.permutation(len(tr_all))]
            for n in NS:
                if n > len(order):
                    continue
                idx = order[:n]
                for arm, init in (("fresh", None), ("carried", pre["carried"]), ("placebo", pre["placebo"])):
                    res[arm][n].append(gain(finetune(T, idx, init, seed), T, te))
        print("seed", seed, "done", flush=True)
    out = dict(target=T["game"], sources=[s["game"] for s in S], oracle=round(oracle, 3), curves={})
    for arm, d in res.items():
        out["curves"][arm] = {n: (round(float(np.mean(v)), 3), round(float(np.std(v) / np.sqrt(len(v))), 3)) for n, v in d.items() if v}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
