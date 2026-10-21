"""v74 offline: how well do different learners turn measured branch values into a policy?

Data: scripts/collect_q_v74.py (roots, value of every slot). Cross-validation by rollout (5 folds).
For each learner, on held-out roots: gain = q[chosen] - q[base] (what the choice is worth over the
constant base, in the agent's own events per hold+horizon steps) and regret = max q - q[chosen].
The oracle (argmax of the measured values) bounds the gain; 'const' has gain 0.

Learners:
  conv_q / conv_ce : the live PolicyNet on ego views (value regression / argmax labels)
  obj_q / obj_ce   : an MLP on an egocentric OBJECT list (foreground components relative to me,
                     with size and motion), the same format in every game
  obj_knn          : k nearest neighbours in the object space, average values
Usage: C:/Python314/python.exe -m scripts.offline_q_v74 DATA.npz [DATA2.npz ...]
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from ragnarok.agent_v71.policy import PolicyNet, ego_view

K_OBJ = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def components(obs, bg, counters):
    fg = (obs != bg).any(-1)
    for y0, y1, x0, x1 in counters:
        fg[y0:y1, x0:x1] = False
    lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        m = lab[sl] == i + 1
        ys, xs = np.nonzero(m)
        out.append((xs.mean() + sl[1].start, ys.mean() + sl[0].start, float(m.sum()),
                    sl[1].stop - sl[1].start, sl[0].stop - sl[0].start))
    return out


def obj_features(obs, prev, me, bg, counters):
    H, W = obs.shape[:2]
    mx, my = (W / 2, H / 2) if me[0] < 0 else (float(me[0]), float(me[1]))
    cur, old = components(obs, bg, counters), components(prev, bg, counters)
    rows = []
    for cx, cy, s, w, h in cur:
        d = abs(cx - mx) + abs(cy - my)
        if d < 4:                                  # me
            continue
        vx = vy = 0.0
        if old:
            j = int(np.argmin([abs(cx - o[0]) + abs(cy - o[1]) + 2 * abs(np.log((s + 1) / (o[2] + 1))) for o in old]))
            if abs(cx - old[j][0]) + abs(cy - old[j][1]) < 16:
                vx, vy = cx - old[j][0], cy - old[j][1]
        rows.append((d, [(cx - mx) / W, (cy - my) / H, np.log1p(s) / 6, w / W, h / H, vx / 8, vy / 8, 1.0]))
    rows.sort(key=lambda r: r[0])
    f = np.zeros((K_OBJ, 8), np.float32)
    for i, (_, r) in enumerate(rows[:K_OBJ]):
        f[i] = r
    return np.r_[[mx / W, my / H], f.ravel()].astype(np.float32)


def conv_inputs(obs, prev, me, bg):
    m = None if me[0] < 0 else (float(me[0]), float(me[1]))
    _, _, pf = ego_view(prev, bg, m, None)
    img, pos, _ = ego_view(obs, bg, m, pf)
    return img.astype(np.uint8), pos


class MLP(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, k))

    def forward(self, x):
        return self.f(x)


def train(net, X, Q, M, base, mode, steps=2000, bs=128, lr=1e-3, extra=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Qt = torch.tensor(np.nan_to_num(Q), device=DEV)
    Mt = torch.tensor(M, device=DEV)
    lab = label_of(Q, M, base)
    Lt = torch.tensor(lab, device=DEV)
    n = len(Q)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=DEV)
        lo = net(*[x[idx] for x in X])
        Mi = Mt[idx]
        if mode == "q":
            k = Mi.sum(1, keepdim=True)
            lo = lo - (lo * Mi).sum(1, keepdim=True) / k
            qc = Qt[idx] - (Qt[idx] * Mi).sum(1, keepdim=True) / k
            loss = (((lo - qc) ** 2) * Mi).sum(1).mean()
        else:
            loss = F.cross_entropy(lo.masked_fill(~Mi, -1e9), Lt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def label_of(Q, M, base):
    Qm = np.where(M, np.nan_to_num(Q, nan=-1e9), -1e9)
    top = Qm.max(1)
    lab = Qm.argmax(1)
    keep = Qm[:, base] == top                        # ties go to the base slot
    lab[keep] = base
    return lab


def predict(net, X, M):
    with torch.no_grad():
        lo = net(*X).masked_fill(~torch.tensor(M, device=DEV), -1e9)
    return lo.argmax(1).cpu().numpy()


def predict_margin(net, X, M, base, taus):
    """Conservative improvement: deviate from the base slot only when the predicted advantage of
    the best slot over the base exceeds tau (value units for q-models, log-odds for ce-models)."""
    with torch.no_grad():
        lo = net(*X).masked_fill(~torch.tensor(M, device=DEV), -1e9).cpu().numpy()
    best = lo.argmax(1)
    adv = lo[np.arange(len(lo)), best] - lo[:, base]
    return {t: np.where(adv > t, best, base) for t in taus}


TAUS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def main(paths):
    res = {}
    for p in paths:
        Dz = np.load(p, allow_pickle=True)
        obs, prev, me, Q, rid = Dz["obs"], Dz["prev"], Dz["me"], Dz["q"], Dz["rid"]
        mask, base, bg, counters = Dz["mask"].astype(bool), int(Dz["base"]), Dz["bg"], Dz["counters"]
        n = len(Q)
        Of = np.stack([obj_features(obs[i], prev[i], me[i], bg, counters) for i in range(n)])
        CI = [conv_inputs(obs[i], prev[i], me[i], bg) for i in range(n)]
        Ci = np.stack([c[0] for c in CI]); Cp = np.stack([c[1] for c in CI])
        Qn = np.nan_to_num(Q, nan=-1e9)
        folds = np.unique(rid) % 5
        fold_of = {r: f for r, f in zip(np.unique(rid), np.random.default_rng(0).permutation(len(np.unique(rid))) % 5)}
        fid = np.array([fold_of[r] for r in rid])
        dev = {}
        out = {k: [] for k in ["oracle", "const", "conv_q", "conv_ce", "obj_q", "obj_ce", "obj_knn"]}
        for m in ("conv_q", "conv_ce", "obj_q", "obj_ce"):
            for t in TAUS:
                out[f"{m}@{t}"] = []
        frac = float(__import__("os").environ.get("FRAC", "1"))
        sub = np.random.default_rng(1).random(len(np.unique(rid))) < frac   # keep a fraction of rollouts
        keep_r = {r for r, k in zip(np.unique(rid), sub) if k}
        for f in range(5):
            tr, te = (fid != f) & np.array([r in keep_r for r in rid]), fid == f
            if te.sum() == 0 or tr.sum() < 20:
                continue
            qb = Qn[te, base]
            pick = {"oracle": label_of(Q[te], np.broadcast_to(mask, Q[te].shape), base), "const": np.full(te.sum(), base)}
            Mtr = np.broadcast_to(mask, (tr.sum(), len(mask))).copy()
            Mte = np.broadcast_to(mask, (te.sum(), len(mask))).copy()
            for mode in ("q", "ce"):
                torch.manual_seed(0)
                net = PolicyNet(len(mask)).to(DEV)
                Xtr = [torch.tensor(Ci[tr], device=DEV).float(), torch.tensor(Cp[tr], device=DEV)]
                train(net, Xtr, Q[tr], Mtr, base, mode)
                Xte = [torch.tensor(Ci[te], device=DEV).float(), torch.tensor(Cp[te], device=DEV)]
                pick[f"conv_{mode}"] = predict(net, Xte, Mte)
                for t, a in predict_margin(net, Xte, Mte, base, TAUS).items():
                    pick[f"conv_{mode}@{t}"] = a
                torch.manual_seed(0)
                mlp = MLP(Of.shape[1], len(mask)).to(DEV)
                train(mlp, [torch.tensor(Of[tr], device=DEV)], Q[tr], Mtr, base, mode)
                pick[f"obj_{mode}"] = predict(mlp, [torch.tensor(Of[te], device=DEV)], Mte)
                for t, a in predict_margin(mlp, [torch.tensor(Of[te], device=DEV)], Mte, base, TAUS).items():
                    pick[f"obj_{mode}@{t}"] = a
            # kNN in object space
            mu, sd = Of[tr].mean(0), Of[tr].std(0) + 1e-6
            A, B = (Of[tr] - mu) / sd, (Of[te] - mu) / sd
            d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            nn_idx = np.argsort(d, 1)[:, :10]
            qk = np.nanmean(np.where(mask, Q[tr], np.nan)[nn_idx], 1)
            qk = np.where(mask, np.nan_to_num(qk, nan=-1e9), -1e9)
            pick["obj_knn"] = np.where(qk[:, base] >= qk.max(1), base, qk.argmax(1))
            for k, a in pick.items():
                out[k].extend((Qn[te][np.arange(te.sum()), a] - qb).tolist())
                dev.setdefault(k, []).extend((a != base).tolist())
        top = (np.where(mask, Qn, -1e9).max(1) - Qn[:, base]).mean()
        devf = {k: round(float(np.mean(v)), 3) for k, v in dev.items()}
        summ = {k: dict(gain=round(float(np.mean(v)), 3), se=round(float(np.std(v) / np.sqrt(len(v))), 3),
                        deviate=devf.get(k),
                        frac_of_oracle=round(float(np.mean(v)) / top, 3) if top > 0 else None) for k, v in out.items() if v}
        res[str(Dz["game"])] = dict(n=n, base=int(base), summary=summ)
        print(str(Dz["game"]), n, flush=True)
        for k, v in summ.items():
            print(f"   {k:14s} gain {v['gain']:+.3f} ± {v['se']:.3f}  deviate {v['deviate']}  of_oracle {v['frac_of_oracle']}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
