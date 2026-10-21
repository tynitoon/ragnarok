"""v75 offline benchmark, honest version (tests T1-T3 of the distillation panel, 21 Oct 2026).

Differences from scripts/offline_q_v74.py (review of the v74 offline numbers):
  * folds group rollouts that share an identical root screen ('link'), or use the start-state id
    ('sid') when the data has it: identical roots no longer leak across folds;
  * the yardstick is the gain over the BEST CONSTANT DEVIATION ('const_dev', chosen on the training
    folds), i.e. only what depends on the screen; Assault's 'pause once' bias no longer counts;
  * 95% intervals by a cluster bootstrap over groups; paired differences against conv_ce;
  * several torch seeds; per-row choices averaged over seeds before the bootstrap;
  * input switches (off = v74 input): FULL_H pads the whole-screen view to the full height (v74's
    pooling dropped rows 192-209, where Assault's heat gauge is); CLEAN removes from the foreground
    the pixels equal to their background row's dominant colour; previous view centred on the
    previous position when the data has it ('pme');
  * losses: ce (v74 labels, ties to the base), q (v74 regression), set (partial-label likelihood:
    any best slot is right, the base when it is among them), regret (expected regret of the
    softmax policy, exactly the offline metric).
Usage: SEEDS=0,1,2 LEARNERS=conv_ce,conv_q,conv_set,conv_regret FULL_H=1 \
       C:/Python314/python.exe -m scripts.offline_v75 DATA.npz [DATA2.npz ...]
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ragnarok.agent_v71.policy import PolicyNet, WIN

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
LEARN = os.environ.get("LEARNERS", "conv_ce,conv_q").split(",")
GROUP = os.environ.get("GROUP", "auto")
FULL_H = os.environ.get("FULL_H") == "1"
CLEAN = os.environ.get("CLEAN") == "1"
STEPS = int(os.environ.get("STEPS", "2000"))
WD = float(os.environ.get("WD", "0"))
NBOOT = 1000


def row_mode(bg):
    flat = (bg[..., 0].astype(np.int64) << 16) | (bg[..., 1].astype(np.int64) << 8) | bg[..., 2]
    out = np.zeros(bg.shape[:2] + (3,), np.uint8)
    for y in range(bg.shape[0]):
        v, c = np.unique(flat[y], return_counts=True)
        m = v[np.argmax(c)]
        out[y, :] = [(m >> 16) & 255, (m >> 8) & 255, m & 255]
    return out


def view(obs, bg, rm, me, prev_fg):
    """ego_view with the v75 switches (FULL_H, CLEAN); identical to policy.ego_view when both off."""
    fg = (obs != bg).any(-1)
    if CLEAN:
        fg &= (obs != rm).any(-1)
    fg = fg.astype(np.float32)
    H, W = fg.shape
    x, y = (W // 2, H // 2) if me is None else (int(me[0]), int(me[1]))
    x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    pad = WIN // 2
    big = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
    big[pad:pad + H, pad:pad + W] = fg
    crop = big[y:y + WIN, x:x + WIN].reshape(WIN // 2, 2, WIN // 2, 2).max((1, 3))
    pos = np.array([x / W, y / H], np.float32)
    gh = gw = WIN // 2
    if FULL_H:
        Hp, Wp = -(-H // gh) * gh, -(-W // gw) * gw
        fp = np.zeros((Hp, Wp), np.float32)
        fp[:H, :W] = fg
        g = fp.reshape(gh, Hp // gh, gw, Wp // gw).max((1, 3))
    else:
        g = fg[: H - H % gh, : W - W % gw].reshape(gh, H // gh, gw, W // gw).max((1, 3))
    pc, pg = prev_fg if prev_fg is not None else (crop, g)
    return np.stack([crop, pc, g, pg]).astype(np.uint8), pos, (crop, g)


def groups(Dz, obs, rid):
    if GROUP in ("auto", "sid") and "sid" in Dz.files:
        return np.asarray(Dz["sid"]), "sid"
    if GROUP == "rid":
        return rid, "rid"
    par = {int(r): int(r) for r in np.unique(rid)}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    seen = {}
    for i in range(len(rid)):
        h = hashlib.md5(obs[i].tobytes()).hexdigest()
        a = find(int(rid[i]))
        if h in seen and find(seen[h]) != a:
            par[find(seen[h])] = a
        seen.setdefault(h, int(rid[i]))
    return np.array([find(int(r)) for r in rid]), "link"


def label_of(Q, M, base):
    Qm = np.where(M, np.nan_to_num(Q, nan=-1e9), -1e9)
    top = Qm.max(1)
    lab = Qm.argmax(1)
    lab[Qm[:, base] == top] = base
    return lab


def train(Ci, Cp, Q, M, base, mode, seed):
    torch.manual_seed(seed)
    net = PolicyNet(M.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WD) if WD else torch.optim.Adam(net.parameters(), lr=1e-3)
    I = torch.tensor(Ci, device=DEV).float(); P = torch.tensor(Cp, device=DEV)
    Qt = torch.tensor(np.nan_to_num(Q), device=DEV); Mt = torch.tensor(M, device=DEV)
    Lt = torch.tensor(label_of(Q, M, base), device=DEV)
    n = len(Q)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (min(128, n),), device=DEV)
        lo, Mi, Qi = net(I[idx], P[idx]), Mt[idx], Qt[idx]
        lom = lo.masked_fill(~Mi, -1e9)
        top = Qi.masked_fill(~Mi, -1e9).max(1, keepdim=True).values
        if mode == "q":
            k = Mi.sum(1, keepdim=True)
            loss = ((((lo - (lo * Mi).sum(1, keepdim=True) / k) - (Qi - (Qi * Mi).sum(1, keepdim=True) / k)) ** 2) * Mi).sum(1).mean()
        elif mode == "ce":
            loss = F.cross_entropy(lom, Lt[idx])
        elif mode == "set":
            best = Mi & (Qi == top)
            onb = best[:, base:base + 1].expand_as(best)
            only_base = F.one_hot(torch.full((len(idx),), base, device=DEV), Mi.shape[1]).bool()
            best = torch.where(onb, only_base, best)
            loss = (torch.logsumexp(lom, 1) - torch.logsumexp(lom.masked_fill(~best, -1e9), 1)).mean()
        elif mode == "regret":
            p = F.softmax(lom, 1)
            loss = (p * (top - Qi) * Mi).sum(1).mean() + 0.01 * (p * F.log_softmax(lom, 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def choose(net, Ci, Cp, M):
    with torch.no_grad():
        lo = net(torch.tensor(Ci, device=DEV).float(), torch.tensor(Cp, device=DEV))
        lo = lo.masked_fill(~torch.tensor(M, device=DEV), -1e9)
    return lo.argmax(1).cpu().numpy()


def boot(vals, cl, rng_seed=0):
    ug = np.unique(cl)
    idx = {g: np.nonzero(cl == g)[0] for g in ug}
    rng = np.random.default_rng(rng_seed)
    m = []
    for _ in range(NBOOT):
        pick = rng.choice(ug, len(ug))
        m.append(np.mean(np.concatenate([vals[idx[g]] for g in pick])))
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main(paths):
    out = {}
    for p in paths:
        Dz = np.load(p, allow_pickle=True)
        obs, prev, me, Q, rid = Dz["obs"], Dz["prev"], Dz["me"], Dz["q"], Dz["rid"]
        pme = Dz["pme"] if "pme" in Dz.files else me
        mask, base, bg = Dz["mask"].astype(bool), int(Dz["base"]), Dz["bg"]
        rm = row_mode(bg) if CLEAN else None
        n, k = len(Q), len(mask)
        M = np.broadcast_to(mask, (n, k)).copy()
        CI = []
        for i in range(n):
            m = None if me[i][0] < 0 else (float(me[i][0]), float(me[i][1]))
            pm = None if pme[i][0] < 0 else (float(pme[i][0]), float(pme[i][1]))
            _, _, pf = view(prev[i], bg, rm, pm, None)
            img, pos, _ = view(obs[i], bg, rm, m, pf)
            CI.append((img, pos))
        Ci = np.stack([c[0] for c in CI]); Cp = np.stack([c[1] for c in CI])
        grp, gmode = groups(Dz, obs, rid)
        ug = np.unique(grp)
        fold_of = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 5))
        fid = np.array([fold_of[g] for g in grp])
        Qn = np.where(mask, np.nan_to_num(Q, nan=-1e9), -1e9)
        qb = Qn[:, base]
        val = {}                                           # arm -> per-row value of the choice
        dev = {}
        oracle = label_of(Q, M, base)
        val["oracle"] = Qn[np.arange(n), oracle]
        dev["oracle"] = (oracle != base)
        val["const"] = qb.copy(); dev["const"] = np.zeros(n, bool)
        cd = np.zeros(n, int)
        for f in range(5):
            tr, te = fid != f, fid == f
            adv = np.nanmean(np.where(mask, Q[tr] - Q[tr][:, [base]], np.nan), 0)
            cd[te] = int(np.nanargmax(np.where(mask, adv, -np.inf)))
        val["const_dev"] = Qn[np.arange(n), cd]; dev["const_dev"] = cd != base
        for name in LEARN:
            mode = name.split("_", 1)[1]
            ch = np.zeros((len(SEEDS), n), int)
            for si, seed in enumerate(SEEDS):
                for f in range(5):
                    tr, te = fid != f, fid == f
                    net = train(Ci[tr], Cp[tr], Q[tr], M[tr], base, mode, 1000 * seed + f)
                    ch[si, te] = choose(net, Ci[te], Cp[te], M[te])
            val[name] = np.mean([Qn[np.arange(n), ch[s]] for s in range(len(SEEDS))], 0)
            dev[name] = np.mean(ch != base, 0)
        ref = val["const_dev"]
        rows = {}
        for a, v in val.items():
            sdg = v - ref
            r = dict(gain_over_base=round(float(np.mean(v - qb)), 3), over_const_dev=round(float(np.mean(sdg)), 3),
                     ci=boot(sdg, grp), deviate=round(float(np.mean(dev[a])), 3))
            if a.startswith("conv_") and a != "conv_ce" and "conv_ce" in val:
                d = v - val["conv_ce"]
                r["vs_conv_ce"] = (round(float(np.mean(d)), 3), boot(d, grp))
            rows[a] = r
        game = str(Dz["game"])
        out[game] = dict(n=n, groups=len(ug), group_mode=gmode, const_dev_slots=np.bincount(cd, minlength=k).tolist(),
                         switches=dict(FULL_H=FULL_H, CLEAN=CLEAN, STEPS=STEPS, WD=WD, SEEDS=SEEDS), arms=rows)
        print(game, f"n={n} groups={len(ug)} ({gmode}) switches FULL_H={FULL_H} CLEAN={CLEAN}", flush=True)
        for a, r in rows.items():
            extra = f"  vs conv_ce {r['vs_conv_ce'][0]:+.3f} {r['vs_conv_ce'][1]}" if "vs_conv_ce" in r else ""
            print(f"   {a:12s} over_base {r['gain_over_base']:+.3f}  over_const_dev {r['over_const_dev']:+.3f} {r['ci']}"
                  f"  deviate {r['deviate']}{extra}", flush=True)
        tag = os.environ.get("TAG")
        if tag:                                            # per-row values, for paired comparisons across runs
            np.savez(f"craft_v6_out/v75/{tag}_{game}_rows.npz", grp=grp, **{a: v for a, v in val.items()})
    tag = os.environ.get("TAG")
    if tag:
        json.dump(out, open(f"craft_v6_out/v75/{tag}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
