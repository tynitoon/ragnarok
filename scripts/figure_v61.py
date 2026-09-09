"""ARC 3 — THE FIGURE (ARC3_PLAN.md 8), built from JSON only (v61_confirm.json + v61_confirm_score.json,
optional v61_probe.json / v61_pre_s0.json for the second panel). Never reads a run's internals.

Top row: one panel per tier position — MEAN MASTERY (the verdict's object) vs practice rounds b, for M4,
fresh (mean of Fa/Fb, band = their spread), L (dashed ceiling), G' (dotted) where present. b = 0 is drawn
and labelled by the recorded store regime: weights only at the first goal, weights plus carried store
at later goals when the store persists (v62, ARC3_PLAN.md 12.6). Caption: verdict, Delta_learn +/- se, per-tier Delta, endpoint
gap, the wording rule's sentence, the mechanism line with the roster ceiling.
Bottom row: the accumulation panel (4.5 ii by default): first-proposal lawfulness of M1..M4 on the probe
world, with the collinearity caveat.

Usage: python -m scripts.figure_v61 [--out craft_v6_out/v61_figure.png]
"""

import argparse
import json
import os
import statistics as st
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def curves(units, arm, pos):
    return [u["arms"][arm][pos]["curve"] for u in units if u["arms"].get(arm)]


def mean_curve(cs):
    return [st.mean(c[b] for c in cs) for b in range(len(cs[0]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="craft_v6_out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = json.load(open(os.path.join(a.dir, "v61_confirm.json")))
    sc = json.load(open(os.path.join(a.dir, "v61_confirm_score.json")))
    units, B = data["units"], data["b_max"]
    store_persists = data.get("store_persists")
    if type(store_persists) is not bool or any(u.get("store_persists") != store_persists for u in units):
        raise ValueError("Figure requires explicit, consistent per-world store metadata")
    store_caption = "store persists across goals within each world" if store_persists else "store resets for each goal"
    n_pos = len(units[0]["arms"]["M4"])
    probe = json.load(open(os.path.join(a.dir, "v61_probe.json"))) if os.path.exists(os.path.join(a.dir, "v61_probe.json")) else None
    fig = plt.figure(figsize=(4.6 * n_pos, 7.8))
    grid = fig.add_gridspec(2, n_pos, height_ratios=[3, 1.6])
    axes = [fig.add_subplot(grid[0, p]) for p in range(n_pos)]
    bs = list(range(B + 1))
    roles = {2: "the rung", 3: "learns-faster position", 4: "reach position"}
    for p in range(n_pos):
        ax = axes[p]
        tier = units[0]["arms"]["M4"][p]["tier"]
        fa, fb = mean_curve(curves(units, "Fa", p)), mean_curve(curves(units, "Fb", p))
        fresh = [(x + y) / 2 for x, y in zip(fa, fb)]
        lo, hi = [min(x, y) for x, y in zip(fa, fb)], [max(x, y) for x, y in zip(fa, fb)]
        ax.fill_between(bs, lo, hi, color="tab:gray", alpha=0.25, label="fresh: Fa/Fb spread")
        ax.plot(bs, fresh, "o-", color="tab:gray", label="fresh agent (mean of two)")
        ax.plot(bs, mean_curve(curves(units, "M4", p)), "s-", color="tab:red", label="experienced agent (4 worlds)")
        if any(u["arms"].get("L") for u in units):
            ax.plot(bs, mean_curve(curves(units, "L", p)), "--", color="black", label="law-knower (ceiling)")
        g = curves(units, "G", p)
        if g:
            ax.plot(bs, mean_curve(g), ":", color="tab:blue", label="store sweeper, no law")
        t = sc["per_tier"][str(tier)] if str(tier) in sc["per_tier"] else sc["per_tier"][tier]
        ax.set_title(f"tier-{tier} goal — {roles.get(tier, '')}\nDelta {t['delta']:+.3f} (se {t['se']:.3f})\n"
                     f"head start {t['delta_0']:+.2f} | end gap {t['endpoint_gap']:+.2f}", fontsize=10)
        initial_sources = "weights + carried store" if store_persists and p > 0 else "weights only"
        ax.set_xlabel(f"practice rounds on this goal (b)\nb=0 head start: {initial_sources}", fontsize=9)
        ax.set_xticks(bs)
        ax.set_ylim(-0.02, 1.02)
        if p == 0:
            ax.set_ylabel("mastery (mean over units)")
            ax.legend(fontsize=8, loc="lower right")
    # second row: accumulation
    ax2 = fig.add_subplot(grid[1, :])
    if probe:
        names = [k for k in probe["arms"] if k.startswith("M")] + ["R"]
        vals = [probe["arms"][k]["lawful"] for k in names]
        ax2.bar(names, vals, color=["tab:red"] * (len(names) - 1) + ["tab:gray"])
        ax2.set_xticks(range(len(names)), [f"{k}\n(n = {probe['arms'][k]['n']})" for k in names])
        ax2.axhline(0.5, ls=":", color="k", lw=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("first proposals that are lawful")
        ax2.set_title("first-proposal lawfulness (probe world 8150; R = untrained).\n"
                      "Unequal sample sizes; worlds-lived-in is collinear with gradient steps.", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "probe not run", ha="center")
    mech = ""
    if probe and probe.get("summary"):
        s = probe["summary"]
        rc = s.get("roster_ceiling", {})
        mech = (f"mechanism: first-proposal lawfulness M {s['r_M']:.2f} vs untrained {s['r_R']:.2f}; "
                f"roster-only ceiling AUC {rc.get('0.0', {}).get('auc_unexercised', float('nan')):.2f}")
    cap = (f"VERDICT: {sc['verdict']} — \"{sc['sentence']}\"   Delta_learn {sc['delta_learn']:+.3f} ± {sc['se']:.3f} "
           f"(N {sc['N']} unseen worlds, {sc['B']} rounds per goal, both arms learning; {store_caption}; "
           f"contemporaneous null from two fresh replicates).  {mech}")
    fig.text(0.01, 0.01, textwrap.fill(cap, width=220, break_long_words=False), fontsize=8)
    fig.suptitle("ARC 3 — does an agent that lived in four worlds of this chemistry learn a fifth faster?", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93), h_pad=2.7)
    out = a.out or os.path.join(a.dir, "v61_figure.png")
    fig.savefig(out, dpi=140)
    print("written", out)


if __name__ == "__main__":
    main()
