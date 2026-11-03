"""Family W development objects and arm wiring (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8, §9, §12; v84 protocol
§6.1-6.2 and A.5-A.9; v84 build notes 8, 12, 14, 18). Evaluator side: builders read the v84 G1b expansion logs
(agent-side data: logs of the agent's own play on earlier games) and write hashed object files; the arms load
them and hand their arrays to ragnarok/agent_v85/critic_s.py as data.

Data. Per burned game h and G1b seed r in (9712, 9713): `<g1b>/<h>_<r>.npz` (checked against the run JSON's
npz_sha256, else Interruption): X = exp_feat (n, 44), ADV = exp_meta[:, 3], process = exp_meta[:, 2], block =
expansion index // 50, billed frames of each expansion = the difference of exp_meta[:, 4] (the first from the JSON's
entry_frames), and a row hash (blake2b of the float32 feature row, as v84 G0-S) for de-duplication.

Recipe (v84 A.5-A.8 on the 44 search-sense features, single ADV head, as in v84 G0-S `scripts/g0s_v84.py`):
- Shared geometry of unit g, identical for every arm of g: S_-g, C_-g and Q_-g of A.5 over the source games
  (g0_v84.geometry), null-space threshold 1e-2 lambda_max (v84 note 8; asserted below). Development: the other 40
  burned games, leave-one-game-out (LOGO). Sealed: all 41 burned games minus an optional exclusion list (§4.4,
  §9 near-siblings; which variant is primary is fixed at F2).
- Source heads once per source game (A.7, g0_v84.source_head): own standardisation, SD floor 0.05, one intercept
  per G1b run, rho by 2-fold cross-run validation, >= 30 positives and negatives; mapped to g (map_head).
- F_S: Sigma = I_d (the learner from scratch, isotropic kappa_F).
- X_S: norm[(1/G) sum u_h u_h^T + tau^2 I], tau^2 = 0.1 / d (A.8 X).
- P_S: the same construction from heads whose ADV labels are permuted within each source run, drawn with
  default_rng([9803, 2, i_h, r]) (i_h = h's index in pools_v83 DEV + CALIBRATION, r its G1b seed), then X's
  eigenvalues rank-matched (A.8 P).
- U_S: Q_-g axes ordered by C_-g eigenvalue with X's eigenvalues rank-matched, i.e. diag(lambda_X) (A.8 U). The
  recipe has no random draw: the §12 seed [9803, 3] is reserved and recorded, not consumed.
- T: "the T_clock prior" (v84 §6.2; build note 12): Sigma from source heads fitted on the 7 run-context features
  only (columns 19-25; the others set to 0, so the SD floor drops them), sigma_from_dirs, real labels. T is a
  critic prior with its own kappa (§4.4 tunes kappa for T), not v84 §6.1's Beta process schedule.
- K_S: "Sigma from this game's own G1b expansions": A.8's K* recipe, norm[u u^T + tau^2 I] with u from a head
  fitted in x_hat (g's shared geometry) on g's own G1b expansions (both runs, 9712 then 9713), one intercept,
  rho by 2-fold over even and odd 50-expansion blocks (g0_pipeline_v84.k_star). Development only.
- S: q* = argmax over Q1..Q4 of the equal-game mean, over the source games, of ADV per billed frame
  (sum of ADV / sum of billed frames of that process's expansions, both runs of a game pooled); ties to the
  lower process. V2s with only q*, no bandit choice (v84 §6.1).
- Dose (§8.2, §9, §12): one rule, an independent draw per unit, positions among g's included heads (pool
  order) drawn without replacement from default_rng([9804, p, i_g]) with i_g = UI[g] (`dose_dev`,
  `dose_sealed`). Development X4_S: p = 1, 2 draws of 4 heads. Sealed X4 and X12: p = 2 and p = 3, one draw
  each of 4 and 12 heads. UI[g] = GI[g] for a burned game (pools_v83 DEV + CALIBRATION order, the development
  pattern) and 41 + g's index in sorted(SEALED + SECOND) for a sealed or second-round game, which equals
  family M's CANON_UNITS row of that unit (objects_m_v85). Same geometry as X_S; kappa = kappa_X (as v84's X_8).
  Review fix W-1/M-5 (choice, to be ratified with family M's dose rule in a dated build note before F2): the
  sealed draws were default_rng([9804, 2]) / ([9804, 3]) with no unit index, i.e. one set of positions applied
  to each unit's own head list, so the 41-source units shared one game set and the 40-source near-sibling
  units got partly different ones; every unit now has its own draw, as in development and in family M.
- X12_S closure object in development (integration with v85_calib.arm_closures, which builds every arm module's
  hooks on a development unit at seed 9826 to hash its closure for F1): scripts/v85_objects.py also writes
  x12_s/<unit>_d1 for development units with X12's recipe (`dose_dev`: [9804, 3, i_g], 12 heads, one draw), as
  family M writes x12s_<unit>.json. No development row runs X12_S (§8.2: B1 only). Review fix W-3: on a
  development unit, X12_S at seed 9826 builds its hooks exactly (the closure hash) but its explorer refuses to
  run (`closure_only_explorer`); a sealed unit takes X4_S and X12_S only at the B1 2-run seeds 9853 and 9856,
  a development unit X4_S only at the G2 dose seeds 9826, 9827, 9829, 9830 (`dose_draw`, §12); a smoke run
  (seed 9800) takes draw 1.

kappa (§4.4; v84 A.9 as run by G0-S): grid 4^-5..1; tuning curves on the 4 room calibration games (pong,
breakout, seaquest, asteroids) x 2 directions (learning stream = one G1b run's expansions in order; test = the
other run's odd 50-expansion blocks, rows whose feature hash occurs in the learning run removed), where both have
>= 30 rows of each class. For calibration game c the Sigmas come from the other 40 games (and K_S from c's
LEARNING run only, A.9's K*_A). Learner: consequence.run_curve (checkpoints ceil(128 2^(k/4)), both-classes
rule). Criterion: mean over the tuning curves of the mean AUC over checkpoints; highest wins, exact ties to the
smaller kappa (g0_stages_v84.select_kappa). Edge rule: if any of F_S, X_S, K_S, P_S, U_S, T sits at a grid end,
the grid is extended once by a factor 4 on that side for every arm and the selection redone. X4_S and X12_S use
kappa_X. Choice: the test set is not subsampled to 5,000 rows (A.4's draw used a 97xx seed, which v85 avoids,
and §12 allots no seed to it); every odd-block row is scored.

Files under an objects root R (the arms use R = <v85 output root>/objects, i.e. Path(ctx.paths['v85'])/objects):
  R/w/<arm>/<unit>[_d<k>].npz   arrays (critic objects: m, sd, Q, Sigma; S: q_star, rate)
  R/w/<arm>/<unit>[_d<k>].json  sidecar: format, arm, unit, draw, recipe, sources, heads, seeds, input SHA-256s,
                                integrity, content_sha256 (SHA-256 over the sorted arrays' names, dtypes, shapes
                                and bytes: deterministic, unlike the npz zip bytes) and file_sha256
  R/w/kappa.json                the kappa rule's output (deterministic bytes)
  R/w/manifest.json             every object's hashes, kappa's hash, provenance; `hashes(R)` gives the F1 keys
                                development_objects and kappa.
A loaded object whose arrays do not match its content_sha256 raises. Run captions record the object's
content_sha256 and kappa.json's SHA-256 (A.5 "object hashes").
"""
import hashlib
import json
import os
import pathlib
import time

import numpy as np

from ragnarok.agent_v85.critic_s import CriticS, SingleProcess
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEALED, SECOND

G1B = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84\g1b")
GAMES = DEV + CALIBRATION                         # the 41 burned games (v84 G0-S order)
GI = {g: i for i, g in enumerate(GAMES)}
UI = dict(GI, **{g: len(GAMES) + i for i, g in enumerate(sorted(SEALED + SECOND))})   # per-unit dose index
SEEDS = (9712, 9713)
HEAD = "adv"
CLOCK_COLS = tuple(range(19, 26))                 # budget share .. new-cell rate (explorer2._features order)
KAPPA_GAMES = ("pong", "breakout", "seaquest", "asteroids")
KAPPA_GRID = tuple(4.0 ** k for k in range(-5, 1))
TUNED = ("f_s", "x_s", "k_s", "p_s", "u_s", "t")
KAPPA_OF = dict(f_s="f_s", x_s="x_s", k_s="k_s", p_s="p_s", u_s="u_s", t="t", x4_s="x_s", x12_s="x_s")
CRITIC_ARMS = tuple(KAPPA_OF)
DEV_ARMS = ("f_s", "x_s", "k_s", "p_s", "u_s", "t", "s", "x4_s")
SEALED_ARMS = ("f_s", "x_s", "p_s", "u_s", "t", "s", "x4_s", "x12_s")
SEED_P, SEED_U = (9803, 2), (9803, 3)
SEED_X4_DEV, SEED_X4_SEALED, SEED_X12_SEALED = (9804, 1), (9804, 2), (9804, 3)
DOSE = {"x4_s": {"dev": {9826: 1, 9829: 1, 9827: 2, 9830: 2},     # §12: G2 dose seeds, draw 1 / 2
                 "sealed": {9853: 1, 9856: 1}},                     # B1 2-run arms, the single sealed draw
        "x12_s": {"dev": {9826: 1},                                  # the F1 closure build only (never runs)
                  "sealed": {9853: 1, 9856: 1}}}                    # B1 only
CLOSURE_ONLY = {("x12_s", 9826)}                  # development (arm, seed) built for the closure hash only
SMOKE_SEED = 9800
FORMAT = "v85-w-object-1"
MIN_HEADS = 2                                     # as v84 arm_sigmas


# ---------------------------------------------------------------- hashing and files
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_sha256(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _replace(tmp, path):
    for t in range(20):                           # Windows: a scanner may hold the target for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if t == 19:
                raise
            time.sleep(0.25)


def _write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_np)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def obj_name(unit, draw=None):
    return unit if draw is None else f"{unit}_d{int(draw)}"


def write_object(root, arm, unit, arrays, meta, draw=None):
    """Write R/w/<arm>/<name>.npz and its sidecar; returns the sidecar dict."""
    d = pathlib.Path(root) / "w" / arm
    d.mkdir(parents=True, exist_ok=True)
    name = obj_name(unit, draw)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    npz = d / f"{name}.npz"
    tmp = d / f"{name}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, npz)
    side = dict(meta, format=FORMAT, arm=arm, unit=unit, draw=draw, content_sha256=content_sha256(arrays),
                file_sha256=file_sha256(npz), arrays={k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()})
    _write_json(d / f"{name}.json", side)
    return side


def load_object(root, arm, unit, draw=None):
    """(arrays, sidecar); raises FileNotFoundError if absent and RuntimeError if the arrays do not match."""
    d = pathlib.Path(root) / "w" / arm
    name = obj_name(unit, draw)
    npz, js = d / f"{name}.npz", d / f"{name}.json"
    if not npz.exists() or not js.exists():
        raise FileNotFoundError(f"family W object {arm}/{name} absent under {root} (built by objects_w_v85)")
    side = json.loads(js.read_text(encoding="utf-8"))
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    if side.get("format") != FORMAT or content_sha256(arrays) != side["content_sha256"]:
        raise RuntimeError(f"family W object {arm}/{name}: content hash mismatch")
    return arrays, side


def write_kappa(root, res):
    p = pathlib.Path(root) / "w" / "kappa.json"
    _write_json(p, dict(res, format=FORMAT))
    return file_sha256(p)


def load_kappa(root):
    p = pathlib.Path(root) / "w" / "kappa.json"
    if not p.exists():
        raise FileNotFoundError(f"family W kappa.json absent under {root} (built by objects_w_v85)")
    k = json.loads(p.read_text(encoding="utf-8"))
    if k.get("format") != FORMAT:
        raise RuntimeError("family W kappa.json: unknown format")
    return k, file_sha256(p)


def write_manifest(root, provenance=None):
    """R/w/manifest.json: every object sidecar's hashes and kappa.json's; returns hashes(R)."""
    root = pathlib.Path(root)
    objs = {}
    for js in sorted((root / "w").glob("*/*.json")):
        s = json.loads(js.read_text(encoding="utf-8"))
        objs.setdefault(s["arm"], {})[obj_name(s["unit"], s["draw"])] = dict(
            content_sha256=s["content_sha256"], file_sha256=s["file_sha256"], json_sha256=file_sha256(js))
    kp = root / "w" / "kappa.json"
    h = hashes(root, objs)
    _write_json(root / "w" / "manifest.json", dict(format=FORMAT, objects=objs, provenance=provenance,
                                                   kappa_sha256=file_sha256(kp) if kp.exists() else None, **h))
    return h


def hashes(root, objs=None):
    """F1 keys (A.10.6, build note 7): development_objects = SHA-256 over the sorted (arm, name, content_sha256)
    triples; kappa = SHA-256 of kappa.json."""
    root = pathlib.Path(root)
    if objs is None:
        objs = {}
        for js in sorted((root / "w").glob("*/*.json")):
            s = json.loads(js.read_text(encoding="utf-8"))
            objs.setdefault(s["arm"], {})[obj_name(s["unit"], s["draw"])] = dict(content_sha256=s["content_sha256"])
    h = hashlib.sha256()
    for arm in sorted(objs):
        for name in sorted(objs[arm]):
            h.update(f"{arm}\0{name}\0{objs[arm][name]['content_sha256']}\n".encode())
    kp = root / "w" / "kappa.json"
    return dict(development_objects=h.hexdigest(), kappa=file_sha256(kp) if kp.exists() else None)


# ---------------------------------------------------------------- data
def load_run(game, seed, g1b=G1B, n_max=None):
    """One G1b expansion stream (module docstring); n_max keeps its first n_max expansions (tests only)."""
    g1b = pathlib.Path(g1b)
    js = json.loads((g1b / f"{game}_{seed}.json").read_text(encoding="utf-8"))
    p = g1b / f"{game}_{seed}.npz"
    sha = file_sha256(p)
    if sha != js["npz_sha256"]:
        raise RuntimeError(f"Interruption: {p.name} does not match its run JSON's npz_sha256")
    with np.load(p) as z:
        X, meta = z["exp_feat"].astype(np.float32), np.asarray(z["exp_meta"], np.float64)
    entry = int(js["entry"]["entry_frames"])
    frames = np.diff(np.concatenate([[float(entry)], meta[:, 4]]))
    if n_max is not None:
        X, meta, frames = X[:n_max], meta[:n_max], frames[:n_max]
    hs = np.array([np.frombuffer(hashlib.blake2b(x.tobytes(), digest_size=8).digest(), np.uint64)[0] for x in X],
                  np.uint64)
    return dict(X=X, y={HEAD: meta[:, 3].astype(np.int8)}, block=(meta[:, 0] // 50).astype(np.int64),
                it=meta[:, 0].astype(np.int64), q=meta[:, 2].astype(np.int64), frames=frames, hash=hs,
                game=game, seed=int(seed), sha256=sha, n=int(len(X)), truncated=n_max is not None)


def load_data(games, g1b=G1B, n_max=None):
    return {g: [load_run(g, r, g1b, n_max) for r in SEEDS] for g in games}


def _inputs(data, games):
    return {f"{g}_{st['seed']}.npz": st["sha256"] for g in games for st in data[g]}


# ---------------------------------------------------------------- source heads, geometry, Sigmas
def source_heads(data, games, kinds=("real", "perm", "clock")):
    """{(h, kind): head or None} once per source game (A.7); kinds: real labels, labels permuted within each
    run (default_rng([9803, 2, i_h, r])), clock-only features (T_clock)."""
    from ragnarok.evaluation import g0_v84 as G
    out = {}
    for h in games:
        st = data[h]
        if "real" in kinds:
            out[(h, "real")] = G.source_head(st, HEAD)
        if "perm" in kinds:
            lab = [G.permuted_labels(s["y"][HEAD], np.random.default_rng([*SEED_P, GI[h], s["seed"]])) for s in st]
            out[(h, "perm")] = G.source_head(st, HEAD, lab)
        if "clock" in kinds:
            keep = np.zeros(st[0]["X"].shape[1], bool)
            keep[list(CLOCK_COLS)] = True
            masked = [dict(s, X=np.where(keep, np.asarray(s["X"], np.float64), 0.0)) for s in st]
            out[(h, "clock")] = G.source_head(masked, HEAD)
    return out


def game_sums(data, games):
    from ragnarok.evaluation import g0_v84 as G
    return {h: G.game_sums(data[h]) for h in games}


def critic_sigmas(sources, heads, sums):
    """Shared geometry over `sources` and the Sigmas of F_S, X_S, P_S, U_S, T (module docstring)."""
    from ragnarok.evaluation import g0_v84 as G
    assert G.EIG_REL == 1e-2, "null-space threshold 1e-2 (v84 note 8)"
    geo = G.geometry(sums, list(sources))
    d = geo["d"]
    real = [h for h in sources if heads.get((h, "real")) is not None]
    perm = [h for h in sources if heads.get((h, "perm")) is not None]
    clock = [h for h in sources if heads.get((h, "clock")) is not None]
    if min(len(real), len(perm), len(clock)) < MIN_HEADS:
        raise ValueError(f"fewer than {MIN_HEADS} usable source heads: real {len(real)}, perm {len(perm)}, "
                         f"clock {len(clock)}")
    V = np.array([G.map_head(heads[(h, "real")], geo) for h in real])
    S_X = G.sigma_from_dirs(V, d)
    lam_X = G.eig_desc(S_X)[0]
    S_P = G.spectrum_matched(G.sigma_from_dirs(np.array([G.map_head(heads[(h, "perm")], geo) for h in perm]), d), lam_X)
    S_T = G.sigma_from_dirs(np.array([G.map_head(heads[(h, "clock")], geo) for h in clock]), d)
    sig = dict(f_s=np.eye(d), x_s=S_X, p_s=S_P, u_s=np.diag(lam_X), t=S_T)
    integ = dict(trace_ok=bool(all(abs(np.trace(sig[a]) - d) < 1e-6 * d for a in sig)),
                 spectrum_ok=bool(all(float(np.max(np.abs(G.eig_desc(sig[a])[0] - lam_X) / np.abs(lam_X))) <= 1e-9
                                      for a in ("p_s", "u_s"))),
                 eig_rel=G.EIG_REL)
    info = dict(d=d, heads=dict(x_s=real, p_s=perm, t=clock),
                heads_missing=dict(x_s=[h for h in sources if h not in real], p_s=[h for h in sources if h not in perm],
                                   t=[h for h in sources if h not in clock]),
                integrity=integ, lam_geo=geo["lam"].tolist())
    return geo, sig, V, real, info


def k_sigma(streams, geo):
    """K_S: A.8's K* on the given streams in x_hat of `geo` (one intercept, rho by even/odd blocks); None if
    the head cannot be fitted."""
    from ragnarok.evaluation import g0_v84 as G
    from ragnarok.evaluation.g0_pipeline_v84 import k_star
    Z = np.vstack([G.project(s["X"], geo) for s in streams])
    y = np.concatenate([s["y"][HEAD] for s in streams]).astype(np.int8)
    block = np.concatenate([s["block"] for s in streams])
    return k_star(Z, y, block, geo["d"])


def best_process(data, sources, n_proc=4):
    """S's q*: argmax of the equal-game mean of ADV per billed frame (module docstring); returns (q*, rates)."""
    rows = []
    for h in sources:
        a, f = np.zeros(n_proc), np.zeros(n_proc)
        for st in data[h]:
            for q in range(n_proc):
                sel = st["q"] == q
                a[q] += float(st["y"][HEAD][sel].sum())
                f[q] += float(st["frames"][sel].sum())
        rows.append(np.where(f > 0, a / np.where(f > 0, f, 1.0), np.nan))
    rate = np.nanmean(np.array(rows), axis=0)
    return int(np.nanargmax(rate)), rate


def dose_picks(n_real, size, rng, draws):
    return [[int(i) for i in rng.choice(n_real, size, replace=False)] for _ in range(draws)]


def dose_dev(unit):
    """Development dose objects of `unit` (module docstring): X4_S, 2 draws of 4 ([9804, 1, i_g]); X12_S's
    closure object, 1 draw of 12 ([9804, 3, i_g]; written by scripts/v85_objects.py, never run)."""
    return {"x4_s": ((*SEED_X4_DEV, UI[unit]), 4, 2), "x12_s": ((*SEED_X12_SEALED, UI[unit]), 12, 1)}


def dose_sealed(unit):
    """Sealed dose objects of `unit` (§9, §12; module docstring): X4_S and X12_S, one draw each of 4 and 12,
    default_rng([9804, 2, i_g]) and default_rng([9804, 3, i_g])."""
    return {"x4_s": ((*SEED_X4_SEALED, UI[unit]), 4, 1), "x12_s": ((*SEED_X12_SEALED, UI[unit]), 12, 1)}


# ---------------------------------------------------------------- builders
def _critic_arrays(geo, S):
    return dict(m=np.asarray(geo["m"], np.float64), sd=np.asarray(geo["sd"], np.float64),
                Q=np.asarray(geo["Q"], np.float64), Sigma=np.asarray(S, np.float64))


def build_unit(root, unit, data, heads, sums, sources, arms=DEV_ARMS, dose=None, tag=""):
    """Write the objects `arms` of `unit` from `sources`. dose: {arm: (seed tuple, size, draws)}. tag: suffix of
    the unit name in the files (sealed variants). Returns {object name: sidecar}."""
    from ragnarok.evaluation import g0_v84 as G
    sources = list(sources)
    assert unit not in sources, "leave-one-game-out: the unit is never its own source"
    geo, sig, V, real, info = critic_sigmas(sources, heads, sums)
    name = unit + tag
    base = dict(sources=sources, inputs=_inputs(data, sources), d=info["d"], integrity=info["integrity"],
                lam_geo=info["lam_geo"], truncated=bool(any(st["truncated"] for h in sources for st in data[h])))
    if unit in data:
        Xu = np.vstack([np.asarray(s["X"], np.float64) for s in data[unit]])
        base["unit_outside_share"] = float(G.outside_share(Xu, geo))
    out = {}
    recipe = dict(f_s="Sigma = I (v84 A.8 F)", x_s="v84 A.8 X over the source heads", p_s="v84 A.8 P, labels "
                  "permuted within each source run, default_rng([9803, 2, i_h, r]); X's spectrum",
                  u_s="v84 A.8 U: diag(lambda_X) on Q axes (seed [9803, 3] reserved, no draw)",
                  t="T_clock: sigma_from_dirs of clock-only source heads (columns 19-25)")
    for arm in arms:
        if arm in sig:
            meta = dict(base, recipe=recipe[arm], heads=info["heads"].get(arm, []),
                        heads_missing=info["heads_missing"].get(arm, []),
                        seeds=dict(p_s=list(SEED_P), u_s=list(SEED_U)).get(arm))
            out[arm] = write_object(root, arm, name, _critic_arrays(geo, sig[arm]), meta)
        elif arm == "k_s":
            if unit not in data:
                raise ValueError(f"K_S needs {unit}'s own G1b expansions")
            S_K = k_sigma(data[unit], geo)
            if S_K is None:
                raise ValueError(f"K_S: no head on {unit}'s own expansions")
            meta = dict(base, recipe="v84 A.8 K* on the unit's own G1b expansions (both runs), in its shared "
                        "geometry", own_inputs=_inputs(data, [unit]),
                        integrity=dict(base["integrity"], trace_ok_k=bool(abs(np.trace(S_K) - info["d"]) < 1e-6 * info["d"])))
            out["k_s"] = write_object(root, "k_s", name, _critic_arrays(geo, S_K), meta)
        elif arm == "s":
            q, rate = best_process(data, sources)
            meta = dict(sources=sources, inputs=base["inputs"], truncated=base["truncated"],
                        recipe="v84 6.1 S: argmax of the equal-game mean ADV per billed frame; ties to the lower process")
            out["s"] = write_object(root, "s", name, dict(q_star=np.int64(q), rate=rate), meta)
        elif arm in ("x4_s", "x12_s"):
            seed, size, draws = dose[arm]
            if len(real) < size:
                raise ValueError(f"{arm}: {len(real)} usable heads < {size}")
            picks = dose_picks(len(real), size, np.random.default_rng(list(seed)), draws)
            for k, pick in enumerate(picks, 1):
                S_d = G.sigma_from_dirs(V[pick], info["d"])
                meta = dict(base, recipe=f"v84 A.8 X over {size} source heads drawn without replacement",
                            seeds=list(seed), draw_sources=[real[i] for i in pick],
                            integrity=dict(base["integrity"], trace_ok_dose=bool(abs(np.trace(S_d) - info["d"]) < 1e-6 * info["d"])))
                out[f"{arm}_d{k}"] = write_object(root, arm, name, _critic_arrays(geo, S_d), meta, draw=k)
        else:
            raise ValueError(f"unknown family W arm {arm}")
    return out


def build_development(root, units, g1b=G1B, data=None, games=GAMES, kappa_games=KAPPA_GAMES, kappa=True,
                      provenance=None):
    """Development objects (Part B item 13): per unit, LOGO over `games`, the arms DEV_ARMS (X4_S: 2 draws of 4,
    default_rng([9804, 1, i_unit]), `dose_dev`); then the kappa rule and the manifest. `data` (tests) replaces the
    G1b logs."""
    games = list(games)
    data = data if data is not None else load_data(games, g1b)
    heads = source_heads(data, games)
    sums = game_sums(data, games)
    built = {}
    for u in units:
        src = [h for h in games if h != u]
        built[u] = build_unit(root, u, data, heads, sums, src, DEV_ARMS, dose_dev(u))
    res = None
    if kappa:
        res = kappa_rule(data, heads, sums, [c for c in kappa_games if c in data], games)
        write_kappa(root, res)
    return dict(built=built, kappa=res, hashes=write_manifest(root, provenance))


def build_sealed(root, units, g1b=G1B, data=None, games=GAMES, exclude=None, tag="", provenance=None):
    """Sealed objects (§9, F2): sources = `games` minus exclude.get(unit, ()); X4 and X12 one draw each per unit,
    default_rng([9804, 2, i_unit]) and default_rng([9804, 3, i_unit]) (`dose_sealed`); no K_S. tag names a
    variant's files (e.g. '~nosib')."""
    games = list(games)
    data = data if data is not None else load_data(games, g1b)
    heads = source_heads(data, games)
    sums = game_sums(data, games)
    built = {}
    for u in units:
        src = [h for h in games if h != u and h not in (exclude or {}).get(u, ())]
        built[u] = build_unit(root, u, data, heads, sums, src, SEALED_ARMS, dose_sealed(u), tag=tag)
    return dict(built=built, hashes=write_manifest(root, provenance))


# ---------------------------------------------------------------- kappa rule (v84 A.9, edge rule)
def tuning_direction(streams, geo, a, b):
    """Learning run a (all expansions in order), test = run b's odd 50-expansion blocks minus the rows whose
    feature hash occurs in run a (no subsample)."""
    from ragnarok.evaluation import g0_v84 as G
    sA, sB = streams[a], streams[b]
    yA = np.asarray(sA["y"][HEAD], np.int8)
    odd = np.flatnonzero(sB["block"] % 2 == 1)
    keep = odd[~np.isin(sB["hash"][odd], sA["hash"])]
    yT = np.asarray(sB["y"][HEAD][keep], np.int8)
    return dict(ZA=G.project(sA["X"], geo), yA=yA, blockA=sA["block"], ZT=G.project(sB["X"][keep], geo), yT=yT,
                counts=(int((yA == 1).sum()), int((yA == 0).sum()), int((yT == 1).sum()), int((yT == 0).sum())),
                removed=1.0 - len(keep) / max(1, len(odd)), learn_seed=sA["seed"], test_seed=sB["seed"])


def tuning_prep(data, heads, sums, games=KAPPA_GAMES, pool=GAMES):
    """The tuning curves' inputs: [(game, direction, dd, {arm: Sigma})] and the skipped directions."""
    from ragnarok.evaluation.g0_pipeline_v84 import k_star
    prep, skipped = [], []
    for c in games:
        src = [h for h in pool if h != c]
        geo, sig, *_ = critic_sigmas(src, heads, sums)
        for di, (a, b) in enumerate(((0, 1), (1, 0))):
            dd = tuning_direction(data[c], geo, a, b)
            if min(dd["counts"]) < 30:
                skipped.append(dict(game=c, dir=di, counts=dd["counts"]))
                continue
            s = dict(sig)
            ks = k_star(dd["ZA"], dd["yA"], dd["blockA"], geo["d"])
            if ks is not None:
                s["k_s"] = ks
            prep.append((c, di, dd, s))
    return prep, skipped


def tuning_rows(prep, kappas, curve=None):
    """One row per (game, direction, arm, kappa): mean AUC over the checkpoints."""
    from ragnarok.agent_v84.consequence import run_curve
    curve = curve or (lambda dd, S, k: run_curve(dd["ZA"], dd["yA"], dd["ZT"], dd["yT"], S, k)[1])
    rows = []
    for c, di, dd, sig in prep:
        for arm in TUNED:
            if arm not in sig:
                continue
            for k in kappas:
                rows.append(dict(game=c, dir=di, arm=arm, kappa=float(k), mean_auc=float(np.mean(curve(dd, sig[arm], k)))))
    return rows


def select_kappa(rows, kappas):
    from ragnarok.evaluation.g0_stages_v84 import select_kappa as sel
    return sel(rows, list(kappas), arms=TUNED)


def edge_extension(sel, kappas):
    """The grid ends taken by any tuned arm (all six are decision arms here), each extended by a factor 4."""
    lo, hi = min(kappas), max(kappas)
    return ([lo / 4] if any(sel.get(a) == lo for a in TUNED) else []) + \
           ([hi * 4] if any(sel.get(a) == hi for a in TUNED) else [])


def kappa_rule(data, heads, sums, games=KAPPA_GAMES, pool=GAMES, grid=KAPPA_GRID, curve=None):
    """The §4.4 kappa rule (module docstring). Returns a JSON-ready dict; kappa[arm] is None for an arm with no
    tuning curve."""
    prep, skipped = tuning_prep(data, heads, sums, games, pool)
    grid = [float(k) for k in grid]
    rows = tuning_rows(prep, grid, curve)
    first = select_kappa(rows, grid)
    extra = edge_extension(first, grid)
    kappas = list(grid)
    sel = first
    if extra:
        rows += tuning_rows(prep, extra, curve)
        kappas = sorted(grid + extra)
        sel = select_kappa(rows, kappas)
    n_curves = {a: len({(r["game"], r["dir"]) for r in rows if r["arm"] == a}) for a in TUNED}
    return dict(kappa={a: sel.get(a) for a in TUNED}, derived={"x4_s": "x_s", "x12_s": "x_s"},
                grid=grid, extended=extra, kappas=kappas, first_pass=first, n_curves=n_curves, games=list(games),
                pool_size=len(pool), skipped=skipped, counts=[dict(game=c, dir=di, counts=dd["counts"],
                                                                   removed=dd["removed"], learn_seed=dd["learn_seed"])
                                                              for c, di, dd, _ in prep],
                inputs=_inputs(data, [g for g in pool if g in data]), rows=rows,
                criterion="mean over tuning curves of the mean AUC over checkpoints; ties to the smaller kappa")


# ---------------------------------------------------------------- arm wiring
def objects_root(ctx):
    return pathlib.Path(ctx.paths["v85"]) / "objects"


def dose_draw(arm, seed, smoke=False, *, unit):
    """The dose draw a run of `unit` uses (§12; module docstring). Smoke runs (seed 9800) use draw 1. A
    development (burned) unit: X4_S draw 1 on 9826/9829, draw 2 on 9827/9830; X12_S only at 9826, the F1
    closure build (CLOSURE_ONLY). A sealed unit: the single sealed draw at the B1 2-run seeds 9853 and 9856.
    Any other (unit pool, seed) is refused."""
    if smoke and int(seed) == SMOKE_SEED:
        return 1
    t = DOSE[arm]["dev" if unit in GI else "sealed"]
    if int(seed) not in t:
        raise ValueError(f"{arm}: seed {seed} has no dose draw on {'a development' if unit in GI else 'a sealed'} "
                         f"unit (§12)")
    return t[int(seed)]


def closure_only_explorer(reason, make=None):
    """An explorer factory for a run built only to hash its closure (§13.4, A.10.6; v85_job --closure-only):
    it builds the explorer exactly (make, default Explorer2s) and its run() refuses, so no frame is played."""
    def build(*a, **k):
        if make is None:
            from ragnarok.agent_v85.explorer2s import Explorer2s as mk
        else:
            mk = make
        ex = mk(*a, **k)

        def refuse(*_a, **_k):
            raise RuntimeError(reason)
        ex.run = refuse
        return ex
    return build


def hooks_w(ctx, arm):
    """Hooks of a family W critic arm: CriticS on the unit's object and the arm's kappa; the critic's identity
    (object content SHA-256, kappa.json SHA-256, kappa) is checked on resume (review fix W-2)."""
    from ragnarok.evaluation.senses_v85 import LOAD_FRAMES
    root = objects_root(ctx)
    draw = dose_draw(arm, ctx.seed, ctx.smoke, unit=ctx.game) if arm in DOSE else None
    arrays, side = load_object(root, arm, ctx.game, draw)
    kap, ksha = load_kappa(root)
    k = kap["kappa"].get(KAPPA_OF[arm])
    if k is None:
        raise ValueError(f"{arm}: no kappa for {KAPPA_OF[arm]}")
    ident = (side["content_sha256"], ksha, float(k))
    steer = CriticS(arrays["m"], arrays["sd"], arrays["Q"], arrays["Sigma"], float(k), restore_frames=LOAD_FRAMES,
                    ident=ident)
    closure_only = bool(not ctx.smoke and ctx.game in GI and (arm, int(ctx.seed)) in CLOSURE_ONLY)
    hooks = dict(steer=steer, w_object=dict(arm=arm, unit=ctx.game, draw=draw, content_sha256=side["content_sha256"],
                                            file_sha256=side["file_sha256"], kappa=float(k), kappa_of=KAPPA_OF[arm],
                                            kappa_sha256=ksha, d=int(arrays["Q"].shape[1]),
                                            n_sources=len(side.get("sources", [])), truncated=side.get("truncated"),
                                            closure_only=closure_only))
    if closure_only:
        hooks["explorer"] = closure_only_explorer(
            f"{arm} on development unit {ctx.game} at seed {ctx.seed}: built for the F1 closure hash only; "
            f"X12 runs in B1 only (§8.2, §9; seeds 9853/9856, §12)")
    return hooks


def hooks_s(ctx):
    arrays, side = load_object(objects_root(ctx), "s", ctx.game)
    q = int(arrays["q_star"])
    return dict(steer=SingleProcess(q), w_object=dict(arm="s", unit=ctx.game, content_sha256=side["content_sha256"],
                                                      file_sha256=side["file_sha256"], q_star=q,
                                                      rate=[float(r) for r in arrays["rate"]],
                                                      n_sources=len(side.get("sources", [])),
                                                      truncated=side.get("truncated")))


def captions_w(hooks, run):
    n = int(run["ex"].exp_n)
    return dict(hooks["w_object"], **hooks["steer"].caption(), expansions=n)
