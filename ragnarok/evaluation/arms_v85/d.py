"""D, the plumbing control (docs/ARC4_V85_PROTOCOL_EN.md A.7, A.10.2, §4.3): expansion 0's rollout plays the
start's own lineage actions (ALE indices) open-loop from s1, with ordinary puts, up to and including the first
step with cum >= M_{k_ref} (oracle_v85.d_script, through Explorer2s(script=...)); the job's evaluator stop at
M_{k_ref} then ends the run at the next restore. Agent seed E_s (§12: 9819 / 9820; 9800 in smoke runs),
asserted here. 1 run per start (I-stage); D control r_g(D:V2s) >= 4 on >= 80% of units (verdict).

Captions: the script's length and SHA-256, the lineage length above s1, M_{k_ref}; `stop_at_script_end`
(the evaluator stop fired on the clone of the script's last step: the first rollout completed, exp_n = 1, and
the first record with cum >= M_{k_ref} is at depth = script length); `lineage_match` (every record of
expansion 0's rollout has the lineage's own cum at its depth: a diagnostic beside the job's A.5 replay
confirmation, which is the D confirmation). Choice: a record belongs to expansion 0's rollout when its depth
is in 1..len(script) and it precedes the first record of a later expansion (billed <= exp_meta[0] billed).
"""
import hashlib

import numpy as np

from ragnarok.evaluation.oracle_v85 import d_script


def build(ctx):
    assert ctx.seed == ctx.E, f"D's agent seed is E_s = {ctx.E} (A.7, §12), not {ctx.seed}"
    script = d_script(ctx.s1, ctx.ladder, ctx.k_ref)
    return dict(kwargs=dict(script=script), script=script)


def captions(hooks, run):
    s, ctx, tr, ex = hooks["script"], run["ctx"], run["trace"], run["ex"]
    n, m = len(s), float(ctx.ladder[ctx.k_ref])
    lin = np.asarray(ctx.s1["cum"], np.float64)
    end0 = ex.exp_meta[0][4] - run["env"].clock_start if ex.exp_meta else None   # billed at expansion 0's end
    mine = [r for r in tr.records if 0 < r[4] <= n and (end0 is None or r[0] <= end0)]
    top = next((r for r in tr.records if r[1] >= m), None)
    return dict(script_steps=n, script_sha256=hashlib.sha256(np.asarray(s, np.uint8).tobytes()).hexdigest(),
                lineage_steps=int(len(ctx.s1["actions"])), k_ref=int(ctx.k_ref), M_k_ref=m,
                stop_at_script_end=bool(tr.stopped and ex.exp_n == 1 and top is not None and top[4] == n),
                records_in_script=len(mine),
                lineage_match=bool(mine and all(float(r[1]) == float(lin[r[4] - 1]) for r in mine)))
