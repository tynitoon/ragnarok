"""start_v85 (A.2-A.4): units, counted-rung rule, and one real start (top lineage, identity, Body cache)."""
import numpy as np
import pytest

from ragnarok.evaluation import start_v85 as S


def test_units_and_order():
    u = S.eligible_units()
    assert len(u) == 16 and all(k >= 4 for k in u.values())
    assert S.unit_order(u) == S.unit_order(list(reversed(sorted(u)))) and sorted(S.unit_order(u)) == sorted(u)


def test_counted_rungs_rule():
    lad = [0, 10, 20, 40, 80, 160]
    # records [billed, cum]: run 1 and run 2
    r1 = [[0, 0], [100_000, 12], [120_000, 25], [400_000, 90], [900_000, 170]]
    r2 = [[0, 0], [200_000, 15], [300_000, 30], [600_000, 85], [999_000, 165]]
    # k=2: gaps 20k/100k -> median 60k >= 50k kept; k=3 and k=4 reached by the same record on both runs
    # -> merged, the lowest (3) kept; k=5 kept
    assert S.counted_rungs(lad, 5, [12, 15], [r1, r2]) == [2, 3, 5]
    # S(s1) above M2 at one start removes k=2
    assert S.counted_rungs(lad, 5, [12, 21], [r1, r2]) == [3, 5]
    # gap rule
    r3 = [[0, 0], [100_000, 12], [110_000, 25], [400_000, 90], [900_000, 170]]
    assert S.counted_rungs(lad, 5, [12], [r3, r3]) == [3, 5]


@pytest.mark.skipif(not (S.V84 / "g1b" / "alien_9712.npz").exists(), reason="v84 G1b logs absent")
def test_real_start_alien_A(tmp_path):
    g, ks = "alien", S.eligible_units()["alien"]
    lad = S.ladder(g)
    s1 = S.top_lineage(g, S.REF_SEEDS["A"], lad, ks)
    assert s1 is not None and s1["S"] >= lad[1] and len(s1["actions"]) == len(s1["cum"])
    ok = S.start_identity(g, S.ENTRY_SEEDS["A"], s1, lad, ks)
    assert ok["ok"], ok
    c = S.entry_cache(g, S.ENTRY_SEEDS["A"], tmp_path / "body.pkl")
    body = S.load_body(c["path"], c["sha256"])
    assert body.tracker is not None and c["entry_frames"] > 0
    with pytest.raises(AssertionError):
        S.load_body(c["path"], "0" * 64)
