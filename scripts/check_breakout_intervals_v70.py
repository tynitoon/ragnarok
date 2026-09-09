"""Check exact RAM-only optimization against the sealed temporal raster judge."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames
from ragnarok.evaluation.fast_pooled_breakout_v70 import pooled_breakout_ram_intervals
from ragnarok.evaluation.pooled_breakout_v70 import pooled_breakout_ram_objects


def check(fixtures):
    sources=("scripts/check_breakout_intervals_v70.py","ragnarok/evaluation/fast_pooled_breakout_v70.py",
             "ragnarok/evaluation/pooled_breakout_v70.py","ragnarok/evaluation/breakout_v70.py",
             "ragnarok/evaluation/cold_pooled_fixtures_v70.py","ragnarok/evaluation/pong_v70.py")
    root=Path(__file__).resolve().parents[1]
    counts=0
    seconds=[0.,0.]
    digest=hashlib.sha256()
    for fixture in fixtures:
        for frame in cold_pooled_fixture_frames(fixture):
            args=frame["ram_previous"],frame["ram_current"]
            started=time.perf_counter()
            reference=pooled_breakout_ram_objects(*args)
            seconds[0]+=time.perf_counter()-started
            started=time.perf_counter()
            candidate=pooled_breakout_ram_intervals(*args)
            seconds[1]+=time.perf_counter()-started
            if candidate!=reference:
                raise AssertionError(f"Geometry changed: {fixture}, window {frame['index']}")
            digest.update(repr(candidate).encode())
            counts+=1
    return dict(kind="exact_breakout_temporal_implementation_equivalence",windows=counts,all_equal=True,
                object_label_sha256=digest.hexdigest(),raster_seconds=seconds[0],interval_seconds=seconds[1],
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                fixture_manifests={str(p):hashlib.sha256((p/"manifest.json").read_bytes()).hexdigest() for p in fixtures},
                new_emulated_frames=0,gate_result=None,agent_training_allowed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",nargs="+",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve previous results")
    result=check(args.fixtures)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ("windows","all_equal","raster_seconds","interval_seconds")},indent=2))
