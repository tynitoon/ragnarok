"""Short charged development CPU profile, to locate overhead rather than score a gate."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import time

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.reference_pool_v70 import _ReferenceEngine


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game",choices=("pong","breakout"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior profiles")
    engine=_ReferenceEngine(EmulatorConfig(game=args.game,seed=18000,max_charged_frames=100_000),
                            observation_mode="maxpooled",reset_convention="cold")
    profile=cProfile.Profile()
    started=time.perf_counter()
    with profile:
        for step in range(512):
            engine.step((step//4)%18)
    elapsed=time.perf_counter()-started
    stream=io.StringIO()
    pstats.Stats(profile,stream=stream).sort_stats("cumulative").print_stats(35)
    result=dict(kind="charged_single_worker_cpu_profile",game=args.game,elapsed_seconds=elapsed,
                decisions=512,report=engine.report(),profile=stream.getvalue(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gate_result=None,
                limitation="Profile instrumentation and no IPC; for hotspot diagnosis only.")
    with args.out.open("x",encoding="utf-8") as target:
        json.dump(result,target,indent=2)
    print(stream.getvalue())
