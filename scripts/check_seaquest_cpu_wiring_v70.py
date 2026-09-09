"""Bounded exact geometry and no-oracle-leak checks before the Seaquest CPU chain."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames
from ragnarok.evaluation.pooled_seaquest_v70 import pooled_seaquest_ram_objects
from ragnarok.evaluation.fast_pooled_seaquest_v70 import fast_pooled_seaquest_ram_objects
from ragnarok.evaluation.seaquest_reference_pool_v70 import SeaquestReferenceEngine,SeaquestReferencePool
from scripts.capture_seaquest_pooled_v70 import _PairRAMTap
from scripts.bench_seaquest_reference_v70 import BASE,source_hashes,require_calibration


def check_geometry():
    proof=require_calibration()
    hashes=source_hashes()
    collections={}
    for name in ("development","reserved"):
        directory=BASE/name
        count=0
        for row in cold_pooled_fixture_frames(directory):
            pair=row["ram_previous"],row["ram_current"]
            if fast_pooled_seaquest_ram_objects(*pair)!=pooled_seaquest_ram_objects(*pair):
                raise AssertionError(f"Graph differs from sealed raster at {name}/{row['index']}")
            count+=1
        assert count==8192
        collections[name]=dict(decisions=count,fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest())
    assert hashes==source_hashes()
    return dict(kind="seaquest_graph_raster_full_fixture_equivalence",passed=True,collections=collections,
                source_sha256=hashes,calibration_sha256=proof,new_charged_frames=0,agent_training=False,gate_result=None)


def check_wiring():
    proof=require_calibration()  # before any new emulator construction
    hashes=source_hashes()
    reports=[]
    config=EmulatorConfig(game="seaquest",seed=18176,frame_skip=4,max_episode_frames=300,max_charged_frames=20000)
    left,right=SeaquestReferenceEngine(config),SeaquestReferenceEngine(config)
    right._decoder=lambda ram:[]
    right._pooled_decoder=lambda a,b:[]
    for step in range(128):
        a,b=left.step((step//4)%18),right.step((step//4)%18)
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    a,b=left.report(),right.report()
    assert a["self_generated_label_sha256"]==b["self_generated_label_sha256"]
    assert a["accounting"]==b["accounting"]
    assert a["perception"]["matched_visible"]>0 and b["perception"]["matched_visible"]==0
    assert left.reference.frames_seen==right.reference.frames_seen==128
    assert left._episode==right._episode==1
    reports.extend((a,b))

    config=EmulatorConfig(game="seaquest",seed=18177,frame_skip=4,max_episode_frames=509,max_charged_frames=20000)
    subject,witness=SeaquestReferenceEngine(config),_ColdPixelEngine(config)
    witness.set_native_wrapper(_PairRAMTap)
    for step in range(129):
        action=(step//4)%18
        pixels,_=subject.step(action)
        expected=witness.step(action)
        assert np.array_equal(pixels,expected)
        assert np.array_equal(subject.engine._ale.previous_ram,witness._ale.previous)
        assert np.array_equal(subject.engine._ale.current_ram,witness._ale.current)
    a=subject.report()
    assert a["branch_pairs"]==1 and a["decoded_branch_frames"]==0
    assert a["accounting"]["restores"]==2 and a["accounting"]["cold_loads"]==2
    reports.extend((a,dict(accounting=witness.accounting())))

    with SeaquestReferencePool([EmulatorConfig(game="seaquest",seed=18178,frame_skip=4,max_charged_frames=1000)]) as pool:
        for forbidden in ("getRAM","_capture","lives","reward","decode_seaquest_ram"):
            try:
                pool._one(0,forbidden)
            except ValueError as error:
                assert "Unknown" in str(error)
            else:
                raise AssertionError("Oracle RPC must be unavailable")
        pixels,boxes=pool.step([3])[0]
        assert pixels.dtype==np.uint8 and pixels.shape==(210,160,3) and boxes.shape[1]==4
        reports.append(pool.report())
        processes=list(pool._processes)
    assert all(not p.is_alive() for p in processes)
    assert hashes==source_hashes() and proof==require_calibration()
    return dict(kind="seaquest_reference_cpu_wiring_check",passed=True,source_sha256=hashes,calibration_sha256=proof,
                engine_reports=reports,new_charged_frames=sum(r["accounting"]["charged_frames"] for r in reports),
                checks=["RGB/boxes/C invariant to removed oracle labels","terminal branch source pair and cold restart alignment",
                        "closed RPC observations and subprocess cleanup"],agent_training=False,gate_result=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("geometry","wiring"))
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve previous checks")
    result=check_geometry() if args.mode=="geometry" else check_wiring()
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("kind","passed","new_charged_frames")},indent=2))
