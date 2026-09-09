"""Fixed complete Seaquest CPU chain, gated by the committed reserved proof."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.seaquest_reference_pool_v70 import SeaquestReferencePool
from scripts.validate_seaquest_pooled_v70 import SOURCE_PATHS as SEALED_SOURCES,verify_seal,_require_committed
from scripts.calibrate_seaquest_coverage_v70 import finish

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"craft_v6_out/v70_seaquest_pooled"
PROTOCOL="docs/ARC4_SEAQUEST_CPU_PROTOCOL.json"
SOURCE_PATHS=tuple(dict.fromkeys((*SEALED_SOURCES,PROTOCOL,
    "scripts/bench_seaquest_reference_v70.py","scripts/check_seaquest_cpu_wiring_v70.py",
    "ragnarok/evaluation/seaquest_reference_pool_v70.py","ragnarok/evaluation/reference_pool_v70.py",
    "ragnarok/evaluation/fast_pooled_seaquest_v70.py","ragnarok/evaluation/seaquest_scene_v70.py",
    "scripts/ref_extract_v70.py","ragnarok/evaluation/perception_v70.py",
    "ragnarok/evaluation/scene_perception_v70.py","ragnarok/evaluation/scene_adapters_v70.py",
    "ragnarok/evaluation/pooled_pong_v70.py","ragnarok/evaluation/breakout_v70.py",
    "ragnarok/evaluation/pooled_breakout_v70.py","ragnarok/evaluation/fast_pooled_breakout_v70.py",
    "ragnarok/evaluation/freeway_v70.py","ragnarok/evaluation/pooled_freeway_v70.py",
    "ragnarok/native/reference_update_v70.c","ragnarok/evaluation/native_reference_v70.py",
    "scripts/build_reference_native_v70.py")))


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def require_calibration():
    seal=verify_seal(BASE)
    path=BASE/"reserved_result.json"
    report=json.loads(path.read_text(encoding="utf-8"))
    if (report.get("kind")!="seaquest_two_raw_frame_reserved_calibration" or report.get("seed")!=17322 or
            report.get("calibration_pass") is not True or report.get("source_sha256")!=seal["source_sha256"] or
            report.get("seal_sha256")!=hashlib.sha256((BASE/"reserved_seal.json").read_bytes()).hexdigest() or
            report.get("fixture_manifest_sha256")!=hashlib.sha256((BASE/"reserved/manifest.json").read_bytes()).hexdigest()):
        raise ValueError("Complete committed reserved temporal proof required")
    if set(report["per_type"])!={"player","shark","submarine","diver","enemy_missile","player_missile","surface_submarine"}:
        raise ValueError("Missing reserved categories")
    for s in report["per_type"].values():
        recomputed=dict(s)
        finish(recomputed)
        if recomputed!=s or not s["passed"] or s["frames"]!=8192:
            raise ValueError("Reserved counts do not meet the unchanged criteria")
    for file in (path,BASE/"reserved/manifest.json",*(ROOT/p for p in SOURCE_PATHS)):
        _require_committed(file)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(*,reference_backend,native_build=None):
    proof=require_calibration()
    hashes=source_hashes()
    checks={}
    for name,kind in (("graph_raster_equivalence.json","seaquest_graph_raster_full_fixture_equivalence"),
                      ("cpu_wiring.json","seaquest_reference_cpu_wiring_check")):
        path=BASE/name
        check=json.loads(path.read_text(encoding="utf-8"))
        if (check.get("kind")!=kind or check.get("passed") is not True or
                check.get("source_sha256")!=hashes or check.get("calibration_sha256")!=proof):
            raise ValueError("Complete current geometry and wiring checks required before benchmark")
        _require_committed(path)
        checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    native_provenance=None
    if reference_backend=="native":
        if native_build is None:
            raise ValueError("Native build must be explicit")
        native_provenance=dict(manifest_sha256=hashlib.sha256(native_build.read_bytes()).hexdigest(),
                               build=json.loads(native_build.read_text(encoding="utf-8")))
    configs=[EmulatorConfig(game="seaquest",seed=18000+i,frame_skip=4,max_charged_frames=100000) for i in range(16)]
    started=time.perf_counter()
    with SeaquestReferencePool(configs,reference_backend=reference_backend,native_build=native_build) as pool:
        startup_seconds=time.perf_counter()-started
        tick=time.perf_counter()
        for step in range(2048):
            outputs=pool.step([(step//4+i)%18 for i in range(16)])
            if any(rgb.shape!=(210,160,3) or boxes.ndim!=2 or boxes.shape[1]!=4 for rgb,boxes in outputs):
                raise RuntimeError("Malformed pixel-derived observation")
        work_seconds=time.perf_counter()-tick
        reports=[pool.report(i) for i in range(16)]
    if source_hashes()!=hashes or require_calibration()!=proof:
        raise RuntimeError("Sources or calibration changed during the benchmark")
    return dict(kind="seaquest_maxpooled_repeat4_cpu_chain_benchmark",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                reference_backend=reference_backend,native_reference_build=native_provenance,
                calibration_sha256=proof,source_sha256=hashes,checks_sha256=checks,workers=16,ordinary_decisions_per_worker=2048,
                action_repeat=4,branch_every_decisions=128,startup_seconds=startup_seconds,work_seconds=work_seconds,
                ordinary_agent_steps_per_second=32768/work_seconds,worker_reports=reports,
                charged_frames=sum(r["accounting"]["charged_frames"] for r in reports),logical_processors=os.cpu_count(),
                versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy","scipy")},gpu_used=False,gate_result=None,
                limitations=["Full RGB IPC, modal reference, every-real-frame oracle scoring and charged branches included.",
                             "ARC3 GPU training concurrently active; startup separate; no controlled universal speedup claim.",
                             "Moving components inherit any moving constituent allocation slot, not persistent physical identities.",
                             "No learned parser or game skill, and no reference training on reserved fixtures."])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-backend",choices=("numpy","native"),required=True)
    parser.add_argument("--native-build",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve the previous measurement")
    result=benchmark(reference_backend=args.reference_backend,native_build=args.native_build)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("reference_backend","startup_seconds","work_seconds","ordinary_agent_steps_per_second","charged_frames")},indent=2))
