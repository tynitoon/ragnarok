"""One reserved Seaquest pair, guarded by committed sources and metadata.

Development scoring modules remain unchanged; this separate executable can only
open the two preregistered reserved seeds after every prerequisite is verified.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.seaquest_v70 import KINDS, REVISION, decode_seaquest_ram, screen_seaquest_objects
from scripts.calibrate_seaquest_v70 import SOURCE_PATHS as RAW_SOURCES
from scripts.calibrate_seaquest_coverage_v70 import SOURCE_PATHS as COVERAGE_SOURCES, new_stats, accumulate, finish
from scripts.capture_cold_v70 import capture as capture_raw
from scripts.seaquest_coverage_controller_v4 import CoverageController
from scripts.committed_seal_v70 import verify_committed_seal

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_SEAQUEST_RESERVED_PROTOCOL.json"
SOURCE_PATHS=tuple(dict.fromkeys((*COVERAGE_SOURCES, PROTOCOL,
    "scripts/committed_seal_v70.py", "scripts/validate_seaquest_reserved_v70.py")))
DEVELOPMENT={
    17311:"development_v5.json", 17331:"coverage_geometry_v5.json",
    17341:"coverage2_geometry_v5.json", 17351:"coverage3_geometry_v5.json",
    17361:"coverage4_geometry_v5.json",
}


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def _write(path, value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def geometry_pass(report, *, additional):
    for s in report["per_type"].values():
        if s["presence_fraction"]<.99:
            return False
        if s["ram_objects"] or s["pixel_objects"]:
            if not s["joint_matches"] or s["box_fraction"]<.99:
                return False
            if additional and (s["positive_presence_fraction"] is None or s["positive_presence_fraction"]<.99):
                return False
    return True


def development_evidence(base, source_hashes):
    evidence={}
    for seed,name in DEVELOPMENT.items():
        path=base/name
        report=json.loads(path.read_text(encoding="utf-8"))
        sources=RAW_SOURCES if seed==17311 else COVERAGE_SOURCES
        kinds=set(KINDS) if seed==17311 else set((*KINDS,"surface_submarine"))
        expected_kind="independent_seaquest_raw_calibration" if seed==17311 else "seaquest_development_coverage_geometry"
        if (report.get("seed")!=seed or report.get("kind")!=expected_kind or
                set(report["per_type"])!=kinds or
                report["source_sha256"]!={p:source_hashes[p] for p in sources}):
            raise ValueError(f"Development provenance mismatch: {name}")
        if not geometry_pass(report,additional=seed!=17311):
            raise ValueError(f"Development geometry remains negative: {name}")
        if seed==17361 and not all(s["passed"] and s["joint_matches"]>=50 for s in report["per_type"].values()):
            raise ValueError("Final coverage development must pass every category and surface subtype")
        evidence[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return evidence


def create_seal(base):
    source_hashes=hashes()
    evidence=development_evidence(base,source_hashes)
    commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    for path in SOURCE_PATHS:
        current=subprocess.check_output(["git","-C",str(ROOT),"hash-object",f"--path={path}",str(ROOT/path)],text=True).strip()
        committed=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",f"HEAD:{path}"],text=True).strip()
        if current!=committed:
            raise ValueError(f"Commit the final source before creating the seal: {path}")
    seal=dict(kind="seaquest_reserved_pair_seal",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              source_commit=commit,source_sha256=source_hashes,development_results=evidence,
              raw_seed=17312,coverage_seed=17362,review_required_before_capture=True)
    _write(base/"reserved_pair_seal.json",seal)
    return seal


def verify_seal(base):
    seal=verify_committed_seal(ROOT,(base/"reserved_pair_seal.json").relative_to(ROOT),SOURCE_PATHS)
    if seal.get("raw_seed")!=17312 or seal.get("coverage_seed")!=17362:
        raise ValueError("Reserved seed pair differs from the protocol")
    if seal.get("development_results")!=development_evidence(base,seal["source_sha256"]):
        raise ValueError("Development evidence differs from the committed seal")
    return seal


def _capture_coverage(destination, source_hashes):
    root=ROOT
    protocol=json.loads((root/"docs/ARC4_SEAQUEST_COVERAGE_V4_PROTOCOL.json").read_text(encoding="utf-8"))
    seed=17362
    hashes=source_hashes
    destination.mkdir(parents=True)
    frames = protocol["frames"]
    engine = _ColdPixelEngine(EmulatorConfig(game="seaquest", seed=seed, frame_skip=1, mode="live",
                max_episode_frames=protocol["episode_limit"], max_charged_frames=protocol["max_charged_frames"]))
    controller = CoverageController()
    images, ram, episodes, actions, charges, shards, previews = [], [], [], [], [], [], []
    episode = 0
    reset_spent = engine.accounting()["reset_emulated_frames"]
    started = time.perf_counter()
    for index in range(1, frames+1):
        if (index-1) % protocol["action_repeat"] == 0:
            action = controller.action(engine.observe(raw=True))
        engine.step(action)
        report = engine.accounting()
        if report["reset_emulated_frames"] != reset_spent:
            episode += 1
            reset_spent = report["reset_emulated_frames"]
        raw = engine.observe(raw=True)
        images.append(raw)
        ram.append(engine._ale.getRAM().copy())  # recorder only; never passed to controller
        episodes.append(episode)
        actions.append(action)
        charges.append(report["emulated_frames"])
        if index in (1, 128, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, frames):
            previews.append((index, raw.copy()))
        if len(images) == 256 or index == frames:
            path = destination/f"frames_{index-len(images)+1:05d}_{index:05d}.npz"
            np.savez_compressed(path, rgb=np.stack(images), ram=np.stack(ram),
                                episode=np.array(episodes, dtype=np.int32), action=np.array(actions, dtype=np.uint8),
                                charged_frame=np.array(charges, dtype=np.int64))
            shards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(images)))
            images, ram, episodes, actions, charges = [], [], [], [], []
        if index % 4096 == 0:
            print(f"coverage {index}/{frames}; charged={report['charged_frames']}; episodes={episode+1}", flush=True)
    if hashes != {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Capture sources changed during the run; preserve this incomplete capture")
    final = engine.accounting()
    if (final["charged_frames"]-128*final["cold_loads"] != frames or
            final["cold_loads"] != episode+1 or final["restores"] or final["experiment_frames"]):
        raise RuntimeError("Ordinary-frame/cold-load accounting identity failed; preserve the incomplete capture")
    sheet = Image.new("RGB", (704, 720), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (index, pixels) in enumerate(previews):
        x, y = i % 4*176, i//4*240
        draw.text((x+8, y+5), f"frame {index}", fill="white")
        sheet.paste(Image.fromarray(pixels), (x+8, y+24))
    sheet.save(destination/"fixed_frames.png")
    # The sealed reader's historical format name contains "development".
    # Partition is separate metadata; changing the format name breaks the reader.
    manifest = dict(kind="scorer_only_cold_raw_development_fixture", partition="reserved", game="seaquest", mode=0,
            reset_convention=CONVENTION, seed=seed, frames=frames, action_repeat=protocol["action_repeat"],
            action_schedule="instrumentation_rgb_controller_v4", raw_every_real_frame=True,
            episode_limit=protocol["episode_limit"], sticky_probability=.25,
            sticky_implementation="worker-python-random-per-frame",
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), elapsed_seconds=time.perf_counter()-started,
            accounting=engine.accounting(), source_sha256=hashes, shards=shards,
            versions={p: importlib.metadata.version(p) for p in ("ale-py", "numpy", "scipy")},
            agent_training_allowed=False, reference_training_allowed=False, agent_entry_states_allowed=False, gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def score_reserved(directory, *, coverage, source_hashes):
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(game="seaquest",seed=17362 if coverage else 17312,
                  frames=32768 if coverage else 8192,action_repeat=4 if coverage else 16,
                  mode=0,episode_limit=108000,raw_every_real_frame=True,
                  sticky_probability=.25,sticky_implementation="worker-python-random-per-frame",
                  reset_convention=CONVENTION,agent_training_allowed=False,gate_result=None)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Reserved fixture parameters differ from the fixed pair")
    if coverage:
        if (manifest.get("action_schedule")!="instrumentation_rgb_controller_v4" or
                manifest.get("reference_training_allowed") is not False or
                manifest.get("agent_entry_states_allowed") is not False or
                manifest.get("source_sha256")!=source_hashes):
            raise ValueError("Reserved coverage provenance differs")
    else:
        raw_capture_sources=("scripts/capture_cold_v70.py","ragnarok/environments/emu_pool.py",
                             "ragnarok/environments/frame_budget_v70.py","ragnarok/environments/cold_emu_pool_v70.py")
        if (manifest.get("action_seed")!=117312 or
                manifest.get("source_sha256")!={p:source_hashes[p] for p in raw_capture_sources}):
            raise ValueError("Reserved uniform capture provenance differs")
    kinds=(*KINDS,"surface_submarine") if coverage else KINDS
    stats={k:new_stats() for k in kinds}
    examples={k:[] for k in kinds}
    for frame,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(directory),1):
        truth,pixels=decode_seaquest_ram(ram),screen_seaquest_objects(rgb)
        for kind,s in stats.items():
            a=[o for o in truth if o.slot=="surface"] if kind=="surface_submarine" else [o for o in truth if o.kind==kind]
            b=[o for o in pixels if o.kind=="submarine" and o.box[1]<57] if kind=="surface_submarine" else [o for o in pixels if o.kind==kind]
            n,within=accumulate(s,a,b)
            if (len(a)!=len(b) or within!=n) and len(examples[kind])<8:
                examples[kind].append(dict(frame=frame,episode=episode,ram=[o.box for o in a],pixels=[o.box for o in b]))
        if frame%4096==0:
            print(f"reserved geometry {manifest['seed']} {frame}/{manifest['frames']}",flush=True)
    for s in stats.values():
        finish(s)
        if not coverage:
            s["passed"]=(s["joint_matches"]>=50 and s["presence_fraction"]>=.99 and s["box_fraction"]>=.99)
    if source_hashes!=hashes():
        raise RuntimeError("Sealed sources changed during reserved scoring")
    return dict(kind="seaquest_reserved_coverage_geometry" if coverage else "seaquest_reserved_raw_geometry",
                seed=manifest["seed"],decoder_revision=REVISION,per_type=stats,
                calibration_pass=all(s["passed"] for s in stats.values()),
                geometry_agreement=geometry_pass(dict(per_type=stats),additional=coverage),
                first_disagreements=examples,source_sha256=source_hashes,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                agent_training_allowed=False,reference_training_allowed=False,gate_result=None)


def run_reserved(base):
    base=Path(base).resolve()
    destinations=[base/"reserved_raw",base/"reserved_coverage"]
    outputs=[base/"reserved_raw_result.json",base/"reserved_coverage_result.json",base/"reserved_pair_result.json"]
    if any(p.exists() for p in (*destinations,*outputs)):
        raise ValueError("Preserve the previous reserved attempt, including incomplete captures")
    seal=verify_seal(base)  # Must happen before constructing either emulator.
    source_hashes=seal["source_sha256"]
    capture_raw(destinations[0],game="seaquest",seed=17312,frames=8192)
    verify_seal(base)
    _capture_coverage(destinations[1],source_hashes)
    verify_seal(base)
    raw=score_reserved(destinations[0],coverage=False,source_hashes=source_hashes)
    _write(outputs[0],raw)
    additional=score_reserved(destinations[1],coverage=True,source_hashes=source_hashes)
    _write(outputs[1],additional)
    verify_seal(base)
    result=dict(kind="seaquest_reserved_pair_calibration",calibration_pass=raw["geometry_agreement"] and additional["calibration_pass"],
                raw_collection_calibration_pass=raw["calibration_pass"],raw_collection_geometry_agreement=raw["geometry_agreement"],
                coverage_collection_calibration_pass=additional["calibration_pass"],source_sha256=source_hashes,
                results_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:2]},
                seal_sha256=hashlib.sha256((base/"reserved_pair_seal.json").read_bytes()).hexdigest(),
                scope="Fixed reserved pair, separate fractions, absent raw categories remain uncalibrated; every category and surface subtype must pass the coverage collection. No tuning on reserved observations.",
                agent_training_allowed=False,reference_training_allowed=False,gate_result=None)
    _write(outputs[2],result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("seal","run"))
    parser.add_argument("--base",type=Path,default=Path("craft_v6_out/v70_seaquest_calibration"))
    args=parser.parse_args()
    result=create_seal(args.base.resolve()) if args.mode=="seal" else run_reserved(args.base)
    print(json.dumps(result,indent=2))
