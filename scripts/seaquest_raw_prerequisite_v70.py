"""Read-only prerequisite for work following the sealed raw Seaquest calibration."""
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.validate_seaquest_reserved_v70 import SOURCE_PATHS as RAW_SEALED_SOURCES,verify_seal

ROOT=Path(__file__).resolve().parents[1]


def require_raw_calibration():
    base=ROOT/"craft_v6_out/v70_seaquest_calibration"
    seal=verify_seal(base)
    path=base/"reserved_pair_result.json"
    pair=json.loads(path.read_text(encoding="utf-8"))
    if (pair.get("kind")!="seaquest_reserved_pair_calibration" or pair.get("calibration_pass") is not True or
            pair.get("source_sha256")!=seal["source_sha256"] or
            pair.get("seal_sha256")!=hashlib.sha256((base/"reserved_pair_seal.json").read_bytes()).hexdigest()):
        raise ValueError("A passed unchanged raw reserved pair is required")
    if set(pair["results_sha256"])!={"reserved_raw_result.json","reserved_coverage_result.json"}:
        raise ValueError("The complete raw result pair is required")
    for name,expected in pair["results_sha256"].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("Raw result proof was changed")
    for proof in (path,*(base/name for name in pair["results_sha256"])):
        relative=proof.relative_to(ROOT).as_posix()
        current=subprocess.check_output(["git","-C",str(ROOT),"hash-object",f"--path={relative}",str(proof)],text=True).strip()
        committed=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",f"HEAD:{relative}"],text=True).strip()
        if current!=committed:
            raise ValueError("Commit the passed raw calibration proof before the next capture")
    return hashlib.sha256(path.read_bytes()).hexdigest()
