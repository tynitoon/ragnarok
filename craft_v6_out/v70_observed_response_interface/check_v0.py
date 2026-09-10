"""Run only the declared synthetic interface fixtures, with a hard wall cap."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATTEMPT = OUT / "check_v0_attempt.json"
RESULT = OUT / "check_v0.json"
LOG = OUT / "check_v0_output.txt"
SOURCES = [
    "docs/ARC4_OBSERVED_RESPONSE_TARGETS_DRAFT.md",
    "ragnarok/learning/observed_response_targets_v70.py",
    "tests/test_observed_response_targets_v70.py",
    "tests/test_counterfactual_occlusion_limits_v70.py",
    "craft_v6_out/v70_observed_response_interface/check_v0.py",
]
command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "tests/test_observed_response_targets_v70.py"]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True)
sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
intent = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_head=head,
              source_sha256=sources, command=command, wall_limit_seconds=60,
              scope="new synthetic interface fixtures only; no model, emulator or experimental data")
with ATTEMPT.open("x", encoding="utf-8") as stream:
    json.dump(intent, stream, indent=2)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
started = time.perf_counter()
status, code, output = "failed", None, ""
try:
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                               text=True, timeout=60)
    code, output = completed.returncode, completed.stdout + completed.stderr
    status = "passed" if code == 0 else "failed"
except subprocess.TimeoutExpired as error:
    status = "budget_exhausted"
    output = (error.stdout or b"") + (error.stderr or b"")
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
elapsed = time.perf_counter() - started
if elapsed > 60:
    status = "budget_exhausted"
with LOG.open("x", encoding="utf-8") as stream:
    stream.write(output)
result = dict(intent, status=status, returncode=code, test_wall_seconds=elapsed,
              cumulative_test_wall_seconds=elapsed,
              remaining_test_wall_seconds=max(0, 60-elapsed),
              output_file=LOG.relative_to(ROOT).as_posix(),
              output_sha256=hashlib.sha256(LOG.read_bytes()).hexdigest())
with RESULT.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
print(output, end="")
print(json.dumps(dict(status=status, wall_seconds=elapsed, result=str(RESULT))))
sys.exit(0 if status == "passed" else 1)
