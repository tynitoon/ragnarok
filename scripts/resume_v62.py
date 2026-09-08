"""Run the frozen v62 pretraining and probe, then stop for scientific review.

No automatic recovery: interrupted/previous outputs require a separate audited restart.
Run with the verified Python environment; --dry-run performs read-only preflight checks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

FREEZE = "v62-stage0-freeze"
PRECHECK_SECONDS = 137.15  # Measured CPU/GPU instrument checks at the authorized resumption.
REMAINING_SECONDS = 42 * 3600 - 6.7 * 3600 - 16124 - PRECHECK_SECONDS
CRITICAL = [f"scripts/{name}.py" for name in (
    "gate_v61", "pretrain_v61", "probe_v61", "confirm_v61", "score_v61",
    "pair_net_v61", "pair_store_v61", "hand_v61", "sweeper_v61",
    "hidden_recipe_v55", "depth_scaling_v49", "childhood_v50", "meta_manager_v51",
)] + ["ragnarok/environments/law_world.py", "ragnarok/environments/tech_tree.py",
      "ragnarok/learning/ppo_discrete.py", "ragnarok/infrastructure/device.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, low=0, high=1):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def commands():
    steps = [{"stage": f"pretrain_s{s}", "lineage": s, "argv": [sys.executable, "-u", "-m",
              "scripts.pretrain_v61", "--lineage", str(s), "--b-max", "3", "--num-envs", "64",
              "--store-persists", "--resume"]} for s in range(3)]
    return steps + [{"stage": "probe", "argv": [sys.executable, "-u", "-m", "scripts.probe_v61",
                     "--lineage", "0", "--num-envs", "64"]}]


def preflight(root):
    out = root / "craft_v6_out"
    require(not (out / "v62_resume_state.json").exists(), "An execution state already exists; audit before restarting.")
    require(not (out / "v62_resume_state.json.tmp").exists(), "An unfinished state write exists; audit before restarting.")
    existing = [p.name for pattern in ("v61_pre_s*", "v61_probe*") for p in out.glob(pattern)
                if "_smoke" not in p.name]
    require(not existing, f"Production outputs already exist; audit before restarting: {existing}")
    gate_path = out / "v61_gate_v62.json"
    gate = read_json(gate_path)
    summary = gate["summary"]
    require(gate["worlds"] == [8102, 8103] and gate["b_max"] == 6 and gate["num_envs"] == 64
            and gate["store_persists"] is True and summary["N"] == 9 and summary["B_test"] == 6,
            "Gate parameters do not match the v62 freeze.")
    require(all(summary[key] is True for key in ("reachable", "fresh_learns", "room", "proceed")),
            "The v62 gate does not authorize proceeding.")
    git = lambda *args: subprocess.run(["git", "--no-optional-locks", *args], cwd=root, check=True, capture_output=True)
    commit = git("rev-parse", f"{FREEZE}^{{commit}}").stdout.decode().strip()
    require(commit.startswith("a189ea5"), "The freeze tag no longer points to the audited commit a189ea5.")
    git("diff", "--exit-code", FREEZE, "--", *CRITICAL)
    hashes = {name: {"working_sha256": digest(root / name),
              "frozen_sha256": hashlib.sha256(git("show", f"{FREEZE}:{name}").stdout).hexdigest()}
              for name in CRITICAL}
    inputs = {name: digest(out / name) for name in ("v55_skill_s0.pt", "v61_roster.json")}
    return dict(freeze=FREEZE, freeze_commit=commit, python=sys.executable, commands=commands(),
                files=hashes, inputs=inputs, gate_sha256=digest(gate_path),
                launcher_sha256=digest(Path(__file__)),
                remaining_seconds=REMAINING_SECONDS, precheck_seconds=PRECHECK_SECONDS, root=str(root))


def verify_provenance(root, provenance):
    for name, hashes in provenance.get("files", {}).items():
        require(digest(root / name) == hashes["working_sha256"], f"Frozen source changed: {name}")
    for name, expected in provenance.get("inputs", {}).items():
        require(digest(root / "craft_v6_out" / name) == expected, f"Frozen input changed: {name}")
    if "gate_sha256" in provenance:
        require(digest(root / "craft_v6_out/v61_gate_v62.json") == provenance["gate_sha256"], "The gate file changed.")


def validate_pretraining(out, lineage):
    prefix = f"v61_pre_s{lineage}"
    data = read_json(out / f"{prefix}.json")
    worlds = data["worlds"]
    require(data["lineage"] == lineage and data["b_max"] == 3 and len(worlds) == 4,
            f"{prefix}: expected four worlds at B=3.")
    seeds = [world["world"] for world in worlds]
    require(seeds == sorted(set(seeds)) and all(8200 + 10 * lineage <= w < 8210 + 10 * lineage for w in seeds),
            f"{prefix}: invalid world sequence.")
    require(isinstance(data["exercised"], list), f"{prefix}: missing exercised cells.")
    artifacts = [out / f"{prefix}.json"]
    for k, world in enumerate(worlds, 1):
        checkpoint = out / f"{prefix}_M{k}.pt"
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"Missing or empty {checkpoint.name}.")
        artifacts.append(checkpoint)
        require(world["k"] == k and world["tiers"] == [2, 3, 4, 4] and len(world["rows"]) == 4
                and len(world["goals"]) == len(set(world["goals"])) == 4, f"{prefix}: incomplete goal stream.")
        for goal, tier, row in zip(world["goals"], world["tiers"], world["rows"]):
            require(row["goal"] == goal and row["tier"] == tier and len(row["curve"]) == 4
                    and all(finite(x) for x in row["curve"]), f"{prefix}: incomplete/invalid learning curve.")
            require(all(len(row[key]) == 3 and all(type(x) is int and x >= 0 for x in row[key])
                        for key in ("rows", "demos")), f"{prefix}: incomplete training rounds.")
    return {path.name: digest(path) for path in artifacts}


def validate_probe(out):
    data = read_json(out / "v61_probe.json")
    require(data["world"] == 8150 and data["lineage"] == 0 and set(data["arms"]) == {"M1", "M2", "M3", "M4", "R"},
            "Probe requires world 8150 and M1..M4/R from lineage 0.")
    for name, arm in data["arms"].items():
        require(type(arm["n"]) is int and 0 < arm["n"] <= 768 and finite(arm["lawful"]),
                f"Invalid proposal count or lawfulness for {name}.")
    m, r, summary = data["arms"]["M4"], data["arms"]["R"], data["summary"]
    pooled = (m["lawful"] * m["n"] + r["lawful"] * r["n"]) / (m["n"] + r["n"])
    se = math.sqrt(pooled * (1 - pooled) * (1 / m["n"] + 1 / r["n"]))
    for key, expected in (("r_M", m["lawful"]), ("r_R", r["lawful"]), ("se", se)):
        require(finite(summary[key]) and math.isclose(summary[key], expected, rel_tol=1e-10, abs_tol=1e-12),
                f"Probe summary {key} does not match the frozen calculation.")
    passed = m["lawful"] - r["lawful"] >= 3 * se
    require(summary["passed"] is passed, "Probe verdict does not match the frozen calculation.")
    return passed


def save_state(path, state, create=False):
    # Exclusive temporary creation also rejects overlapping writers; replace publishes complete JSON.
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    require(not create or not path.exists(), "An execution state was created by another launcher.")
    os.replace(temporary, path)


def run_child(argv, root, remaining):
    if remaining <= 0:
        raise TimeoutError("ARC 3 compute budget exhausted before child launch.")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(argv, cwd=root, shell=False, creationflags=flags)
    try:
        print(f"Child PID {child.pid}: {argv}", flush=True)
        code = child.wait(timeout=remaining)
        if code:
            raise subprocess.CalledProcessError(code, argv)
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


def run_pipeline(root, provenance):
    out = root / "craft_v6_out"
    path = out / "v62_resume_state.json"
    state = dict(status="starting", controller_pid=os.getpid(),
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 provenance=provenance, completed=[], elapsed_seconds=0.0)
    save_state(path, state, create=True)
    start = time.monotonic()
    result = 1
    try:
        for step in provenance["commands"]:
            verify_provenance(root, provenance)
            elapsed = time.monotonic() - start
            remaining = provenance["remaining_seconds"] - elapsed
            if remaining <= 0:
                raise TimeoutError("ARC 3 compute budget exhausted before next stage.")
            state.update(status="running", stage=step["stage"], elapsed_seconds=elapsed)
            save_state(path, state)
            print(f"Starting {step['stage']} ({remaining / 3600:.2f} h remaining)", flush=True)
            run_child(step["argv"], root, remaining)
            if step["stage"] == "probe":
                passed = validate_probe(out)
                artifacts = {"v61_probe.json": digest(out / "v61_probe.json")}
                state["status"] = "probe_passed_review_required" if passed else "probe_stop"
            else:
                artifacts = validate_pretraining(out, step["lineage"])
            state["completed"].append(dict(stage=step["stage"], artifacts=artifacts))
            state["elapsed_seconds"] = time.monotonic() - start
            save_state(path, state)
            if step["stage"] == "probe":
                result = 0
                break  # No confirmatory command is ever dispatched, including after a passed probe.
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        state.update(status="budget_exhausted", error=str(exc))
    except KeyboardInterrupt:
        state.update(status="interrupted", error="Interrupted; child stopped, outputs preserved.")
        result = 130
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        state["elapsed_seconds"] = time.monotonic() - start
        save_state(path, state)
        print(f"Stopped: {state['status']}; state: {path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        provenance = preflight(root)
        if args.dry_run:
            print(json.dumps(provenance, indent=2))
            return 0
        return run_pipeline(root, provenance)
    except Exception as exc:
        print(f"Launch refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
