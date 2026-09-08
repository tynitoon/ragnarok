"""CPU-only checks of the external v62 launcher's scientific and process stop boundaries."""
import json
import math
import subprocess
import sys

import pytest

from scripts import resume_v62 as launch


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "craft_v6_out").mkdir()
    return tmp_path


def pretraining(out, s):
    worlds = []
    for k in range(1, 5):
        (out / f"v61_pre_s{s}_M{k}.pt").write_bytes(b"nonempty-checkpoint")
        worlds.append(dict(world=8200 + 10 * s + k - 1, k=k, goals=[1, 2, 3, 4], tiers=[2, 3, 4, 4],
                           rows=[dict(goal=g, tier=t, curve=[0, .25, .5, 1], rows=[1, 2, 3], demos=[0, 1, 1])
                                 for g, t in zip([1, 2, 3, 4], [2, 3, 4, 4])]))
    write_json(out / f"v61_pre_s{s}.json", dict(lineage=s, b_max=3, worlds=worlds, exercised=[[1, 2]]))


def probe(out, passed):
    # Both samples have 768 proposals. Equal rates STOP; a 0.5 gap passes the frozen threshold.
    rm, rr = (.75, .25) if passed else (.5, .5)
    arms = {name: dict(lawful=rm if name != "R" else rr, n=768) for name in ("M1", "M2", "M3", "M4", "R")}
    data = dict(world=8150, lineage=0, arms=arms,
                summary=dict(r_M=rm, r_R=rr, se=math.sqrt(.5 / 768), passed=passed))
    write_json(out / "v61_probe.json", data)
    return data


def state(root):
    return launch.read_json(root / "craft_v6_out/v62_resume_state.json")


def provenance():
    return dict(commands=launch.commands(), remaining_seconds=600)


@pytest.mark.parametrize("passed,expected", [(True, "probe_passed_review_required"), (False, "probe_stop")])
def test_pipeline_stops_after_probe_for_both_scientific_outcomes(root, monkeypatch, passed, expected):
    called = []

    def simulate(argv, workdir, remaining):
        called.append(argv)
        out = workdir / "craft_v6_out"
        if argv[3] == "scripts.pretrain_v61":
            pretraining(out, int(argv[5]))
        else:
            probe(out, passed)

    monkeypatch.setattr(launch, "run_child", simulate)
    assert launch.run_pipeline(root, provenance()) == 0
    assert state(root)["status"] == expected
    assert [argv[3] for argv in called] == ["scripts.pretrain_v61"] * 3 + ["scripts.probe_v61"]
    assert len(state(root)["completed"]) == 4
    assert not (root / "craft_v6_out/v62_resume_state.json.tmp").exists()


@pytest.mark.parametrize("defect", ["missing_checkpoint", "short_curve", "missing_world"])
def test_incomplete_pretraining_blocks_all_later_stages(root, monkeypatch, defect):
    called = []

    def simulate(argv, workdir, remaining):
        called.append(argv)
        out = workdir / "craft_v6_out"
        pretraining(out, 0)
        path = out / "v61_pre_s0.json"
        data = launch.read_json(path)
        if defect == "missing_checkpoint":
            (out / "v61_pre_s0_M4.pt").write_bytes(b"")
        elif defect == "short_curve":
            data["worlds"][1]["rows"][2]["curve"] = [0, .5]
        else:
            data["worlds"].pop()
        write_json(path, data)

    monkeypatch.setattr(launch, "run_child", simulate)
    assert launch.run_pipeline(root, provenance()) == 1
    assert len(called) == 1
    assert state(root)["status"] == "failed"
    assert state(root)["completed"] == []
    assert (root / "craft_v6_out/v61_pre_s0.json").exists()


@pytest.mark.parametrize("defect", ["wrong_pass", "nan_se", "zero_n", "missing_arm"])
def test_malformed_probe_cannot_report_success(root, monkeypatch, defect):
    def simulate(argv, workdir, remaining):
        out = workdir / "craft_v6_out"
        if argv[3] == "scripts.pretrain_v61":
            pretraining(out, int(argv[5]))
            return
        data = probe(out, False)
        if defect == "wrong_pass":
            data["summary"]["passed"] = True
        elif defect == "nan_se":
            data["summary"]["se"] = float("nan")
        elif defect == "zero_n":
            data["arms"]["M4"]["n"] = 0
        else:
            data["arms"].pop("M2")
        write_json(out / "v61_probe.json", data)

    monkeypatch.setattr(launch, "run_child", simulate)
    assert launch.run_pipeline(root, provenance()) == 1
    assert state(root)["status"] == "failed"
    assert len(state(root)["completed"]) == 3


def test_budget_exhaustion_does_not_launch_next_command(root, monkeypatch):
    now, called = [0], []
    monkeypatch.setattr(launch.time, "monotonic", lambda: now[0])

    def simulate(argv, workdir, remaining):
        called.append(argv)
        pretraining(workdir / "craft_v6_out", 0)
        now[0] = 601

    monkeypatch.setattr(launch, "run_child", simulate)
    assert launch.run_pipeline(root, provenance()) == 1
    assert len(called) == 1
    assert state(root)["status"] == "budget_exhausted"


def test_child_error_stops_pipeline_and_preserves_state(root):
    info = provenance()
    info["commands"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(7)"]
    assert launch.run_pipeline(root, info) == 1
    result = state(root)
    assert result["status"] == "failed" and result["completed"] == []
    assert "CalledProcessError" in result["error"]


def test_timeout_reaps_only_the_child_it_launched(root, monkeypatch):
    children = []
    actual_popen = subprocess.Popen

    def record(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(launch.subprocess, "Popen", record)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.run_child([sys.executable, "-c", "import time; time.sleep(30)"], root, .2)
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("name", ["v62_resume_state.json", "v61_pre_s0.json", "v61_probe.json"])
def test_existing_production_artifacts_refuse_launch_before_git_or_gpu(root, monkeypatch, name):
    artifact = root / "craft_v6_out" / name
    artifact.write_text("preserve this", encoding="utf-8")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("No subprocess should start"))
    with pytest.raises(ValueError, match="already exist"):
        launch.preflight(root)
    assert artifact.read_text(encoding="utf-8") == "preserve this"


def test_source_change_between_stages_blocks_next_child(root, monkeypatch):
    frozen = root / "frozen.py"
    frozen.write_text("original", encoding="utf-8")
    info = provenance()
    info["files"] = {"frozen.py": {"working_sha256": launch.digest(frozen)}}
    called = []

    def simulate(argv, workdir, remaining):
        called.append(argv)
        pretraining(workdir / "craft_v6_out", 0)
        frozen.write_text("changed", encoding="utf-8")

    monkeypatch.setattr(launch, "run_child", simulate)
    assert launch.run_pipeline(root, info) == 1
    assert len(called) == 1
    assert "Frozen source changed" in state(root)["error"]
