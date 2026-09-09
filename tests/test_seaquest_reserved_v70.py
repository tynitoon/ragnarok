import pytest

from scripts import validate_seaquest_reserved_v70 as reserved


def test_no_emulator_before_valid_committed_seal(tmp_path,monkeypatch):
    calls=[]
    def rejected(_):
        raise ValueError("uncommitted seal")
    monkeypatch.setattr(reserved,"verify_seal",rejected)
    monkeypatch.setattr(reserved,"capture_raw",lambda *a,**k:calls.append("raw"))
    monkeypatch.setattr(reserved,"_capture_coverage",lambda *a,**k:calls.append("coverage"))
    with pytest.raises(ValueError,match="uncommitted"):
        reserved.run_reserved(tmp_path)
    assert calls==[] and list(tmp_path.iterdir())==[]


def test_incomplete_reserved_capture_is_never_overwritten(tmp_path,monkeypatch):
    (tmp_path/"reserved_raw").mkdir()
    def forbidden(_):
        pytest.fail("Existing reserved attempt must be rejected before further work")
    monkeypatch.setattr(reserved,"verify_seal",forbidden)
    with pytest.raises(ValueError,match="previous reserved attempt"):
        reserved.run_reserved(tmp_path)


def test_fixed_pair_order_and_failure_are_preserved(tmp_path,monkeypatch):
    calls=[]
    seal={"source_sha256":{}}
    (tmp_path/"reserved_pair_seal.json").write_text("{}",encoding="utf-8")
    monkeypatch.setattr(reserved,"verify_seal",lambda base:seal)
    monkeypatch.setattr(reserved,"capture_raw",lambda dest,**kw:calls.append(("raw",kw)))
    monkeypatch.setattr(reserved,"_capture_coverage",lambda dest,hashes:calls.append(("coverage",dest.name)))
    monkeypatch.setattr(reserved,"score_reserved",lambda dest,coverage,source_hashes:
                        dict(calibration_pass=False,geometry_agreement=not coverage))
    result=reserved.run_reserved(tmp_path)
    assert calls==[("raw",dict(game="seaquest",seed=17312,frames=8192)),("coverage","reserved_coverage")]
    assert result["calibration_pass"] is False
    assert (tmp_path/"reserved_pair_result.json").exists()


def test_positive_geometry_errors_cannot_hide_in_empty_frames():
    s=dict(presence_fraction=.995,ram_objects=100,pixel_objects=50,joint_matches=50,
           box_fraction=1.0,positive_presence_fraction=.5)
    assert not reserved.geometry_pass(dict(per_type={"actor":s}),additional=True)


def test_reserved_writer_is_readable_by_sealed_cold_reader(tmp_path,monkeypatch):
    import hashlib
    import json
    import numpy as np
    from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

    protocol_path="docs/ARC4_SEAQUEST_COVERAGE_V4_PROTOCOL.json"
    (tmp_path/"docs").mkdir()
    (tmp_path/protocol_path).write_text(json.dumps(dict(frames=2,episode_limit=108000,
        max_charged_frames=1000,action_repeat=4)),encoding="utf-8")
    source_hashes={protocol_path:hashlib.sha256((tmp_path/protocol_path).read_bytes()).hexdigest()}

    class FakeEngine:
        def __init__(self,config):
            assert config.seed==17362
            self.steps=0
            self._ale=self
        def getRAM(self):
            return np.zeros(128,dtype=np.uint8)
        def observe(self,raw):
            return np.zeros((210,160,3),dtype=np.uint8)
        def step(self,action):
            self.steps+=1
        def accounting(self):
            return dict(reset_emulated_frames=68,emulated_frames=128+self.steps,
                        charged_frames=128+self.steps,cold_loads=1,restores=0,
                        experiment_frames=0,format_detection_emulated_frames=60)

    monkeypatch.setattr(reserved,"ROOT",tmp_path)
    monkeypatch.setattr(reserved,"SOURCE_PATHS",(protocol_path,))
    monkeypatch.setattr(reserved,"_ColdPixelEngine",FakeEngine)
    destination=tmp_path/"capture"
    manifest=reserved._capture_coverage(destination,source_hashes)
    assert manifest["partition"]=="reserved"
    assert len(list(cold_fixture_frames(destination)))==2


@pytest.mark.parametrize("bad_seed",[17331,17341,17351])
def test_each_older_collection_must_pass_before_sealing(tmp_path,monkeypatch,bad_seed):
    import json
    monkeypatch.setattr(reserved,"ROOT",tmp_path)
    monkeypatch.setattr(reserved,"RAW_SOURCES",())
    monkeypatch.setattr(reserved,"COVERAGE_SOURCES",())
    base=tmp_path/"reports"
    base.mkdir()
    for seed,name in reserved.DEVELOPMENT.items():
        kinds=reserved.KINDS if seed==17311 else (*reserved.KINDS,"surface_submarine")
        stats={k:dict(presence_fraction=1.0,positive_presence_fraction=1.0,
                      ram_objects=100,pixel_objects=100,joint_matches=100,box_fraction=1.0,passed=True)
               for k in kinds}
        if seed!=17311 and seed!=17361:
            stats["surface_submarine"].update(ram_objects=0,pixel_objects=0,joint_matches=0,
                                               box_fraction=None,positive_presence_fraction=None,passed=False)
        if seed==bad_seed:
            stats["player"]["positive_presence_fraction"]=.98
        report=dict(seed=seed,source_sha256={},per_type=stats,
                    kind="independent_seaquest_raw_calibration" if seed==17311 else "seaquest_development_coverage_geometry")
        (base/name).write_text(json.dumps(report),encoding="utf-8")
    with pytest.raises(ValueError,match="geometry remains negative"):
        reserved.development_evidence(base,{})
