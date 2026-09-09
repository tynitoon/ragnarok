import json
import numpy as np
import pytest
from scripts import validate_seaquest_pooled_v70 as capture
from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames


def test_reject_missing_seal_before_attempt_or_constructor(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,"verify_seal",lambda base:(_ for _ in ()).throw(ValueError("missing seal")))
    monkeypatch.setattr(capture,"_ColdPixelEngine",lambda *a,**k:pytest.fail("emulator opened"))
    with pytest.raises(ValueError,match="missing seal"):
        capture.run_reserved(tmp_path)
    assert not (tmp_path/"reserved_attempt.json").exists()


@pytest.mark.parametrize("prior",["reserved","reserved_result.json","reserved_attempt.json"])
def test_preserve_any_previous_attempt(tmp_path,monkeypatch,prior):
    (tmp_path/prior).touch()
    monkeypatch.setattr(capture,"verify_seal",lambda *a:pytest.fail("verification should not begin"))
    with pytest.raises(ValueError,match="Preserve"):
        capture.run_reserved(tmp_path)


def test_constructor_failure_is_durable_and_cannot_retry(tmp_path,monkeypatch):
    (tmp_path/"reserved_seal.json").write_text("{}",encoding="utf-8")
    monkeypatch.setattr(capture,"verify_seal",lambda base:{})
    def failed(destination):
        assert (tmp_path/"reserved_attempt.json").exists()
        raise RuntimeError("constructor failed")
    monkeypatch.setattr(capture,"_capture_reserved",failed)
    with pytest.raises(RuntimeError,match="constructor failed"):
        capture.run_reserved(tmp_path)
    with pytest.raises(ValueError,match="Preserve"):
        capture.run_reserved(tmp_path)


def test_actual_writer_reader_contract_across_cold_restart_and_short_repeat(tmp_path,monkeypatch):
    protocol=dict(development_seed=17321,decisions=2,episode_limit=108000)
    (tmp_path/"docs").mkdir()
    (tmp_path/capture.DEVELOPMENT_PROTOCOL).write_text(json.dumps(protocol),encoding="utf-8")
    monkeypatch.setattr(capture,"ROOT",tmp_path)
    monkeypatch.setattr(capture,"SOURCE_PATHS",(capture.DEVELOPMENT_PROTOCOL,))
    monkeypatch.setattr(capture,"require_raw_calibration",lambda:"proof")

    class Native:
        def __init__(self,offset):
            self.offset,self.frames=offset,0
        def getRAM(self):
            return np.full(128,self.offset+self.frames,dtype=np.uint8)
        def act(self,action):
            self.frames+=1
        def getEpisodeFrameNumber(self):
            return self.frames

    class Engine:
        def __init__(self,config):
            self._ale=Native(0)
            self.calls,self.loads,self.real=0,1,0
            self._frames=[np.zeros((210,160,3),dtype=np.uint8)]*2
        def set_native_wrapper(self,factory):
            self.factory=factory
            self._ale=factory(self._ale)
        def observe(self,raw):
            return self._frames[-1].copy()
        def step(self,action):
            self.calls+=1
            if self.calls==2:
                self.loads=2
                self._ale=self.factory(Native(100))
                self._frames=[np.full((210,160,3),100,dtype=np.uint8)]*2
            for _ in range(4 if self.calls==1 else 1):
                self._ale.act(action)
                self.real+=1
                self._frames=[self._frames[-1],np.full((210,160,3),int(self._ale.getRAM()[0]),dtype=np.uint8)]
            return np.maximum(*self._frames)
        def accounting(self):
            return dict(reset_emulated_frames=68*self.loads,charged_frames=128*self.loads+self.real,
                        cold_loads=self.loads,experiment_frames=0,restores=0)

    monkeypatch.setattr(capture,"_ColdPixelEngine",Engine)
    destination=tmp_path/"fixture"
    report=capture._capture_reserved(destination)
    rows=list(cold_pooled_fixture_frames(destination))
    assert report["ordinary_real_frames"]==5 and report["accounting"]["charged_frames"]==261
    assert [int(r["ram_previous"][0]) for r in rows]==[3,100]
    assert [int(r["ram_current"][0]) for r in rows]==[4,101]
    assert [int(r["raw_previous"][0,0,0]) for r in rows]==[3,100]
    assert [int(r["raw_current"][0,0,0]) for r in rows]==[4,101]
