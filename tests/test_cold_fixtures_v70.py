"""Reject legacy undercharges and missing cold transitions before calibration."""
import hashlib
import json

import numpy as np
import pytest

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames


def write_fixture(path, *, charges=(133,134), episodes=(0,0), final_charge=None):
    shard=path/"part.npz"
    np.savez_compressed(shard,rgb=np.zeros((2,210,160,3),dtype=np.uint8),ram=np.zeros((2,128),dtype=np.uint8),
                        episode=np.array(episodes,dtype=np.int32),charged_frame=np.array(charges,dtype=np.int64),
                        action=np.zeros(2,dtype=np.uint8))
    manifest=dict(kind="scorer_only_cold_raw_development_fixture",agent_training_allowed=False,
                  reset_convention="fixed_hardware_cold_start_v1",game="breakout",raw_every_real_frame=True,frames=2,
                  shards=[dict(path=shard.name,sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),frames=2)],
                  accounting=dict(charged_frames=charges[-1] if final_charge is None else final_charge,
                                  cold_loads=episodes[-1]+1,experiment_frames=0,restores=0,
                                  format_detection_emulated_frames=60*(episodes[-1]+1)))
    (path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")


def test_complete_load_and_reset_costs_are_validated(tmp_path):
    write_fixture(tmp_path)
    assert len(list(cold_fixture_frames(tmp_path)))==2
    write_fixture(tmp_path,charges=(133,266),episodes=(0,1))
    assert len(list(cold_fixture_frames(tmp_path)))==2
    write_fixture(tmp_path,charges=(73,74))
    with pytest.raises(ValueError,match="charge"):
        list(cold_fixture_frames(tmp_path))
    write_fixture(tmp_path,charges=(133,206),episodes=(0,1))
    with pytest.raises(ValueError,match="charge"):
        list(cold_fixture_frames(tmp_path))


def test_missing_episode_or_final_ledger_is_rejected(tmp_path):
    write_fixture(tmp_path,charges=(133,134),episodes=(0,2))
    with pytest.raises(ValueError,match="Noncontiguous"):
        list(cold_fixture_frames(tmp_path))
    write_fixture(tmp_path,final_charge=194)
    with pytest.raises(ValueError,match="ledger"):
        list(cold_fixture_frames(tmp_path))


def test_tampered_fixture_cannot_be_scored(tmp_path):
    write_fixture(tmp_path)
    with (tmp_path/"part.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="hash"):
        list(cold_fixture_frames(tmp_path))
