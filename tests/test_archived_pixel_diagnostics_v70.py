"""Numerical archive-intake mechanics on synthetic RGB, never native archives."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.learning.archived_pixel_diagnostics_v70 import ArchivedPixelDiagnostics,weight_sha256
from ragnarok.learning.parser_v70 import CounterfactualParser
from tests.test_owned_rgb_collector_v70 import SyntheticPool
from tests.test_owned_pixel_objectives_v70 import ScriptedParser as BaseScriptedParser,reset_engine

torch.set_num_threads(1)


class ScriptedParser(BaseScriptedParser):
    def forward(self,rgb,effect=None):
        if effect is None:effect=torch.zeros((len(rgb),12),dtype=torch.float32,device=rgb.device)
        return super().forward(rgb,effect)


def collector(terminal=999):
    pool=SyntheticPool(1)
    reset_engine(pool.engines[0],terminal)
    return OwnedRGBCollector(pool,1,session_id='synthetic-archive'),pool


def record(own,action=1):return own.take(own.step([action]))[0]


def test_real_parser_graphs_have_finite_admitted_gradients_without_weight_updates():
    own,_=collector()
    torch.manual_seed(764);parser=CounterfactualParser()
    before=weight_sha256(parser)
    diagnostic=ArchivedPixelDiagnostics(parser,stream_id='synthetic')
    rows=[diagnostic.step(i,record(own)) for i in range(3)]
    assert all(row['prediction_eligible'] and row['weights_unchanged'] for row in rows)
    assert all(np.isfinite(row['diagnostic_sum']) and row['gradient']['global_norm_after']<=1.000001 for row in rows)
    assert all(parameter.grad is None for parameter in parser.parameters())
    assert weight_sha256(parser)==before and diagnostic._end==12


def test_raw_button_metadata_changes_neither_features_nor_losses():
    own,_=collector();saved=record(own)
    first=ScriptedParser();second=copy.deepcopy(first)
    a=ArchivedPixelDiagnostics(first,stream_id='same').step(0,saved)
    b=ArchivedPixelDiagnostics(second,stream_id='same').step(0,replace(saved,requested_action=17))
    assert a==b and len(first.seen)==len(second.seen)
    for left,right in zip(first.seen,second.seen,strict=True):
        for x,y in zip(left,right,strict=True):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert torch.count_nonzero(left[1])==0


def test_collection_gap_discards_identity_history_without_inventing_frames():
    own,_=collector();diagnostic=ArchivedPixelDiagnostics(ScriptedParser(),stream_id='same')
    diagnostic.step(0,record(own));diagnostic.step(1,record(own))
    old_stream=diagnostic._stream
    skipped=record(own)
    row=diagnostic.step(3,record(own))
    assert row['history_restarted'] and row['identity_anchors']==0
    assert diagnostic._stream is not old_stream and set(diagnostic._stream._history)=={12,16}
    assert skipped.chronology.end_frame==12  # Its endpoint was only reused as the next actual source.


def test_inconsistent_contiguous_clock_is_rejected_and_diagnostic_stops():
    own,_=collector();diagnostic=ArchivedPixelDiagnostics(ScriptedParser(),stream_id='same')
    diagnostic.step(0,record(own));second=record(own)
    bad=replace(second,chronology=replace(second.chronology,start_frame=5,end_frame=9))
    with pytest.raises(ValueError,match='Contiguous'):diagnostic.step(1,bad)
    with pytest.raises(ValueError,match='stopped'):diagnostic.step(1,second)


def test_truncated_p_stays_unknown_even_when_other_objectives_are_finite():
    own,_=collector(terminal=3)
    row=ArchivedPixelDiagnostics(ScriptedParser(),stream_id='short').step(0,record(own))
    assert row['actual_frames']==3 and not row['prediction_eligible']
    assert row['losses']['prediction']==row['losses']['foreground']==0
    assert row['delivered_end_frame']==3 and row['weights_unchanged']


def test_empty_saved_operation_adds_no_temporal_target_or_delivered_time():
    own,_=collector();saved=record(own)
    empty=np.empty((0,*saved.trace.endpoint.shape),np.uint8)
    saved=replace(saved,trace=replace(saved.trace,raw=empty,pooled=empty.copy(),
        endpoint=np.maximum(*saved.trace.start_pair)),chronology=replace(saved.chronology,end_frame=0))
    diagnostic=ArchivedPixelDiagnostics(ScriptedParser(),stream_id='empty')
    first=diagnostic.step(0,saved);second=diagnostic.step(1,saved)
    assert first['history_restarted'] and not second['history_restarted']
    assert second['observed_instances']==second['identity_anchors']==second['delivered_end_frame']==0
    assert not second['prediction_eligible'] and second['gradient']['global_norm_before']==0
    assert set(diagnostic._stream._history)=={0}


def test_owned_eight_frame_pair_uses_c_four_without_aging_ordinary_memory():
    own,_=collector();parser=ScriptedParser()
    diagnostic=ArchivedPixelDiagnostics(parser,stream_id='same')
    diagnostic.step(0,record(own));anchor=own.current_anchor()
    operation=own.branch(1,expected_anchor=anchor.token,frames=8)
    branch=own.take(operation)[0]
    before=diagnostic._index,diagnostic._end,diagnostic._epoch,diagnostic._stream
    row=diagnostic.branch(branch.operation_id,start_pair=branch.anchor.pair,effect=branch.effect_assumption,
        intervention=branch.intervention,control=branch.control,requested_frames=8)
    assert row['common_prefix_frames']==8 and row['weights_unchanged']
    assert (diagnostic._index,diagnostic._end,diagnostic._epoch,diagnostic._stream)==before
    with pytest.raises(ValueError,match='unique'):
        diagnostic.branch(branch.operation_id,start_pair=branch.anchor.pair,effect=branch.effect_assumption,
            intervention=branch.intervention,control=branch.control,requested_frames=8)


def test_nonfinite_head_fails_before_any_gradient_is_admitted():
    own,_=collector();parser=ScriptedParser()
    diagnostic=ArchivedPixelDiagnostics(parser,stream_id='same')
    def bad(_module,_args,output):
        output.solidity_logits[0,0,0,0]=float('nan')
        return output
    hook=parser.register_forward_hook(bad)
    try:
        with pytest.raises(ValueError):diagnostic.step(0,record(own))
    finally:hook.remove()
    assert diagnostic._stopped and all(p.grad is None for p in parser.parameters())


def test_reentrant_processing_is_refused_without_disturbing_current_lock():
    diagnostic=ArchivedPixelDiagnostics(ScriptedParser(),stream_id='same')
    own,_=collector();saved=record(own)
    diagnostic._lock.acquire()
    try:
        with pytest.raises(ValueError,match='active'):diagnostic.step(0,saved)
        assert not diagnostic._stopped
    finally:diagnostic._lock.release()
    assert diagnostic.step(0,saved)['weights_unchanged']
