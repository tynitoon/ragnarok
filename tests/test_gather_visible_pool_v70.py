"""Isolated decoder substitution and unchanged paid path on synthetic objects."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation import gather_visible_pool_v70 as gather
from ragnarok.evaluation import visible_reference_pool_v70 as original
from tests.test_cached_visible_pool_v70 import install_fixture
from tests.test_restorable_ram_history_v70 import same_history


def fixture(monkeypatch,terminal=999):
    install_fixture(monkeypatch,terminal)
    monkeypatch.setattr(gather,'mspacman_visible_scene',original.mspacman_visible_scene)


def test_ordinary_frame_grouping_and_branch_bodies_are_the_exact_original_methods():
    for name in ('_step','_on_frame','_record_reference'):
        assert getattr(gather.GatherVisibleEngine,name) is getattr(original.VisibleReferenceEngine,name)
    before=ast.parse(textwrap.dedent(inspect.getsource(original._serve_visible)))
    after=ast.parse(textwrap.dedent(inspect.getsource(gather._serve_gather)))
    before.body[0].name=after.body[0].name
    for node in ast.walk(before):
        if isinstance(node,ast.Name) and node.id=='VisibleReferenceEngine':node.id='GatherVisibleEngine'
    assert ast.dump(before)==ast.dump(after)


@pytest.mark.parametrize('terminal',[999,300,509])
def test_full_reports_outputs_and_restored_histories_match_original(monkeypatch,terminal):
    fixture(monkeypatch,terminal)
    baseline=original.VisibleReferenceEngine(EmulatorConfig(game='ms_pacman'))
    subject=gather.GatherVisibleEngine(EmulatorConfig(game='ms_pacman'))
    assert subject._decoder is gather.mspacman_visible_scene and not hasattr(subject,'_group_cache')
    for index in range(129):
        for left,right in zip(baseline.step(index%2),subject.step(index%2),strict=True):
            np.testing.assert_array_equal(left,right)
    assert subject.report()==baseline.report()
    same_history(subject.engine._ale.export_history(),baseline.engine._ale.export_history())


def test_gather_selected_before_first_source_and_paid_decoder_error_latches(monkeypatch):
    fixture(monkeypatch)
    calls=[]
    def failed(current,previous):calls.append(True);raise MemoryError('synthetic gather decoder')
    monkeypatch.setattr(gather,'mspacman_visible_scene',failed)
    subject=gather.GatherVisibleEngine(EmulatorConfig(game='ms_pacman'))
    assert calls==[]
    before=subject.report()['accounting']['charged_frames']
    with pytest.raises(MemoryError):subject.step(1)
    assert calls==[True] and subject.report()['accounting']['charged_frames']==before+1
    with pytest.raises(ValueError,match='stopped'):subject.step(0)
    assert calls==[True]


def test_late_error_dict_allocation_cannot_allow_a_second_paid_command(monkeypatch):
    fixture(monkeypatch)
    subject=gather.GatherVisibleEngine(EmulatorConfig(game='ms_pacman'))
    def fail(*args,**kwargs):raise MemoryError('synthetic allocation loss')
    subject._decoder=fail
    monkeypatch.setattr(original,'dict',fail,raising=False)
    before=subject.engine.accounting()['charged_frames']
    with pytest.raises(MemoryError):subject.step(0)
    assert subject._failure is None and subject._gather_chain_stopped
    with pytest.raises(ValueError,match='stopped'):subject.step(1)
    monkeypatch.delattr(original,'dict')
    assert subject.report()['failure']['error_type']=='UnreportedStepFailure'
    assert subject.report()['accounting']['charged_frames']==before+1


def serve_synthetic(connection,config,branch_every,reference_backend,native_build):
    with pytest.MonkeyPatch.context() as patch:
        fixture(patch)
        cls=gather.GatherVisibleEngine
        def construct(*args,**kwargs):
            engine=cls(*args,**kwargs)
            if config.seed==1:
                def fail(*args):raise ValueError('synthetic paid decoder failure')
                engine._decoder=fail
            return engine
        patch.setattr(gather,'GatherVisibleEngine',construct)
        gather._serve_gather(connection,config,branch_every,reference_backend,native_build)


def test_spawned_failure_drains_both_ledgers_and_stops_every_worker(monkeypatch):
    monkeypatch.setattr(gather,'_serve_gather',serve_synthetic)
    pool=gather.GatherVisiblePool([EmulatorConfig(game='ms_pacman',seed=i) for i in range(2)])
    processes=list(pool._processes)
    try:
        for forbidden in ('getRAM','clone','restore','reward','lives','branch_pair'):
            with pytest.raises(ValueError,match='Unknown'):pool._one(0,forbidden)
        with pytest.raises(ValueError,match='paid decoder failure'):pool.step([0,0])
        reports=[pool.report(i) for i in range(2)]
        assert reports[0]['ordinary_raw_frames']==4 and reports[1]['ordinary_raw_frames']==0
        assert reports[0]['accounting']['charged_frames']-reports[1]['accounting']['charged_frames']==3
        with pytest.raises(ValueError,match='trajectory stopped'):pool.step([1,1])
        assert reports==[pool.report(i) for i in range(2)]
    finally:pool.close()
    assert not any(p.is_alive() for p in processes)


def test_other_games_are_refused_before_native_construction(monkeypatch):
    monkeypatch.setattr(original,'_ScorerHistoryColdEngine',lambda *_:pytest.fail('native constructor reached'))
    for game in ('asteroids','boxing','frostbite','riverraid','kangaroo'):
        with pytest.raises(ValueError,match='MsPacman'):gather.GatherVisibleEngine(EmulatorConfig(game=game))
        with pytest.raises(ValueError,match='MsPacman'):gather.GatherVisiblePool([EmulatorConfig(game=game)])
