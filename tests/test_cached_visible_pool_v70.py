"""Exact frozen-chain delta and Windows transport on synthetic native objects."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation import cached_visible_pool_v70 as cached
from ragnarok.evaluation import visible_reference_pool_v70 as original
from tests.test_visible_reference_pool_v70 import factory
from tests.test_restorable_ram_history_v70 import same_history


def install_fixture(monkeypatch,terminal=999):
    factory(monkeypatch,terminal=terminal)
    monkeypatch.setattr(original,'mspacman_visible_scene',lambda now,previous:[
        SceneObject('player:0','player',(10+int(now[0])%100,20,8,10)),
        SceneObject('pill:0','pill',(10,60,4,2),False),
        SceneObject('pill:1','pill',(14,60,4,2),False)])


def test_step_ast_changes_only_grouping_call_and_transport_changes_only_engine_class():
    def tree(function):return ast.parse(textwrap.dedent(inspect.getsource(function)))
    before=tree(original.VisibleReferenceEngine._step)
    after=tree(cached.CachedVisibleEngine._step)
    changed=0
    for node in ast.walk(before):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='group_visible_sources':
            node.func=ast.parse('self._group_cache.group',mode='eval').body
            node.keywords=[];changed+=1
    assert changed==1 and ast.dump(before)==ast.dump(after)
    before=tree(original._serve_visible);after=tree(cached._serve_cached)
    before.body[0].name=after.body[0].name
    for node in ast.walk(before):
        if isinstance(node,ast.Name) and node.id=='VisibleReferenceEngine':node.id='CachedVisibleEngine'
    assert ast.dump(before)==ast.dump(after)


@pytest.mark.parametrize('terminal',[999,300,509])
def test_all_outputs_reports_branch_digests_and_cold_histories_match_original(monkeypatch,terminal):
    install_fixture(monkeypatch,terminal)
    config=EmulatorConfig(game='ms_pacman')
    baseline,subject=original.VisibleReferenceEngine(config),cached.CachedVisibleEngine(config)
    for i in range(129):
        left,right=baseline.step(i%2),subject.step(i%2)
        for a,b in zip(left,right,strict=True):np.testing.assert_array_equal(a,b)
    assert baseline.report()==subject.report()
    same_history(baseline.engine._ale.export_history(),subject.engine._ale.export_history())
    assert subject._decoder is original.mspacman_visible_scene
    assert subject._group_cache._cache['pill'][1][0].members==frozenset(('pill:0','pill:1'))


def test_cache_failure_remains_paid_and_cannot_restart_the_inherited_chain(monkeypatch):
    install_fixture(monkeypatch)
    subject=cached.CachedVisibleEngine(EmulatorConfig(game='ms_pacman'))
    before=subject.report()['accounting']['charged_frames']
    def fail(*args):raise MemoryError('synthetic grouping allocation')
    monkeypatch.setattr(subject._group_cache,'group',fail)
    with pytest.raises(MemoryError):subject.step(1)
    report=subject.report()
    assert report['accounting']['charged_frames']==before+4 and report['decisions']==0
    with pytest.raises(ValueError,match='Trajectory stopped'):subject.step(1)
    assert subject.report()==report


def serve_synthetic(connection,config,branch_every,reference_backend,native_build):
    with pytest.MonkeyPatch.context() as patch:
        install_fixture(patch)
        engine_class=cached.CachedVisibleEngine
        def construct(*args,**kwargs):
            engine=engine_class(*args,**kwargs)
            if config.seed==1:
                def fail(*args):raise ValueError('synthetic paid decoder failure')
                engine._decoder=fail
            return engine
        patch.setattr(cached,'CachedVisibleEngine',construct)
        cached._serve_cached(connection,config,branch_every,reference_backend,native_build)


def test_spawned_failure_drains_reports_and_stops_all_workers_without_resend(monkeypatch):
    monkeypatch.setattr(cached,'_serve_cached',serve_synthetic)
    subject=cached.CachedVisiblePool([EmulatorConfig(game='ms_pacman',seed=i) for i in range(2)])
    processes=list(subject._processes)
    try:
        for command in ('getRAM','clone','restore','reward','lives','branch_pair'):
            with pytest.raises(ValueError,match='Unknown'):subject._one(0,command)
        with pytest.raises(ValueError,match='paid decoder'):subject.step([1,0])
        reports=[subject.report(i) for i in range(2)]
        assert reports[0]['ordinary_raw_frames']==4 and reports[1]['ordinary_raw_frames']==0
        assert reports[0]['accounting']['charged_frames']-reports[1]['accounting']['charged_frames']==3
        with pytest.raises(ValueError,match='trajectory stopped'):subject.step([0,0])
        assert reports==[subject.report(i) for i in range(2)]
    finally:subject.close()
    assert not any(p.is_alive() for p in processes)


def test_other_games_refused_before_any_native_construction(monkeypatch):
    monkeypatch.setattr(original,'_ScorerHistoryColdEngine',lambda *args:pytest.fail('native constructor reached'))
    for game in ('asteroids','boxing','frostbite','riverraid','kangaroo'):
        with pytest.raises(ValueError,match='MsPacman'):cached.CachedVisibleEngine(EmulatorConfig(game=game))
        with pytest.raises(ValueError,match='MsPacman'):cached.CachedVisiblePool([EmulatorConfig(game=game)])
