"""Synthetic chain fidelity and closed Windows transport, no native emulator."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation import rectangle_visible_pool_v70 as candidate
from ragnarok.evaluation import cached_visible_pool_v70 as cached
from ragnarok.evaluation import visible_reference_pool_v70 as original
from tests.test_cached_visible_pool_v70 import install_fixture
from tests.test_restorable_ram_history_v70 import same_history


def test_step_stop_report_methods_inherited_and_worker_changes_only_engine():
    for name in ('_step', 'step', 'report'):
        assert getattr(candidate.RectangleVisibleEngine, name) is getattr(cached.CachedVisibleEngine, name)
    def tree(fn):
        return ast.parse(textwrap.dedent(inspect.getsource(fn)))
    before, after = tree(cached._serve_cached), tree(candidate._serve_rectangles)
    before.body[0].name = after.body[0].name
    for node in ast.walk(before):
        if isinstance(node, ast.Name) and node.id == 'CachedVisibleEngine':
            node.id = 'RectangleVisibleEngine'
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize('terminal', [999, 300, 509])
def test_pixels_groups_scores_ledgers_digests_and_cold_history_match(monkeypatch, terminal):
    install_fixture(monkeypatch, terminal)
    config = EmulatorConfig(game='ms_pacman')
    baseline, subject = original.VisibleReferenceEngine(config), candidate.RectangleVisibleEngine(config)
    assert vars(subject._group_cache) == {}
    for index in range(129):
        for left, right in zip(baseline.step(index % 2), subject.step(index % 2), strict=True):
            np.testing.assert_array_equal(left, right)
    assert baseline.report() == subject.report()
    same_history(baseline.engine._ale.export_history(), subject.engine._ale.export_history())
    assert subject._decoder is original.mspacman_visible_scene
    assert vars(subject._group_cache) == {}


@pytest.mark.parametrize('diagnostic_failure', [False, True])
def test_paid_failure_stops_even_when_error_diagnostic_allocation_fails(monkeypatch, diagnostic_failure):
    install_fixture(monkeypatch)
    subject = candidate.RectangleVisibleEngine(EmulatorConfig(game='ms_pacman'))
    before = subject.report()['accounting']['charged_frames']
    def fail(*args, **kwargs):
        raise MemoryError('synthetic rectangle allocation failure')
    monkeypatch.setattr(candidate, 'rectangle_visible_groups', fail)
    if diagnostic_failure:
        monkeypatch.setattr(original, 'dict', fail, raising=False)
    with pytest.raises(MemoryError):
        subject.step(1)
    assert subject._cache_chain_stopped
    with pytest.raises(ValueError, match='Trajectory stopped'):
        subject.step(1)
    if diagnostic_failure:
        assert subject._failure is None
        monkeypatch.delattr(original, 'dict')
    report = subject.report()
    assert report['accounting']['charged_frames'] == before + 4
    assert report['decisions'] == 0 and report['failure'] is not None
    assert subject.report() == report


def serve_synthetic(connection, config, branch_every, reference_backend, native_build):
    with pytest.MonkeyPatch.context() as patch:
        install_fixture(patch)
        engine_class = candidate.RectangleVisibleEngine
        def construct(*args, **kwargs):
            engine = engine_class(*args, **kwargs)
            if config.seed == 1:
                def fail(*args):
                    raise ValueError('synthetic paid rectangle decoder failure')
                engine._decoder = fail
            return engine
        patch.setattr(candidate, 'RectangleVisibleEngine', construct)
        candidate._serve_rectangles(connection, config, branch_every, reference_backend, native_build)


def test_spawned_failure_drains_all_reports_without_resend(monkeypatch):
    monkeypatch.setattr(candidate, '_serve_rectangles', serve_synthetic)
    pool = candidate.RectangleVisiblePool([EmulatorConfig(game='ms_pacman', seed=i) for i in range(2)])
    processes = list(pool._processes)
    try:
        for command in ('getRAM', 'clone', 'restore', 'reward', 'lives', 'branch_pair'):
            with pytest.raises(ValueError, match='Unknown'):
                pool._one(0, command)
        with pytest.raises(ValueError, match='paid rectangle decoder'):
            pool.step([1, 0])
        reports = [pool.report(index) for index in range(2)]
        assert reports[0]['ordinary_raw_frames'] == 4 and reports[1]['ordinary_raw_frames'] == 0
        assert reports[0]['accounting']['charged_frames'] - reports[1]['accounting']['charged_frames'] == 3
        with pytest.raises(ValueError, match='trajectory stopped'):
            pool.step([0, 0])
        assert reports == [pool.report(index) for index in range(2)]
    finally:
        pool.close()
    assert not any(process.is_alive() for process in processes)


def test_other_games_refused_before_native_construction(monkeypatch):
    monkeypatch.setattr(original, '_ScorerHistoryColdEngine', lambda *args: pytest.fail('Native constructor reached'))
    for game in ('asteroids', 'boxing', 'frostbite', 'riverraid', 'kangaroo'):
        with pytest.raises(ValueError, match='MsPacman'):
            candidate.RectangleVisibleEngine(EmulatorConfig(game=game))
        with pytest.raises(ValueError, match='MsPacman'):
            candidate.RectangleVisiblePool([EmulatorConfig(game=game)])
