"""Real spawn/pipe with synthetic native engines, no ROM or oracle data."""
import numpy as np
import pytest
from ragnarok.environments import rgb_archive_pool_v70 as pool
from ragnarok.environments.emu_pool import EmulatorConfig
from tests.test_rgb_archive_v70 import create
from tests.test_rgb_trace_pool_v70 import Connection


def synthetic_worker(connection,config,*,archive_limits):
    def factory(_config,*,max_archive_entries,max_archive_bytes):
        subject=create(entries=max_archive_entries,byte_limit=max_archive_bytes)
        subject.accounting=subject._budget.report
        return subject
    pool._RGBArchiveEngine=factory
    pool._serve_rgb_archive(connection,config,archive_limits=archive_limits)


class SyntheticPool(pool.RGBArchivePool):
    _worker_target=staticmethod(synthetic_worker)


def test_two_real_workers_exchange_only_tokens_and_rgb_with_global_capacity_and_cleanup():
    with SyntheticPool([EmulatorConfig(seed=i) for i in range(2)],max_archive_entries=5,max_archive_bytes=2_000_001) as subject:
        processes=list(subject._processes)
        before=subject.observe(0)
        token=subject.archive_save(0)
        subject.step([1,0])
        current=subject.observe(0)
        intervention,control=subject.archive_pair_trace(token,1,index=0)
        assert intervention.raw.shape==control.raw.shape==(8,210,160,3)
        assert not np.array_equal(before,current) and np.array_equal(subject.observe(0),current)
        report=subject.report(0)
        assert set(report)=={'accounting','failure','archive'}
        assert report['accounting']['charged_frames']==1400 and report['accounting']['ordinary_agent_steps']==1
        assert report['archive']['entries']==1 and report['failure'] is None
        reports=[subject.report(i) for i in range(2)]
        assert sum(r['archive']['max_entries'] for r in reports)==4
        assert sum(r['archive']['max_serialized_bytes'] for r in reports)==2_000_000
        subject.archive_drop(token)
        assert subject.report(0)['archive']['serialized_bytes']==0
        assert len(subject.step([0,1]))==2
    assert all(not p.is_alive() for p in processes)


def test_worker_rejects_native_bytes_legacy_snapshots_and_all_oracles(monkeypatch):
    subject=create()
    subject.accounting=subject._budget.report
    monkeypatch.setattr(pool,'_RGBArchiveEngine',lambda *a,**k:subject)
    forbidden=('getRAM','lives','reward','clone','restore','release','_capture','_archive','_decode_saved','archive_import')
    connection=Connection([(name,()) for name in forbidden]+[('archive_save',()),('report',()),('close',())])
    pool._serve_rgb_archive(connection,None,archive_limits=(2,1_000_000))
    assert all(not ok and 'Unknown' in error[1] for ok,error in connection.sent[1:1+len(forbidden)])
    assert connection.sent[-3][0] and isinstance(connection.sent[-3][1],str)
    assert connection.sent[-2][0] and connection.closed


def test_parent_stops_after_failed_archive_action_but_preserves_reports_and_drop():
    with SyntheticPool([EmulatorConfig()],max_archive_entries=1) as subject:
        token=subject.archive_save()
        before=subject.report()
        with pytest.raises(ValueError,match='cap'): subject.archive_save()
        assert subject.report()==before
        for call in (lambda:subject.step([1]),subject.archive_save,lambda:subject.archive_pair_trace(token,1)):
            with pytest.raises(ValueError,match='stopped'): call()
        subject.archive_drop(token)
        assert subject.report()['archive']['entries']==0


@pytest.mark.parametrize('entries,byte_limit', [(1,1000),(True,1000),(20001,1000),(4,2000000001)])
def test_invalid_global_caps_refuse_before_any_worker_starts(entries,byte_limit):
    with pytest.raises(ValueError):
        SyntheticPool([EmulatorConfig(seed=i) for i in range(2)],max_archive_entries=entries,max_archive_bytes=byte_limit)
