"""Prospective, bounded native wiring of paid RGB traces; never agent training."""
import json
import hashlib
import numpy as np
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.environments.frame_budget_v70 import BudgetExceeded
from ragnarok.environments.rgb_branch_trace_v70 import _RGBTraceColdEngine, RGBBranchTrace
from ragnarok.environments.rgb_trace_difference_v70 import difference
from ragnarok.environments.rgb_trace_pool_v70 import RGBTracePool
from scripts.temporal_validation_seal_v70 import ROOT, sha, utc, runtime, write_new, require_committed

PROTOCOL = 'docs/ARC4_RGB_BRANCH_TRACE_PROTOCOL.json'
PREFLIGHT = 'craft_v6_out/v70_rgb_branch_trace/wiring_preflight_v0.json'
OUTPUT = ROOT/'craft_v6_out/v70_rgb_branch_trace/wiring_v0.json'
GAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
SOURCE_PATHS = (
    'ragnarok/environments/emu_pool.py', 'ragnarok/environments/cold_emu_pool_v70.py',
    'ragnarok/environments/frame_budget_v70.py', 'ragnarok/environments/rgb_branch_trace_v70.py',
    'ragnarok/environments/rgb_trace_difference_v70.py', 'ragnarok/environments/rgb_trace_pool_v70.py',
    'ragnarok/evaluation/triple_pooled_fixtures_v70.py', 'scripts/temporal_validation_seal_v70.py',
    'tests/test_rgb_branch_trace_v70.py', 'tests/test_rgb_trace_difference_v70.py',
    'tests/test_rgb_trace_pool_v70.py', 'scripts/check_rgb_branch_trace_v70.py',
    'tests/test_rgb_branch_trace_wiring_v70.py', PROTOCOL, 'docs/ARC4_BRANCH_TRACE_DRAFT.md')
SHARED = dict(frame_skip=4, mode='lab', sticky_probability=.25, experiment_fraction=.2,
              max_charged_frames_per_engine=2000, branch_action=5, control_action=0,
              requested_branch_frames=8, restore_charge=60)
NORMAL = dict(seed_per_game=18700, ordinary_decisions_before_branch=128,
              ordinary_decisions_after_branch=16, max_episode_frames=108000, engines_per_game=3)
TERMINALS = [dict(game='pong', seed=18711, ordinary_decisions_before_branch=127,
                  max_episode_frames=509, expected_trace_frames_per_branch=1),
             dict(game='pong', seed=18712, ordinary_decisions_before_branch=128,
                  max_episode_frames=509, expected_trace_frames_per_branch=0)]
SPAWN = dict(game='pong', seed=18720, ordinary_decisions_before_branch=128,
             ordinary_decisions_after_branch=16, max_episode_frames=108000, workers=1, direct_witnesses=1)


def source_hashes():
    return {p: sha(ROOT/p) for p in SOURCE_PATHS}


def preflight():
    sources = source_hashes()
    protocol = json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    review = json.loads((ROOT/PREFLIGHT).read_text(encoding='utf-8'))
    current_runtime = runtime()
    if (protocol.get('games') != list(GAMES) or protocol.get('shared') != SHARED
            or protocol.get('normal_case') != NORMAL or protocol.get('terminal_cases') != TERMINALS
            or protocol.get('spawn_case') != SPAWN or protocol.get('agent_training_allowed') is not False
            or review.get('passed') is not True or review.get('source_sha256') != sources
            or review.get('runtime') != current_runtime):
        raise ValueError('Complete exact independent RGB trace wiring preflight required')
    for p in (*SOURCE_PATHS, PREFLIGHT):
        require_committed(p)
    return dict(source_sha256=sources, runtime=current_runtime, preflight_review_sha256=sha(ROOT/PREFLIGHT))


def config(game, seed, limit):
    return EmulatorConfig(game=game, seed=seed, frame_skip=4, mode='lab',
        max_episode_frames=limit, max_charged_frames=2000, sticky_probability=.25, experiment_fraction=.2)


def same_state(left, right):
    return (left._ale.getEpisodeFrameNumber() == right._ale.getEpisodeFrameNumber()
        and left._last_action == right._last_action and left._sticky_rng.getstate() == right._sticky_rng.getstate()
        and len(left._frames) == len(right._frames)
        and all(np.array_equal(a, b) for a, b in zip(left._frames, right._frames, strict=True)))


def frame_witness_pair(engine):
    if not engine._budget.can_experiment(16, restores=2):
        raise BudgetExceeded('Complete frame witness pair must fit before its first act')
    anchor = engine._capture()
    result = []
    for action in (5, 0):
        raw, pooled = [], []
        try:
            for _ in range(8):
                if engine._ale.game_over():
                    break
                pooled.append(engine._advance(action, 1, experimental=True))
                raw.append(engine.observe(raw=True))
            endpoint = engine.observe()
            empty = (0, *endpoint.shape)
            result.append(RGBBranchTrace(np.stack(raw) if raw else np.empty(empty, dtype=np.uint8),
                np.stack(pooled) if pooled else np.empty(empty, dtype=np.uint8), endpoint))
        finally:
            engine._restore(anchor)
    return tuple(result)


def same_traces(left, right):
    return len(left) == len(right) == 2 and all(
        np.array_equal(getattr(a, field), getattr(b, field))
        for a, b in zip(left, right, strict=True) for field in ('raw', 'pooled', 'endpoint'))


def trace_digests(traces):
    result = {}
    for kind in ('raw', 'pooled'):
        delta = difference(*traces, image_kind=kind)
        digest = hashlib.sha256()
        for array in (delta.changed, delta.first_changed_frame):
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
            digest.update(array.tobytes())
        result[kind] = dict(sha256=digest.hexdigest(), compared_frames=len(delta.changed),
            unpaired_intervention_frames=delta.unpaired_intervention_frames,
            unpaired_control_frames=delta.unpaired_control_frames)
    return result


def direct_case(create, game, seed, warmup, limit, expected=None):
    cfg = config(game, seed, limit)
    subject = create(f'{game}:{seed}:trace', lambda: _RGBTraceColdEngine(cfg))
    endpoint = create(f'{game}:{seed}:endpoint_witness', lambda: _ColdPixelEngine(cfg))
    witness = create(f'{game}:{seed}:frame_witness', lambda: _ColdPixelEngine(cfg))
    for step in range(warmup):
        images = [e.step((step//4)%18) for e in (subject, endpoint, witness)]
        if not all(np.array_equal(images[0], image) for image in images[1:]):
            raise ValueError('Ordinary RGB differs before branch')
    if expected is not None and any(e._ale.getEpisodeFrameNumber() != 509-expected
                                   for e in (subject, endpoint, witness)):
        raise ValueError('Prescribed terminal boundary was not reached; no replacement seed')
    traces = subject.branch_pair_trace(5, control=0, frames=8)
    ends = endpoint.branch_pair(5, control=0, frames=8)
    frames = frame_witness_pair(witness)
    if (not same_traces(traces, frames) or any(not np.array_equal(t.endpoint, e)
            for t, e in zip(traces, ends, strict=True))
            or not same_state(subject, endpoint) or not same_state(subject, witness)):
        raise ValueError('Per-frame traces, endpoints or restored state differ')
    lengths = [len(t.raw) for t in traces]
    if expected is not None and lengths != [expected, expected]:
        raise ValueError('Prescribed truncated trace length differs')
    for step in range(warmup, warmup+16):
        images = [e.step((step//4)%18) for e in (subject, endpoint, witness)]
        if (not all(np.array_equal(images[0], image) for image in images[1:])
                or not same_state(subject, endpoint) or not same_state(subject, witness)):
            raise ValueError('Ordinary continuation or next cold differs after restoration')
    a, b, c = [e.accounting() for e in (subject, endpoint, witness)]
    if (a != b or a['experiment_agent_steps'] != 2 or c['experiment_agent_steps'] != sum(lengths)
            or {k:v for k,v in a.items() if k != 'experiment_agent_steps'} !=
               {k:v for k,v in c.items() if k != 'experiment_agent_steps'}):
        raise ValueError('Paid witness ledgers differ beyond the declared call counter')
    return dict(game=game, seed=seed, ordinary_decisions=warmup+16, trace_frames=lengths,
                difference=trace_digests(traces), passed=True)


def run():
    attempt = OUTPUT.with_name(OUTPUT.stem+'_attempt.json')
    if OUTPUT.exists() or attempt.exists():
        raise ValueError('Preserve all wiring attempts/results; no retry or overwrite')
    proof = preflight()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='started', started_utc=utc(), **proof)
    write_new(attempt, record)
    objects, collected, cases = [], [], []
    pool = None
    constructing = None
    constructor_count = 0

    def create(name, factory):
        nonlocal constructing, constructor_count
        if constructor_count >= 26:
            raise ValueError('Declared constructor cap reached')
        constructing = name
        constructor_count += 1
        obj = factory()
        objects.append((name, obj))
        constructing = None
        return obj

    def drain():
        for name, obj in objects:
            try:
                collected.append(dict(name=name, accounting=obj.accounting(),
                    failure=getattr(obj, '_trace_failure', None)))
            except BaseException as error:
                collected.append(dict(name=name, accounting_error=str(error)))
        objects.clear()

    try:
        for game in GAMES:
            cases.append(direct_case(create, game, 18700, 128, 108000))
            drain()
        for case in TERMINALS:
            cases.append(direct_case(create, case['game'], case['seed'],
                case['ordinary_decisions_before_branch'], 509, case['expected_trace_frames_per_branch']))
            drain()
        cfg = config('pong', 18720, 108000)
        direct = create('spawn_direct', lambda: _RGBTraceColdEngine(cfg))
        constructing = 'spawn_worker'
        constructor_count += 1
        pool = RGBTracePool([cfg])
        constructing = None
        for name in ('getRAM', 'reward', 'lives', '_capture', 'clone', 'restore', 'release'):
            try:
                pool._one(0, name)
            except ValueError as error:
                if 'Unknown' not in str(error):
                    raise
            else:
                raise ValueError('Forbidden native/oracle RPC became available')
        for step in range(128):
            if not np.array_equal(pool.step([(step//4)%18])[0], direct.step((step//4)%18)):
                raise ValueError('Spawn ordinary RGB differs')
        traces = pool.branch_pair_trace(5, control=0, frames=8)
        if not same_traces(traces, direct.branch_pair_trace(5, control=0, frames=8)):
            raise ValueError('Spawn trace payload differs')
        for step in range(128, 144):
            if not np.array_equal(pool.step([(step//4)%18])[0], direct.step((step//4)%18)):
                raise ValueError('Spawn continuation RGB differs')
        report = pool.report()
        collected.append(dict(name='spawn_worker', **report))
        if report != dict(accounting=direct.accounting(), failure=direct._trace_failure):
            raise ValueError('Spawn/direct complete report differs')
        processes = list(pool._processes)
        pool.close()
        pool = None
        if any(p.is_alive() for p in processes):
            raise ValueError('Spawn worker did not close')
        drain()
        cases.append(dict(game='pong', seed=18720, spawn=True, passed=True, difference=trace_digests(traces)))
        if (constructor_count != 26 or len(collected) != 26
                or any(r.get('failure') is not None or 'accounting' not in r
                       or r['accounting']['charged_frames'] > 2000 for r in collected)
                or sum(r['accounting']['charged_frames'] for r in collected) > 52000
                or preflight() != proof):
            raise ValueError('Constructor count, complete reports or final proofs differ')
        result = dict(kind='paid_rgb_branch_trace_native_wiring', passed=True, created_utc=utc(),
            **proof, cases=cases, engine_reports=collected, constructor_count=constructor_count,
            charged_frames=sum(r['accounting']['charged_frames'] for r in collected),
            accounting_complete=True, agent_training_allowed=False, gate_result=None)
        write_new(OUTPUT, result)
        record.update(status='complete', completed_utc=utc(), result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        drain()
        if pool is not None and not any(r['name']=='spawn_worker' for r in collected):
            try:
                collected.append(dict(name='spawn_worker', **pool.report()))
            except BaseException as report_error:
                collected.append(dict(name='spawn_worker', accounting_error=str(report_error)))
        record.update(status='failed', failed_utc=utc(), error_type=type(error).__name__, error=str(error),
            completed_cases=cases, engine_reports=collected, constructor_count=constructor_count,
            constructor_without_return=constructing,
            known_charged_frames=sum(r['accounting']['charged_frames'] for r in collected if 'accounting' in r),
            accounting_complete=False)
        raise
    finally:
        try:
            if pool is not None:
                pool.close()
        finally:
            attempt.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    result = run()
    print(json.dumps({k:result[k] for k in ('passed', 'constructor_count', 'charged_frames')}))
