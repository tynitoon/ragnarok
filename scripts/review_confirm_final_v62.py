"""Independent metadata-only closure review; never imports experimental code.

Hashes checkpoint bytes without decoding them. No score, inference, world
construction, Torch, NumPy, subprocess experiment, or mutable old artifact.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

FREEZE = 'a189ea5bb0f9c8b7e9016a50aefc1cbca603426f'
MERGED = '037e2fb8fd04eff665d4690f077647046398f77423d000c8e552750ebff62685'
CRITICAL = [f'scripts/{n}.py' for n in (
    'gate_v61', 'pretrain_v61', 'probe_v61', 'confirm_v61', 'score_v61',
    'pair_net_v61', 'pair_store_v61', 'hand_v61', 'sweeper_v61',
    'hidden_recipe_v55', 'depth_scaling_v49', 'childhood_v50', 'meta_manager_v51')]
CRITICAL += ['ragnarok/environments/law_world.py', 'ragnarok/environments/tech_tree.py',
             'ragnarok/learning/ppo_discrete.py', 'ragnarok/infrastructure/device.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc(value):
    d = datetime.fromisoformat(value)
    require(d.utcoffset() is not None, 'Timezone is required')
    return d


def strict_pairs(items):
    out = {}
    for k, v in items:
        require(k not in out, f'Duplicate JSON key {k}')
        out[k] = v
    return out


def finite(v, low=0, high=1):
    return type(v) in (int, float) and math.isfinite(v) and low <= v <= high


def review(root, result_commit, review_started):
    start = time.monotonic()
    out = root / 'craft_v6_out'
    score = out / 'v61_confirm_score.json'
    target = out / 'v62_confirm_final_review.json'
    require(not target.exists(), 'Preserve the existing final review')
    require(not score.exists(), 'The production score must still be absent')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only review required')
    evidence = {}
    git_calls = 0

    def git(*args):
        nonlocal git_calls
        git_calls += 1
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])

    def bind(relative, expected=None):
        path = root / relative
        require(path.is_file(), f'Missing evidence {relative}')
        actual = sha(path)
        require(expected is None or actual == expected, f'Changed evidence {relative}')
        require(relative not in evidence or evidence[relative] == actual, f'Changed during review {relative}')
        evidence[relative] = actual
        return path

    def read(relative):
        path = bind(relative)
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=strict_pairs,
                          parse_constant=lambda v: math.inf if relative == 'craft_v6_out/v61_gate_v62.json' and v == 'Infinity'
                          else (_ for _ in ()).throw(ValueError(v)))

    def text(relative):
        return bind(relative).read_text(encoding='utf-8')

    def bind_map(mapping, prefix=''):
        for relative, expected in mapping.items():
            bind(prefix + relative, expected)

    def objids(commit, paths):
        raw = git('ls-tree', '-r', '--full-tree', commit, '--', *sorted(paths)).decode()
        result = {}
        for line in raw.splitlines():
            meta, path = line.split('\t', 1)
            mode, kind, oid = meta.split()
            require(kind == 'blob' and mode == '100644', f'Plain committed file required {path}')
            result[path] = oid
        require(set(result) == set(paths), f'Missing committed files at {commit}')
        return result

    def before(path, boundary):
        raw = git('log', '-1', '--format=%H|%cI', result_commit, '--', path).decode().strip()
        commit, timestamp = raw.split('|')
        require(utc(timestamp) < boundary, f'Commit after dispatch: {path}')
        return dict(commit=commit, committed_utc=timestamp, before_utc=boundary.isoformat())

    final = read('craft_v6_out/v62_confirm_second_recovery_state.json')
    proof = final['provenance']
    initial = read('craft_v6_out/v62_resume_state.json')
    original = read('craft_v6_out/v62_confirm_state.json')
    recovery = read('craft_v6_out/v62_confirm_recovery_state.json')
    prior = initial['provenance']
    require(final['status'] == 'confirm_complete_review_required' and 'error' not in final,
            'A clean completed controller is required')
    require(final['controller_pid'] == 11880 and final['detached_from_job'] is True,
            'Reviewed controller identity differs')
    require(final['merged_sha256'] == MERGED, 'Unexpected final aggregate')
    require(prior['freeze_commit'] == FREEZE and set(prior['files']) == set(CRITICAL), 'Freeze registry differs')
    require(git('rev-parse', 'v62-stage0-freeze^{commit}').decode().strip() == FREEZE, 'Freeze tag moved')
    sources = {}
    for p in CRITICAL:
        expected = prior['files'][p]
        frozen = hashlib.sha256(git('show', f'{FREEZE}:{p}')).hexdigest()
        require(frozen == expected['working_sha256'] == expected['frozen_sha256'], f'Frozen bytes differ {p}')
        bind(p, frozen)
        sources[p] = frozen
    plan = text('ARC3_PLAN.md')
    frozen_plan = git('show', f'{FREEZE}:ARC3_PLAN.md').decode()
    section = lambda t: t.split('### 12.6 ', 1)[1].split('\n## ', 1)[0].strip()
    require(section(plan) == section(frozen_plan), 'Authoritative section 12.6 differs')
    text('docs/ARC3_CONFIRM_FINAL_REVIEW_DRAFT.md')
    bind_map(prior['inputs'], 'craft_v6_out/')
    bind('craft_v6_out/v61_gate_v62.json', prior['gate_sha256'])
    gate = read('craft_v6_out/v61_gate_v62.json')
    nonfinite = []
    def visit(value, path=()):
        if isinstance(value, dict):
            for k,v in value.items(): visit(v,path+(k,))
        elif isinstance(value, list):
            for k,v in enumerate(value): visit(v,path+(k,))
        elif isinstance(value,float) and not math.isfinite(value):
            nonfinite.append((path,value))
    visit(gate)
    require(nonfinite == [(('summary','b_star_median','2',0),math.inf)], 'Unexpected nonfinite gate value')
    require(gate['worlds'] == [8102, 8103] and gate['b_max'] == 6 and gate['num_envs'] == 64
            and gate['store_persists'] is True and gate['summary']['N'] == 9 and gate['summary']['B_test'] == 6
            and all(gate['summary'][k] is True for k in ('reachable', 'fresh_learns', 'room', 'proceed')), 'Gate prerequisite differs')
    require(initial['status'] == 'probe_passed_review_required' and
            [s['stage'] for s in initial['completed']] == ['pretrain_s0', 'pretrain_s1', 'pretrain_s2', 'probe'], 'Pretraining/probe incomplete')
    require(original['provenance']['prior_provenance'] == prior and
            original['provenance']['prior_completed'] == initial['completed'], 'Original prerequisite chain differs')
    bind('scripts/resume_v62.py', prior['launcher_sha256'])
    bind('scripts/confirm_resume_v62.py', original['provenance']['launcher_sha256'])
    bind('craft_v6_out/v62_resume_state.json', original['provenance']['prior_state_sha256'])
    pretrain_worlds = []
    pretrain_hashes = []
    for s in range(3):
        bind_map(initial['completed'][s]['artifacts'], 'craft_v6_out/')
        data = read(f'craft_v6_out/v61_pre_s{s}.json')
        require(data['lineage'] == s and data['b_max'] == 3 and len(data['worlds']) == 4, 'Wrong pretraining design')
        expected_worlds = list(range(8200 + 10*s, 8204 + 10*s))
        require([w['world'] for w in data['worlds']] == expected_worlds, 'Pretraining worlds differ')
        log = text(f'craft_v6_out/v61_pre_s{s}.log')
        require('skipped (logged)' not in log and '[resume]' not in log, 'Unexpected pretraining exclusion/replay')
        nav = [float(v) for v in re.findall(r'nav min ([0-9.]+)', log)]
        require(len(nav) == 4 and all(.85 <= v <= 1 for v in nav), 'Pretraining nav log differs')
        for k, w in enumerate(data['worlds'], 1):
            require(w['k'] == k and w['tiers'] == [2,3,4,4] and len(set(w['goals'])) == 4 and len(w['rows']) == 4, 'Incomplete pretraining stream')
            for g, tier, row in zip(w['goals'], w['tiers'], w['rows'], strict=True):
                require(row['goal'] == g and row['tier'] == tier and len(row['curve']) == 4 and all(finite(v) for v in row['curve']), 'Invalid pretraining curve')
                require(all(len(row[z]) == 3 and all(type(v) is int and v >= 0 for v in row[z]) for z in ('rows','demos')), 'Incomplete pretraining counters')
            ck = f'craft_v6_out/v61_pre_s{s}_M{k}.pt'
            require(bind(ck).stat().st_size > 0, 'Empty checkpoint')
            pretrain_hashes.append(evidence[ck])
            require(f"WORLD {w['world']} (k={k}) | stream {w['goals']} tiers {w['tiers']}" in log, 'Pretraining log stream differs')
        pretrain_worlds.append(expected_worlds)
        read(f'craft_v6_out/v62_pretrain_s{s}_receipt.json')
    require(len(set(pretrain_hashes)) == 12, 'Twelve distinct pretraining checkpoint artifacts required')
    probe = read('craft_v6_out/v61_probe.json')
    probe_review = read('craft_v6_out/v62_probe_review.json')
    require(probe_review['kind'] == 'v62_probe_independent_review' and probe_review['passed'] is True, 'Probe review not PASS')
    require(probe_review['source_sha256'] == sources, 'Probe sources differ')
    bind('craft_v6_out/v61_probe.json', probe_review['probe_sha256'])
    bind('craft_v6_out/v62_resume_state.json', probe_review['controller_state_sha256'])
    bind_map(probe_review['completed_artifact_sha256'], 'craft_v6_out/')
    bind_map(initial['completed'][3]['artifacts'], 'craft_v6_out/')
    require(probe['world'] == 8150 and probe['lineage'] == 0, 'Wrong burned probe')
    m, r = probe['arms']['M4'], probe['arms']['R']
    require(m['n'] == r['n'] == 768 and m['lawful']*768 == 618 and r['lawful']*768 == 398, 'Probe counts differ')
    pool = (m['lawful'] + r['lawful'])/2
    se = math.sqrt(pool*(1-pool)*(2/768))
    require(m['lawful']-r['lawful'] >= 3*se and probe['summary']['passed'] is True, 'Frozen probe STOP')
    for key, val in [('r_M',m['lawful']),('r_R',r['lawful']),('se',se)]:
        require(math.isclose(probe['summary'][key], val, abs_tol=1e-12), 'Probe summary mismatch')

    python = 'C:\\Python314\\python.exe'
    expected_pre = [dict(stage=f'pretrain_s{s}', lineage=s, argv=[python,'-u','-m','scripts.pretrain_v61','--lineage',str(s),'--b-max','3','--num-envs','64','--store-persists','--resume']) for s in range(3)]
    expected_pre.append(dict(stage='probe', argv=[python,'-u','-m','scripts.probe_v61','--lineage','0','--num-envs','64']))
    commands = [dict(stage=f'confirm_s{s}', lineage=s, argv=[python,'-u','-m','scripts.confirm_v61','--lineage',str(s),'--n-units','3','--b-max','6','--num-envs','64','--store-persists','--g-units','2','--resume']) for s in range(3)]
    require(prior['commands'] == expected_pre and prior['python'] == python, 'Pretraining/probe command changed')
    require(original['provenance']['commands'] == recovery['provenance']['commands'] == commands and proof['command'] == commands[2], 'Confirmatory command changed')
    require(proof['prior_provenance'] == recovery['provenance'], 'Recovery provenance differs')
    bind_map(recovery['provenance']['recovery']['immutable_files'])
    first = read('craft_v6_out/v62_confirm_interruption_20260909/interruption.json')
    correction = read('craft_v6_out/v62_confirm_interruption_20260909/budget_correction.json')
    first_review = read('craft_v6_out/v62_confirm_recovery_review.json')
    require(first_review['passed'] is True and correction['evidence_sha256'] == evidence['craft_v6_out/v62_confirm_interruption_20260909/interruption.json'], 'First recovery review differs')
    charged = math.ceil((utc(first['last_boot_utc']) - utc(first['original_started_utc'])).total_seconds())
    require(charged == correction['charged_upper_bound_seconds'] == 2166 and first['charged_upper_bound_seconds'] == 9366, 'UTC correction missing')
    require(first['observed_checkpoints'] == [] and len(first['snapshots']) == 4, 'Unexpected first interruption artifacts')
    for item in first['snapshots']:
        saved = bind('craft_v6_out/v62_confirm_interruption_20260909/'+item['path'], item['sha256'])
        require(saved.stat().st_size == item['bytes'], 'First snapshot size differs')
        current = bind('craft_v6_out/'+item['path'])
        require(current.read_bytes().startswith(saved.read_bytes()) if item['path']=='v61_confirm_s0.log' else current.read_bytes()==saved.read_bytes(), 'First historical record overwritten')
    second = read('craft_v6_out/v62_confirm_interruption_20260910/interruption.json')
    bind_map(proof['evidence_sha256'])
    require(len(proof['evidence_sha256']) == 12 and len(second['snapshots']) == 6, 'Second evidence set differs')
    for item in second['snapshots']:
        saved = bind('craft_v6_out/v62_confirm_interruption_20260910/'+item['snapshot'], item['sha256'])
        require(saved.stat().st_size == item['bytes'], 'Second snapshot size differs')
        current = bind(item['source'])
        require(current.read_bytes().startswith(saved.read_bytes()) if item['snapshot']=='v61_confirm_s2.log' else current.read_bytes()==saved.read_bytes(), 'Second historical record overwritten')
    require(recovery['completed'] == proof['completed_before'] == final['completed'][:2], 'Completed first six worlds replaced')
    require(sum(len(s['artifacts']) for s in proof['completed_before']) == 28, 'Preserved artifact count differs')
    for s in proof['completed_before']:
        bind_map(s['artifacts'], 'craft_v6_out/')
    bind('craft_v6_out/v61_confirm_ckpt/s2_8320_M4.json', proof['preserved_m4_sha256'])
    old_merged = read('craft_v6_out/v62_confirm_interruption_20260910/v61_confirm.json')
    second_review = read('craft_v6_out/v62_confirm_second_recovery_review.json')
    require(second_review['kind'] == 'v62_second_recovery_independent_review' and second_review['passed'] is True, 'Second review not PASS')
    require(all(second_review[k] == v for k,v in proof.items() if k not in ('review_sha256','launch_receipt_sha256')), 'Second review/proof changed')
    bind('craft_v6_out/v62_confirm_second_recovery_review.json', proof['review_sha256'])
    bind_map(proof['source_sha256'])
    launch = read('craft_v6_out/v62_confirm_second_recovery_launch.json')
    dispatch = read('craft_v6_out/v62_confirm_second_recovery_wmi_dispatch.json')
    bind('craft_v6_out/v62_confirm_second_recovery_launch.json', proof['launch_receipt_sha256'])
    require(launch['status'] == 'dispatched' and launch['controller_pid'] == final['controller_pid'] and launch['detached_from_job'] is True
            and launch['source_sha256'] == proof['source_sha256'] and launch['review_sha256'] == proof['review_sha256']
            and launch['argv'] == [python,'-u','-m','scripts.recover_confirm_after_interrupt_v62']
            and launch['absolute_deadline_utc'] == proof['absolute_deadline_utc'], 'Launch handshake differs')
    require(dispatch['return_value'] == 0 and dispatch['launcher_pid'] == launch['launcher_pid'] and dispatch['show_window'] == 0, 'WMI dispatch differs')
    require(Path(launch['cwd']).resolve() == root and Path(dispatch['cwd']).resolve() == root, 'Launch cwd differs')
    require(utc(dispatch['observed_utc']) <= utc(launch['started_utc']) <= utc(final['started_utc']), 'Dispatch chronology differs')
    stdout = text('craft_v6_out/v62_confirm_second_recovery_stdout.log')
    require('Child PID 21216: '+repr(commands[2]['argv']) in stdout and stdout.count('Child PID') == 1
            and stdout.rstrip().endswith('Stopped: confirm_complete_review_required'), 'Final supervisor log differs')
    require(bind('craft_v6_out/v62_confirm_second_recovery_stderr.log').stat().st_size == 0, 'Final stderr is not empty')
    prior_stdout = text('craft_v6_out/v62_confirm_recovery_stdout.log')
    require([ast.literal_eval(x) for x in re.findall(r'Child PID \d+: (\[.*\])',prior_stdout)] == [c['argv'] for c in commands], 'Prior actual commands differ')
    for name in ('v62_resume_stdout.log','v62_resume_stderr.log','v62_confirm_stdout.log','v62_confirm_stderr.log'):
        bind('craft_v6_out/'+name)

    units = []
    unit_metadata = []
    artifacts = {}
    checkpoint_names = []
    for s in range(3):
        lineage = read(f'craft_v6_out/v61_confirm_s{s}.json')
        require(lineage['lineage'] == s and lineage['b_max'] == 6 and lineage['store_persists'] is True and len(lineage['units']) == 3, 'Lineage design differs')
        require([u['world'] for u in lineage['units']] == list(range(8300+10*s,8303+10*s)), 'World admission/order differs')
        log = text(f'craft_v6_out/v61_confirm_s{s}.log')
        require('skipped (logged)' not in log, 'Unexpected excluded world')
        nav = [float(v) for v in re.findall(r'nav min ([0-9.]+)', log)]
        require(len(nav) == (4 if s in (0,2) else 3) and all(.85 <= v <= 1 for v in nav), 'Confirmatory nav log differs')
        require(log.count('CONFIRMATORY lineage') == (2 if s in (0,2) else 1) and log.count('done: 3 units') == 1, 'Unexpected extra experiment attempt')
        require(re.findall(r'\[resume\] (\S+)',log) == (['s2_8320_M4'] if s == 2 else []), 'Unexpected resumed arm')
        completed_map = final['completed'][s]['artifacts']
        require(final['completed'][s]['stage'] == f'confirm_s{s}', 'Controller lineage order differs')
        bind_map(completed_map,'craft_v6_out/')
        expected_files = {f'v61_confirm_s{s}.json'}
        for index,u in enumerate(lineage['units']):
            require(u['id'] == f"s{s}_{u['world']}" and u['lineage']==s and u['b_max']==6 and u['store_persists'] is True
                    and u['tiers']==[2,3,4] and len(u['goals'])==len(set(u['goals']))==3
                    and set(u['arms'])=={'M4','Fa','Fb','L','G'}, 'Malformed unit')
            require(f"UNIT {u['id']} | stream {u['goals']} tiers {u['tiers']}" in log, 'Unit log metadata differs')
            for arm,rows in u['arms'].items():
                if arm=='G' and not (s==0 and index<2):
                    require(rows is None,'Undeclared G prime unit')
                    continue
                require(isinstance(rows,list) and len(rows)==3,'Incomplete arm')
                for g,tier,row in zip(u['goals'],u['tiers'],rows,strict=True):
                    require(row['goal']==g and row['tier']==tier and len(row['curve'])==7 and all(finite(v) for v in row['curve']), 'Incomplete goal curve')
                    if arm=='L':
                        require(len(set(row['curve']))==1,'L reference must be flat')
                    else:
                        require(all(isinstance(row.get(k),list) and len(row[k])==6 and all(type(v)is int and v>=0 for v in row[k]) for k in ('rows','demos')), 'Missing round counters')
                relative=f"v61_confirm_ckpt/{u['id']}_{arm}.json"
                require(read('craft_v6_out/'+relative)==rows,'Checkpoint differs from aggregate')
                if not (s == 2 and index == 0 and arm == 'M4') and arm != 'L':
                    current_log = log.rsplit('CONFIRMATORY lineage',1)[1]
                    unit_log = current_log.split(f"UNIT {u['id']} |",1)[1].split('\nUNIT ',1)[0]
                    label = "G'" if arm == 'G' else arm
                    logged = re.findall(r'\['+re.escape(label)+r'\] goal (\d+) \(tier (\d+)\): curve (\[[^\]]*\])',unit_log)
                    require([(int(g),int(t),ast.literal_eval(c)) for g,t,c in logged] ==
                            [(r['goal'],r['tier'],r['curve']) for r in rows], 'Closed goal log differs from checkpoint')
                expected_files.add(relative)
                checkpoint_names.append(relative.split('/')[-1])
            unit_metadata.append(dict(id=u['id'],lineage=s,world=u['world'],goals=u['goals'],tiers=u['tiers'],g_prime=u['arms']['G'] is not None,
                                      fresh_init_seeds=[100000+1000*s+10*(u['world']-8000)+a for a in (1,2)]))
        require(set(completed_map)==expected_files,'Unexpected controller artifact set')
        artifacts.update(completed_map)
        units.extend(lineage['units'])
    require(len(artifacts)==41 and len(checkpoint_names)==38,'Wrong complete artifact count')
    require(sorted(p.name for p in (out/'v61_confirm_ckpt').iterdir())==sorted(checkpoint_names),'Unexpected production checkpoint')
    merged=read('craft_v6_out/v61_confirm.json')
    bind('craft_v6_out/v61_confirm.json',MERGED)
    require(merged==dict(b_max=6,store_persists=True,units=units),'Aggregate is not exact concatenation')
    require(old_merged==dict(b_max=6,store_persists=True,units=units[:6]),'Original six-world aggregate was not preserved')
    require(re.findall(r'\[Fa\] goal (\d+)',text('craft_v6_out/v62_confirm_interruption_20260910/v61_confirm_s2.log'))==['5'], 'Interrupted Fa scope differs')
    suffix=(root/'craft_v6_out/v61_confirm_s2.log').read_bytes()[len((root/'craft_v6_out/v62_confirm_interruption_20260910/v61_confirm_s2.log').read_bytes()):].decode()
    fa=[int(g) for g in re.findall(r'\[Fa\] goal (\d+)',suffix)]
    require(fa==[g for u in units[6:] for g in u['goals']], 'Resumed Fa was not a complete stream')

    remaining=42*3600-6.7*3600-16124-137.15
    require(prior['remaining_seconds']==remaining and prior['precheck_seconds']==137.15,'Initial budget differs')
    require(original['provenance']['design']['remaining_seconds']==remaining-initial['elapsed_seconds'],'Pre-probe expenditure lost')
    allowance=remaining-initial['elapsed_seconds']-charged
    require(allowance==84742.0902837==correction['remaining_seconds']==proof['original_recovery_allowance_seconds'], 'Corrected allowance differs')
    origin=utc(recovery['started_utc'])
    require(origin==utc('2026-09-09T09:37:40Z') and proof['original_recovery_started_utc']==recovery['started_utc'],'Recovery origin reset')
    deadline=datetime.fromtimestamp(math.floor(origin.timestamp()+allowance),timezone.utc)
    stopped=utc(final['stopped_utc'])
    started=utc(final['started_utc'])
    require(deadline==utc('2026-09-10T09:10:02Z')==utc(proof['absolute_deadline_utc']) and started<stopped<deadline,'Original deadline exceeded/reset')
    require(abs(final['initial_remaining_seconds']-(deadline-started).total_seconds())<.01 and 0<final['elapsed_seconds']<final['initial_remaining_seconds'],'Invalid monotonic allowance')
    base_cost=6.7*3600+16124+137.15+initial['elapsed_seconds']+charged
    window=(stopped-origin).total_seconds()
    total=base_cost+window
    require(total<42*3600,'Whole ARC3 ceiling exceeded')
    chronology={
        'probe_review':before('craft_v6_out/v62_probe_review.json',utc(original['started_utc'])),
        'first_recovery_review':before('craft_v6_out/v62_confirm_recovery_review.json',origin),
        'second_recovery_review':before('craft_v6_out/v62_confirm_second_recovery_review.json',utc(launch['started_utc'])),
    }
    for p in proof['source_sha256']:
        chronology[p]=before(p,utc(launch['started_utc']))
    result_time=git('show','-s','--format=%cI',result_commit).decode().strip()
    require(utc(result_time)>=stopped,'Closed result commit precedes closure')
    git('merge-base','--is-ancestor',result_commit,'HEAD')
    self_path=Path(__file__).resolve().relative_to(root).as_posix()
    bind(self_path)
    committed=objids(result_commit,set(evidence)-{self_path})
    head_ids=objids('HEAD',evidence)
    require(all(head_ids[p]==v for p,v in committed.items()),'Evidence/source blob changed after result commit')
    for p,oid in head_ids.items():
        actual=git('hash-object',f'--path={p}',str(root/p)).decode().strip()
        require(actual==oid,'Working file differs from committed blob '+p)

    command = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | "
               "Where-Object { $_.ProcessId -in 11880,21216 -or $_.CommandLine -match "
               "'scripts[.\\\\/](confirm_v61|confirm_resume_v62|recover_confirm_v62|recover_confirm_after_interrupt_v62)' } | "
               "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    processes=subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',command],text=True).strip()
    require(not processes,'Experiment process remains active')
    for p,h in evidence.items():
        require(sha(root/p)==h,'Evidence changed during final review '+p)
    require(not score.exists(),'Score appeared during the review')
    ended=datetime.now(timezone.utc)
    review_seconds=(ended-review_started).total_seconds()
    require(review_seconds>=0,'Review start is in the future')
    require(base_cost+math.ceil(window)+math.ceil(review_seconds)<42*3600,'Conservative review-inclusive ceiling exceeded')
    require(not any(n=='torch' or n.startswith('numpy') for n in sys.modules),'Scientific runtime unexpectedly imported')
    receipt=dict(
        kind='v62_confirm_final_independent_review',passed=True,reviewer='/root/review_recovery_v62',
        reviewed_utc=ended.isoformat(),freeze_commit=FREEZE,result_commit=result_commit,
        merged_sha256=MERGED,controller_state_sha256=evidence['craft_v6_out/v62_confirm_second_recovery_state.json'],
        source_sha256=sources,orchestration_source_sha256={p:evidence[p] for p in ('scripts/resume_v62.py','scripts/confirm_resume_v62.py','scripts/recover_confirm_v62.py',*proof['source_sha256'])},
        evidence_sha256=evidence,committed_blob_ids=head_ids,chronology=chronology,
        design=dict(b_max=6,n_units=9,n_lineages=3,n_units_per_lineage=3,num_envs=64,store_persists=True,g_prime_units=['s0_8300','s0_8301']),
        complete_units=unit_metadata,artifact_counts=dict(lineage_json=3,arm_checkpoints=38,total=41,preserved_first_two_lineages=28,preserved_s2_m4=1,
              pretraining_worlds=12,pretraining_goal_runs=48,pretraining_rounds=144,confirm_trained_goal_runs=81,confirm_trained_rounds=486,g_prime_goal_runs=6,g_prime_rounds=36,all_confirm_curves=114,curve_points=798),
        pretraining_worlds=pretrain_worlds,probe=dict(prior_review_pass_preserved=True,M4_lawful=618,R_lawful=398,n_each=768,prior_auc_reproduction_not_repeated=True),
        budget=dict(ceiling_hours=42,previous_v61_seconds=24120,v62_gate_seconds=16124,precheck_seconds=137.15,
            pretrain_and_probe_controller_seconds=initial['elapsed_seconds'],first_interruption_charged_seconds=charged,rejected_first_interruption_field_seconds=9366,
            original_recovery_started_utc=origin.isoformat(),original_recovery_allowance_seconds=allowance,absolute_deadline_utc=deadline.isoformat(),
            confirmation_stopped_utc=stopped.isoformat(),recovery_window_wall_seconds_including_second_interruption_and_wait=window,
            second_recovery_monotonic_seconds=final['elapsed_seconds'],second_recovery_wall_seconds=(stopped-started).total_seconds(),
            wall_minus_monotonic_seconds=(stopped-started).total_seconds()-final['elapsed_seconds'],
            total_through_completion_seconds=total,total_through_completion_hours=total/3600,
            completion_margin_to_deadline_seconds=(deadline-stopped).total_seconds(),
            conservative_ceil_window_seconds=math.ceil(window),review_full_wall_seconds=review_seconds,
            total_with_ceil_window_and_full_review_seconds=base_cost+math.ceil(window)+math.ceil(review_seconds)),
        review_cost=dict(started_utc=review_started.isoformat(),metadata_program_seconds=time.monotonic()-start,git_calls=git_calls,
                        full_review_wall_seconds=review_seconds,gpu_work=0,inferences=0,weight_decodes=0,world_constructions=0,score_calls=0),
        verifier_history=[dict(source_commit='4c3d248',status='stopped_before_receipt',elapsed_tool_seconds=1.3045589,
             reason='Strict JSON constant reader rejected the frozen gate Infinity sentinel at summary.b_star_median.2[0].',
             resolution='Only that historically hash-bound gate constant is permitted and its exact location verified; every confirmation/pretraining curve remains strictly finite. No source, data, score, threshold or receipt was replaced.')],
        checks=['All 17 frozen scientific source bytes equal a189ea5 and committed HEAD; section 12.6 unchanged.',
                'Three complete pretraining lineages and 12 distinct checkpoint byte hashes match the earlier independently reviewed probe; no tensor decoded.',
                'Every complete arm JSON exactly equals its lineage; the nine-unit JSON exactly equals their ordered concatenation.',
                'All 28 completed s0/s1 artifacts and s2_8320_M4 are unchanged; both historical log prefixes and original controller records are intact.',
                'First interrupted M4 fragment and second interrupted Fa fragment are preserved, not additional independent replicates; only reviewed full-arm replay occurred.',
                'Fixed commands, first three admitted worlds, all nav log minima, persisted-store flags, tiers, round counters, G prime scope, and 64-env configuration checked.',
                'Original absolute deadline includes second disappearance and all investigation/idle time until successful closure; no allowance restart.',
                'Second launch has matching PID/hash/deadline handshake, clean close, empty stderr; process query empty and production score absent immediately before receipt.',
                'All source/evidence bytes rehashed after validation; this receipt expresses instrument validity only.'],
        limitations=['No scientific confirmation verdict, delta, significance, or figure is calculated here; the frozen scorer must run separately after this receipt is committed.',
                     'Training trajectories and deterministic CUDA replay are not independently rerun; confirmation retains arm-level JSON, not every action, optimizer state, or trained final network.',
                     'Pretraining tensor finiteness and probe AUC rely on the unchanged earlier independent receipt; this review hashes bytes and does not decode weights.',
                     'Navigation admission and random-seed provenance are checked from frozen source and available logs/commands, not by constructing fresh worlds or executing a nav gate.',
                     'Current process absence and Git chronology are available-record checks, not signed runtime attestation or forensic proof against arbitrary outside modification.',
                     'The first corrected interruption charge excludes idle time after the final boot under its reviewed rule; the entire second interruption is within the unchanged absolute window.',
                     'Reported wall time is conservative charged workstation time, not a measurement of GPU utilization; the small wall/monotonic discrepancy is reported without attributing a clock cause.',
                     'Area after b=0 may retain initial advantage; tiers3/4 b=0 includes weights plus within-world store. No test-world law-head AUC or pixel/Atari transfer follows.'],
        confirmation_validity_confirmed=True,scientific_verdict=None,score_absent_at_review=True,
        permitted_next_step='Commit this neutral review, recheck absent production score, then run only the frozen scorer and separately reviewed figure; no retraining or criterion change.')
    payload=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
    require(not score.exists(),'Score appeared before publication')
    with target.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(passed=True,receipt_sha256=sha(target),source_count=len(sources),evidence_count=len(evidence),
                          review_seconds=receipt['review_cost']['metadata_program_seconds'],budget=receipt['budget']),allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-commit',required=True)
    parser.add_argument('--review-started-utc',required=True)
    args=parser.parse_args()
    review(Path(__file__).resolve().parents[1],args.result_commit,utc(args.review_started_utc))
