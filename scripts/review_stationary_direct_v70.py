"""Review CLOSED direct-path fit artifacts only; never construct or run a model.

Budget and numeric tolerances are fixed before reading the fit. Reuse the pinned
independent arithmetic reviewer, never the production model/loss/metric helpers.
Partial executions retain their status and inspected evidence; an instrument
PASS does not make an incomplete scientific execution complete.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE_REL = 'scripts/review_stationary_fit_v70.py'
BASE_SHA = '6666e25037cc3c7146eb531cb7d8e36551235c422696cd227cc851d305ee54fa'
require_hash = hashlib.sha256((ROOT / BASE_REL).read_bytes()).hexdigest()
if require_hash != BASE_SHA:
    raise RuntimeError('The independently reviewed arithmetic helper changed')
_spec = importlib.util.spec_from_file_location('_stationary_review_arithmetic', ROOT / BASE_REL)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
base.MAX_REVIEW_SECONDS = 120
require = base.require
PREFIX = 'craft_v6_out/v70_stationary_direct_fit/'
REVIEW_PROTOCOL = 'docs/ARC4_STATIONARY_DIRECT_REVIEW_PROTOCOL.json'
FIT_PROTOCOL = 'docs/ARC4_STATIONARY_DIRECT_PROTOCOL.json'
MANIFEST = 'docs/ARC4_STATIONARY_DIRECT_INITIAL_IDENTITY.json'
PREFLIGHT_SHA = '891f582ead06ef741bfbc72541cefdb0d151e8817055415853ad02cbc2eb4e83'
SOURCE_COMMIT, PREFLIGHT_COMMIT, LAUNCHER_COMMIT = '75278b2', '948148a', '520bd6f'
CONSTANT_NAMES = ('visual', 'constant')
NEW_SOURCES = {
    'ragnarok/learning/direct_response_envelope_v70.py', 'tests/test_direct_response_envelope_v70.py',
    'scripts/fit_stationary_direct_v70.py', 'tests/test_stationary_direct_fit_v70.py',
    'docs/ARC4_STATIONARY_DIRECT_PATH_DRAFT.md', FIT_PROTOCOL, MANIFEST,
}
REVIEW_SPEC = dict(version=0, max_review_seconds=120, cpu_threads=1, gpu_used=False,
    complete_metric_rows=120, complete_bce_comparisons=9, complete_evaluation_archives=12,
    complete_decoded_evaluation_members=18, complete_target_members=1, complete_checkpoint_decodes=2,
    metric_atol=1e-12, bce_atol=1e-6, bce_rtol=1e-6, sigmoid_atol=1e-7,
    total_threshold=.5, component_threshold=.5, component_metrics='descriptive only',
    partial_execution='audit available registered artifacts and accounting; retain INCOMPLETE; no scientific PASS',
    budget_scope='all verification through final evidence rehash; final receipt and console excluded',
    forbidden='model construction, model forward, backward, optimizer, simulation, new data and future-branch decoding',
    retry=False)


def expected_names():
    return {'initial_weights.pt', 'final_weights.pt'} | {
        arm + suffix for arm in ('initial', 'visual', 'constant')
        for suffix in ('_logits.npz', '_predictions.npz', '_components_logits.npz', '_components_predictions.npz')}


def check_accounting(result):
    counts, entered = result['completed_updates'], result['optimizer_step_entries']
    pending = result['pending_optimizer_arm']
    require(set(counts) == set(entered) == set(CONSTANT_NAMES), 'Unknown optimizer arm')
    require(pending is None or pending in CONSTANT_NAMES, 'Unknown pending optimizer arm')
    for arm in CONSTANT_NAMES:
        require(type(counts[arm]) is int and type(entered[arm]) is int and 0 <= counts[arm] <= entered[arm] <= 256,
                'Invalid optimizer count')
        require(entered[arm] - counts[arm] == int(pending == arm), 'Uncertain update lacks the exact pending arm')
    require(result['optimizer_updates'] == sum(counts.values())
            and result['optimizer_updates_upper_bound'] == sum(counts.values()) + int(pending is not None)
            and result['optimizer_update_count_exact'] is (pending is None), 'Optimizer lower/upper bound differs')
    fentry, freturn = result['model_forward_entries'], result['model_forward_returns']
    limits = dict(training=dict(visual=256, constant=256), evaluation=dict(visual=2, constant=1))
    require(set(fentry) == set(freturn) == set(limits), 'Unknown forward phase')
    for phase in limits:
        require(set(fentry[phase]) == set(freturn[phase]) == set(CONSTANT_NAMES), 'Unknown forward arm')
        for arm in CONSTANT_NAMES:
            a, b = fentry[phase][arm], freturn[phase][arm]
            require(type(a) is int and type(b) is int and 0 <= b <= a <= limits[phase][arm], 'Invalid forward count')
    require(sum(fentry[p][a]-freturn[p][a] for p in limits for a in CONSTANT_NAMES) <= 1,
            'More than one forward remained pending despite global STOP')
    require(result['model_forward_count_exact'] is (fentry == freturn), 'Forward exactness flag differs')
    for arm in CONSTANT_NAMES:
        require(counts[arm] <= freturn['training'][arm] <= counts[arm]+1, 'Training forward/update mismatch')
    if any(fentry['training'].values()):
        require(result.get('initial_identity_verified') is True, 'Training began before initial identity was verified')
    require(result['execution_status'] in ('complete', 'incomplete'), 'Unknown execution status')
    if result['execution_status'] == 'complete':
        require(counts == entered == dict(visual=256, constant=256) and pending is None
                and fentry == freturn == limits and not result['failures'], 'Complete execution has missing/uncertain calls')
    else:
        require(result['passed'] is False and result['verdict']['complete'] is False
                and result['verdict']['passed'] is False, 'Incomplete execution cannot publish scientific PASS')
    return dict(completed_updates=counts, pending_optimizer_arm=pending,
                optimizer_lower_bound=result['optimizer_updates'], optimizer_upper_bound=result['optimizer_updates_upper_bound'],
                model_forward_entries=fentry, model_forward_returns=freturn)


def check_components(total, contextual, direct, initial=False):
    np = base.np
    require(total.shape == contextual.shape == direct.shape and total.dtype == contextual.dtype == direct.dtype == np.float32,
            'Component dimensions or float32 dtype differ')
    require(np.array_equal(np.add(contextual, direct, dtype=np.float32), total), 'Saved contributions do not sum exactly')
    if initial:
        require(not np.count_nonzero(direct) and np.array_equal(contextual, total), 'Initial direct path is not zero')


def incomplete_journal(events, result, trailing_bytes):
    """Validate the available ordinary-event prefix; retain any incomplete tail."""
    expected = [('evaluation_started', 'visual', 'initial'), ('evaluation_complete', 'visual', 'initial')]
    for step in range(1, 257):
        for arm in CONSTANT_NAMES:
            expected.extend((event, arm, step) for event in ('update_started', 'optimizer_step_enter', 'optimizer_step_return', 'update_complete'))
    for arm in CONSTANT_NAMES:
        expected.extend((event, arm, 'final') for event in ('evaluation_started', 'evaluation_complete'))
    ordinary, terminal, last_time = 0, False, -1.
    terminal_events=[]
    returned = dict(visual=0, constant=0)
    for row in events:
        stamp = row['elapsed_seconds']
        require(math.isfinite(stamp) and last_time <= stamp <= result['work_seconds'], 'Nonmonotone or uncharged partial journal')
        last_time = stamp
        if row['event'] in ('failed', 'finished'):
            terminal = True
            terminal_events.append(row['event'])
            require(terminal_events in (['failed'],['finished'],['failed','finished']), 'Repeated or reversed terminal events')
            require(row['completed_updates'] == result['completed_updates']
                    and row['pending_optimizer_arm'] == result['pending_optimizer_arm'], 'Partial terminal counters differ')
            if row['event'] == 'finished':
                require(row['step_entries'] == result['optimizer_step_entries']
                        and row['forward_entries'] == result['model_forward_entries']
                        and row['forward_returns'] == result['model_forward_returns'], 'Partial final snapshot differs')
                require(result['failures'][:len(row['failures'])] == row['failures'], 'Journal failures were lost')
            continue
        require(not terminal and ordinary < len(expected), 'Work continued after a terminal event')
        kind, arm, ordinal = expected[ordinary]
        require(row['event'] == kind and row['arm'] == arm
                and row['checkpoint' if kind.startswith('evaluation_') else 'update'] == ordinal, 'Available journal is not the fixed schedule prefix')
        ordinary += 1
        if kind == 'optimizer_step_return':
            returned[arm] += 1
        if kind == 'update_complete':
            require(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm_before_clip']), 'Nonfinite journal scalar')
    require(all(returned[a] <= result['completed_updates'][a] for a in CONSTANT_NAMES), 'Journal reports excess returned updates')
    require(sum(result['completed_updates'][a]-returned[a] for a in CONSTANT_NAMES)<=1,
            'More than one update returned after the available journal prefix')
    return dict(mode='available prefix of incomplete execution', parsed_events=len(events),
        ordinary_prefix_events=ordinary, logged_optimizer_returns=returned, trailing_bytes=trailing_bytes,
        unlogged_returned_updates={a:result['completed_updates'][a]-returned[a] for a in CONSTANT_NAMES},
        limitation='A broken journal may omit returned operations; no optimizer or model trajectory is reconstructed')


class Review(base.Review):
    def __init__(self, result_sha, closed_commit):
        super().__init__(result_sha, closed_commit)
        self.metric_rows = self.bce_comparisons = 0

    def archive(self, name, description):
        path = self.evidence_file(PREFIX+name, description['sha256'])
        with base.np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(description['arrays']), 'Archive members differ: '+name)
            arrays = {}
            for key in archive.files:
                value = archive[key]
                self.decoded_arrays.append(dict(path=PREFIX+name,key=key))
                require(base.array_signature(value) == description['arrays'][key], 'Array signature differs: '+name+'/'+key)
                require(value.shape == (12,210,160) and value.dtype == base.np.float32, 'Expected saved float32 image-sized logits/probabilities')
                arrays[key] = value
        self.time_check()
        return arrays

    def measures(self, probability, logits, targets, cases, recorded, label):
        np = base.np
        require(np.isfinite(logits).all() and np.isfinite(probability).all()
                and np.all((probability>=0)&(probability<=1)), 'Nonfinite metric input: '+label)
        sigmoid = np.exp(-np.logaddexp(0.,-logits.astype(np.float64)))
        error = float(np.max(np.abs(sigmoid-probability)))
        require(error <= base.SIGMOID_ATOL, 'Saved sigmoid mismatch: '+label)
        values = base.metrics_independent(probability, targets, cases)
        base.same_metrics({k:v for k,v in recorded.items() if k!='balanced_bce'}, values, label)
        loss, per_case = base.bce_independent(logits,targets)
        saved = recorded['balanced_bce']
        require(math.isfinite(saved) and abs(loss-saved)<=base.BCE_ATOL+base.BCE_RTOL*abs(loss), 'BCE differs: '+label)
        self.metric_rows += 12
        self.bce_comparisons += 1
        self.time_check()
        return dict(metrics=values,bce_float64=loss,bce_recorded=saved,bce_abs_error=abs(loss-saved),
                    per_case_bce=per_case,sigmoid_max_abs_error=error)

    def inspect_weights(self, result, identity):
        torch,np=base.torch,base.np
        shapes={'direct_weight':[1,6,1,1],'direct_bias':[1]}
        shapes.update({'contextual.'+key:row['shape'] for key,row in identity['state_arrays'].items()})
        out={}
        def inspect(state, initial):
            require(set(state)==set(shapes),'Wrong augmented state population')
            finite=True;nonfinite={}
            for key,shape in shapes.items():
                value=state[key]
                require(isinstance(value,torch.Tensor) and list(value.shape)==shape and value.dtype==torch.float32
                        and value.device.type=='cpu','Wrong tensor shape/dtype/device')
                raw=value.detach().numpy()
                n=int(np.count_nonzero(~np.isfinite(raw)))
                if n: nonfinite[key]=n;finite=False
                if initial:
                    if key.startswith('contextual.'):
                        require(base.array_signature(raw)==identity['state_arrays'][key[len('contextual.'):]],'Historical contextual tensor identity differs')
                    else: require(not np.count_nonzero(raw),'Nonzero initial direct path')
            require(sum(v.numel() for v in state.values())==10184,'Wrong parameter population')
            require(finite or result['execution_status']=='incomplete','Nonfinite final weights in complete fit')
            return dict(parameters=10184,state_tensors=14,all_finite=finite,nonfinite_elements=nonfinite)
        if 'initial_weights.pt' in result['artifacts']:
            out['initial']=inspect(self.state(PREFIX+'initial_weights.pt'),True)
        if 'final_weights.pt' in result['artifacts']:
            states=self.state(PREFIX+'final_weights.pt')
            require(set(states)<=set(CONSTANT_NAMES) and bool(states),'Unknown final arm')
            if result['execution_status']=='complete': require(set(states)==set(CONSTANT_NAMES),'Missing final arm')
            out['final']={arm:inspect(state,False) for arm,state in states.items()}
        return out

    def perform(self):
        runtime=base.configure_review_runtime()
        base.torch.Tensor.backward=base.forbidden
        base.torch.autograd.grad=base.forbidden
        result=self.json_file(PREFIX+'fit_v0.json',self.result_sha)
        complete=result['execution_status']=='complete'
        require(result['kind']=='stationary_direct_path_development_fit','Wrong direct result kind')
        self.committed({PREFIX+'fit_v0.json':self.result_sha},self.closed_commit)
        review_spec=self.json_file(REVIEW_PROTOCOL)
        require(review_spec==REVIEW_SPEC,'Prospective review protocol differs')
        preflight=self.json_file(PREFIX+'preflight_v0.json',PREFLIGHT_SHA)
        require(preflight['kind']=='stationary_direct_independent_preflight' and preflight['passed'] is True,'Wrong preflight')
        for key in ('source_sha256','evidence_sha256','runtime'):
            require(result[key]==preflight[key],'Exact preflight map differs: '+key)
        require(runtime==result['runtime'],'Numerical runtime differs')
        self.sources=result['source_sha256']
        require(len(self.sources)==119 and len(result['evidence_sha256'])==8 and NEW_SOURCES<=set(self.sources),'Wrong source/proof closure')
        for rel,digest in self.sources.items(): require(base.sha_file(self.safe_path(rel))==digest,'Source SHA differs: '+rel)
        self.committed(self.sources,SOURCE_COMMIT)
        for rel,digest in result['evidence_sha256'].items(): self.evidence_file(rel,digest)
        inherited=self.json_file('craft_v6_out/v70_stationary_affine_fit/fit_v0.json')
        require(set(self.sources)==set(inherited['source_sha256'])|NEW_SOURCES
                and all(self.sources[k]==v for k,v in inherited['source_sha256'].items()),'Inherited 112 sources differ')
        protocol=self.json_file(FIT_PROTOCOL)
        require(protocol==result['protocol'] and protocol['parameters']==10184 and protocol['updates_per_arm']==256
                and protocol['max_work_seconds']==1200 and protocol['max_binary_output_bytes']==64000000,'Fixed fitting protocol differs')
        self.checks['accounting']=check_accounting(result)
        require(result['new_synthetic_steps']==result['new_initial_screens']==result['new_emulated_frames']==0
                and result['inherited_synthetic_steps']==216 and result['gpu_used'] is False,'Unexpected new work')
        attempt=self.json_file(PREFIX+'fit_v0_attempt.json',result['attempt_sha256'])
        for key in ('protocol','source_sha256','evidence_sha256','runtime','preflight_sha256'):
            require(attempt[key]==result[key],'Attempt differs: '+key)
        require(attempt['kind']=='stationary_direct_fit_attempt' and result['preflight_sha256']==PREFLIGHT_SHA,'Attempt/preflight identity differs')
        if result['scientific_payload_sha256'] is not None:
            payload=self.json_file(PREFIX+'fit_v0_payload.json',result['scientific_payload_sha256'])
            require(payload['kind']=='stationary_direct_fit_scientific_payload','Wrong payload kind')
            for key,value in payload.items():
                if key not in ('kind','claim','failures'): require(result[key]==value,'Final receipt changed payload: '+key)
            require(result['failures'][:len(payload['failures'])]==payload['failures'],'Payload failures lost')
        else: require(not complete,'Missing payload cannot be complete')
        if complete:
            require(result['candidate_execution_complete'] and result['final_provenance_verified']
                    and result['initial_identity_verified'] and 0<result['work_seconds']<1200,'Incomplete final closure or identity')
        else: require(math.isfinite(result['work_seconds']) and result['work_seconds']>=0,'Invalid incomplete duration')
        journal=self.safe_path(PREFIX+'fit_v0_progress.jsonl')
        events=[];trailing=0
        if journal.exists():
            self.evidence_file(PREFIX+journal.name,result['progress_sha256'])
            data=journal.read_bytes()
            pieces=data.splitlines(keepends=True)
            for i,line in enumerate(pieces):
                try:
                    require(line.endswith(b'\n'),'Unclosed journal line')
                    events.append(json.loads(line))
                except (ValueError,json.JSONDecodeError):
                    require(not complete and i==len(pieces)-1,'Invalid nonterminal or complete journal line')
                    trailing=len(line)
        self.checks['journal']=base.validate_journal(events,result) if complete else incomplete_journal(events,result,trailing)
        artifacts=result['artifacts']
        require(set(artifacts)<=expected_names(),'Unknown binary artifact')
        if result['initial_identity_verified']:
            require({'initial_weights.pt','initial_logits.npz','initial_predictions.npz',
                     'initial_components_logits.npz','initial_components_predictions.npz'}<=set(artifacts),
                    'Verified initial identity lacks its preserved state or outputs')
        if complete: require(set(artifacts)==expected_names(),'Missing complete binary artifact')
        for name,row in artifacts.items():
            path=self.evidence_file(PREFIX+name,row['sha256'])
            require(path.stat().st_size==row['bytes'],'Artifact size differs')
        binary_paths={name:self.safe_path(PREFIX+name) for name in expected_names() if self.safe_path(PREFIX+name).exists()}
        binary_bytes=sum(path.stat().st_size for path in binary_paths.values())
        require(binary_bytes<=64000000,'Binary storage cap exceeded')
        self.checks['unregistered_partial_artifacts']={}
        for name,path in binary_paths.items():
            if name not in artifacts:
                self.evidence_file(PREFIX+name)
                self.checks['unregistered_partial_artifacts'][name]=dict(bytes=path.stat().st_size,sha256=self.evidence[PREFIX+name])
        self.checks['binary_bytes']=binary_bytes
        identity=self.json_file(MANIFEST)
        control=self.json_file('craft_v6_out/v70_stationary_control/control_v0.json')
        cases=control['cases']
        require(result['cases']==cases and [(c['game'],c['anchor_index']) for c in cases]
                == [('synthetic_pair_'+str(i//2),i%2) for i in range(12)],'Case identity/order differs')
        data_rel='craft_v6_out/v70_stationary_control/control_v0_dataset.npz'
        with base.np.load(self.safe_path(data_rel),allow_pickle=False) as archive:
            require(set(archive.files)=={'past_pairs','observed_response_envelopes','noop_futures','action_futures'},'Control archive population differs')
            targets=archive['observed_response_envelopes']
        self.decoded_arrays.append(dict(path=data_rel,key='observed_response_envelopes'))
        require(targets.dtype==bool and targets.shape==(12,210,160) and targets.sum(axis=(1,2)).tolist()==base.COUNTS
                and base.array_signature(targets)==control['artifacts']['control_v0_dataset.npz']['arrays']['observed_response_envelopes'],'Target signature/count differs')
        self.checks['metrics']={};self.checks['components']={};self.checks['saved_array_conditions']={}
        for arm in ('initial','visual','constant'):
            arrays={}
            for suffix in ('_logits.npz','_predictions.npz','_components_logits.npz','_components_predictions.npz'):
                name=arm+suffix
                if name in artifacts: arrays.update(self.archive(name,artifacts[name]))
            keys=set(arrays)
            expected={'logits','probability','contextual_logits','direct_logits','contextual_probability','direct_probability'}
            if complete: require(keys==expected,'Missing complete evaluation members')
            finite={key:bool(base.np.isfinite(value).all()) for key,value in arrays.items()}
            self.checks['saved_array_conditions'][arm]=dict(members=list(arrays),finite=finite)
            if {'logits','contextual_logits','direct_logits'}<=keys and all(finite[k] for k in ('logits','contextual_logits','direct_logits')):
                check_components(arrays['logits'],arrays['contextual_logits'],arrays['direct_logits'],arm=='initial')
            if arm in result['metrics']:
                require({'logits','probability'}<=keys,'Metrics without retained logits/probabilities')
                self.checks['metrics'][arm]=self.measures(arrays['probability'],arrays['logits'],targets,cases,result['metrics'][arm],arm)
            if arm in result['component_metrics']:
                require(set(result['component_metrics'][arm])=={'contextual','direct'},'Component metric population differs')
                self.checks['components'][arm]={}
                for part in ('contextual','direct'):
                    require({part+'_logits',part+'_probability'}<=keys,'Component measures without retained arrays')
                    self.checks['components'][arm][part]=self.measures(arrays[part+'_probability'],arrays[part+'_logits'],targets,cases,
                        result['component_metrics'][arm][part],arm+'/'+part)
            if arm=='initial' and result['initial_identity_verified']:
                require(artifacts['initial_logits.npz']['arrays']['logits']==identity['initial_logits']
                        and artifacts['initial_predictions.npz']['arrays']['probability']==identity['initial_predictions']
                        and result['metrics']['initial']==identity['initial_metrics'],'Archived initial output/metric identity differs')
            if arm=='constant' and complete:
                require(all(base.np.all(v==v.flat[0]) for v in arrays.values()),'Constant-input outputs are not spatially constant')
        if 'zero' in result['metrics']:
            zero=base.metrics_independent(base.np.zeros(targets.shape,dtype=base.np.float32),targets,cases)
            base.same_metrics(result['metrics']['zero'],zero,'zero');self.metric_rows+=12
            self.checks['zero_metrics']=zero
        if complete:
            require(set(result['metrics'])=={'initial','zero','visual','constant'}
                    and set(result['component_metrics'])=={'initial','visual','constant'},'Complete metric population differs')
            recomputed={arm:row['metrics'] for arm,row in self.checks['metrics'].items()}
            recomputed['zero']=self.checks['zero_metrics']
            verdict=base.verdict_independent(recomputed)
            base.same_metrics(result['verdict'],verdict,'sum verdict')
            base.same_metrics(result['candidate_verdict'],verdict,'candidate sum verdict')
            require(result['passed'] is verdict['passed'] and self.metric_rows==120 and self.bce_comparisons==9,'Whole-sum gate or evaluation counts differ')
        self.checks['weights']=self.inspect_weights(result,identity)
        self.checks['launch']=self.launch(result,attempt)
        self.checks['execution_status']=result['execution_status']
        self.checks['scientific_passed']=result['passed']
        self.committed({rel:digest for rel,digest in self.evidence.items() if rel.startswith(PREFIX)},self.closed_commit)
        for rel in (BASE_REL,'scripts/review_stationary_direct_v70.py','tests/test_review_stationary_direct_v70.py'):
            self.evidence_file(rel,BASE_SHA if rel==BASE_REL else None)
        self.committed(self.sources)
        self.committed(self.evidence)
        for rel,digest in {**self.sources,**self.evidence}.items():
            self.time_check();require(base.sha_file(self.safe_path(rel))==digest,'Evidence changed during review: '+rel)
        self.time_check()

    def launch(self,result,attempt):
        prefix=PREFIX+'launch_v0/'
        intent=self.json_file(prefix+'intent.json');dispatch=self.json_file(prefix+'dispatch.json')
        launcher=self.evidence_file(prefix+'launch.py',intent['launcher_sha256'])
        console_path=self.evidence_file(prefix+'console.log')
        lines=console_path.read_text(encoding='utf-8-sig').splitlines()
        require(len(lines)>=2,'Missing closed launch console')
        first=json.loads(lines[0]);last=json.loads(lines[-1])
        require(intent['kind']=='stationary_direct_fit_wmi_launch_intent' and dispatch['kind']=='stationary_direct_fit_wmi_dispatch'
                and dispatch['return_value']==0 and dispatch['intent_sha256']==self.evidence[prefix+'intent.json']
                and dispatch['preflight_sha256']==PREFLIGHT_SHA and first['pid']==dispatch['pid']
                and first['detached_from_job'] is True and first['mode']=='fixed_stationary_direct_path_fit','WMI launch identity differs')
        require(intent['preflight_sha256']==PREFLIGHT_SHA and intent['show_window']==dispatch['show_window']==0
                and intent['gpu_allowed'] is False and intent['maximum_scientific_work_seconds']==1200
                and intent['updates_per_arm']==256 and intent['new_attempt_only'] is True,'Launch budget or visibility differs')
        require(Path(intent['cwd']).resolve()==ROOT.resolve()
                and intent['command_line'].replace('\\','/')=='"C:/Python314/python.exe" -u "'+launcher.as_posix()+'"',
                'WMI command or working directory differs')
        for field,commit in (('source_commit',SOURCE_COMMIT),('independent_preflight_commit',PREFLIGHT_COMMIT),('launcher_commit',LAUNCHER_COMMIT)):
            require(self.git('rev-parse',intent[field])==self.git('rev-parse',commit),'Launch commit differs')
        for field in ('execution_status','verdict','completed_updates','optimizer_update_count_exact','work_seconds'):
            require(last[field]==result[field],'Console differs from result')
        if result['execution_status']=='complete': require(len(lines)==2,'Unexpected complete-fit console output')
        self.committed({prefix+'launch.py':intent['launcher_sha256']},LAUNCHER_COMMIT)
        self.committed({prefix+name:self.evidence[prefix+name] for name in ('intent.json','dispatch.json','console.log')},self.closed_commit)
        self.committed({PREFIX+'preflight_v0.json':PREFLIGHT_SHA},PREFLIGHT_COMMIT)
        for older,newer in ((SOURCE_COMMIT,PREFLIGHT_COMMIT),(PREFLIGHT_COMMIT,LAUNCHER_COMMIT),(LAUNCHER_COMMIT,self.closed_commit)):
            self.git('merge-base','--is-ancestor',older,newer)
        began=datetime.fromisoformat(attempt['created_utc'])
        intended=datetime.fromisoformat(intent['created_utc'].replace('Z','+00:00'))
        closed=datetime.fromisoformat(self.git('show','-s','--format=%cI',self.closed_commit))
        pref_date=datetime.fromisoformat(self.git('show','-s','--format=%cI',PREFLIGHT_COMMIT))
        source_date=datetime.fromisoformat(self.git('show','-s','--format=%cI',SOURCE_COMMIT))
        require(source_date<=pref_date<=intended<=began
                and closed+timedelta(seconds=1)>=began+timedelta(seconds=result['work_seconds']),'Launch/closure chronology differs')
        return dict(pid=first['pid'],detached_from_job=True,closed_commit=self.closed_commit,
                    console_matches_result=True,live_process_queries=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-closed-result',action='store_true',required=True)
    parser.add_argument('--result-sha256',required=True)
    parser.add_argument('--closed-commit',required=True)
    args=parser.parse_args()
    require(len(args.result_sha256)==64 and all(c in '0123456789abcdef' for c in args.result_sha256),'Explicit closed result SHA required')
    output=ROOT/(PREFIX+'fit_review_v0.json');attempt=ROOT/(PREFIX+'fit_review_v0_attempt.json')
    require(not output.exists() and not attempt.exists(),'Preserve prior review traces; no automatic retry')
    review=Review(args.result_sha256,args.closed_commit)
    base.write_exclusive(attempt,dict(kind='stationary_direct_independent_review_attempt',created_utc=datetime.now(timezone.utc).isoformat(),
        result_sha256=args.result_sha256,closed_commit=args.closed_commit,protocol=REVIEW_SPEC,
        reviewer_sha256=base.sha_file(Path(__file__)),arithmetic_helper_sha256=BASE_SHA))
    attempt_sha=base.sha_file(attempt)
    failures=[]
    try: review.perform()
    except BaseException as error: failures.append(dict(type=type(error).__name__,message=str(error)))
    elapsed=time.monotonic()-review.started
    if elapsed>=120: failures.append(dict(type='TimeoutError',message='Fixed review ceiling exceeded'))
    receipt=dict(kind='stationary_direct_fit_independent_result_review',passed=not failures,
        reviewed_utc=datetime.now(timezone.utc).isoformat(),result_sha256=args.result_sha256,
        source_sha256=review.sources,evidence_sha256=review.evidence,review_attempt_sha256=attempt_sha,
        checks=review.checks,failures=failures,decoded_arrays=review.decoded_arrays,decoded_checkpoints=review.checkpoints,
        costs=dict(review_seconds=elapsed,main_process_cpu_seconds=time.process_time()-review.cpu_started,max_seconds=120,cpu_threads=1,
            metric_rows_recomputed=review.metric_rows,bce_comparisons=review.bce_comparisons,
            new_model_constructions=0,new_model_forwards=0,gradients=0,optimizer_updates=0,
            new_synthetic_steps=0,new_emulated_frames=0,gpu_used=False),
        limits=[
            'Instrument validity is distinct from scientific PASS; a reviewed partial execution stays INCOMPLETE.',
            'Only summed-output metrics determine the gate; component removals are descriptive and cannot identify the cause of learning.',
            'A direct path can carry a PASS without improved contextual representations; clipping, joint gradients and compensating biases differ from the prior CNN.',
            'Only saved targets, returned logits/probabilities and current checkpoints are decoded; no past/future pixels, old checkpoint, network or optimizer trajectory is replayed.',
            'The same twelve observed development cases establish no independent generalization, palette invariance, temporal skill, Atari perception, transfer or acceleration.',
            'Historical runtime comparisons cannot establish a causal speed improvement; only charged closure and the frozen limits are checked.',
            'The conservative wall ceiling bounds single-thread verification; main-process CPU excludes Git subprocesses. Final receipt/console follow verification.',
        ])
    base.write_exclusive(output,receipt)
    print(json.dumps(dict(passed=receipt['passed'],failures=failures,review_seconds=elapsed,
        metric_rows=review.metric_rows,bce_comparisons=review.bce_comparisons,receipt_sha256=base.sha_file(output)),allow_nan=False))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
