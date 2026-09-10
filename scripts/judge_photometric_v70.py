"""One judge extraction and primary score from sealed pixel proposals; no network."""
from dataclasses import asdict

from scripts.photometric_common_v70 import (
    OUT, check_time, elapsed, read_json, sha, utc, verify_sources, write_json,
)


def main():
    intent = read_json(OUT / 'judge_v0_attempt.json')
    verify_sources(intent['source_sha256'])
    if sha(OUT / 'fit_v0.json') != intent['fit_result_sha256']:
        raise ValueError('Fit seal changed')
    check_time(intent)
    import numpy as np
    from ragnarok.evaluation.selective_correspondence_score_v70 import (
        STRATA, VISIBLE, WITNESSES, SceneVerdict, WitnessVerdict, score,
    )
    representations = ('final', 'initial', 'rgb_reference')
    fit_result = read_json(OUT / 'fit_v0.json')

    def sealed_json(name):
        if sha(OUT / name) != fit_result['artifacts'][name]['sha256']:
            raise ValueError('Judge artifact changed')
        return read_json(OUT / name)

    def proposals(name, shape):
        if sha(OUT / name) != fit_result['artifacts'][name]['sha256']:
            raise ValueError('Proposal archive changed')
        with np.load(OUT / name, allow_pickle=False) as archive:
            data = {k: archive[k].copy() for k in archive.files}
        kinds = dict(destination=np.int64, admitted=np.bool_, best_index=np.int64,
                     round_trip_score=np.float32, endpoint_similarity=np.float32)
        if set(data) != set(kinds) or any(data[k].shape != shape or data[k].dtype != dtype
                                        for k, dtype in kinds.items()):
            raise ValueError('Proposal schema differs')
        best, dest, admitted = data['best_index'], data['destination'], data['admitted']
        round_trip, similarity = data['round_trip_score'], data['endpoint_similarity']
        if (not all(np.isfinite(a).all() for a in data.values()) or best.min() < 0
                or best.max() >= shape[1]*shape[2] or round_trip.min() < 0
                or round_trip.max() > 1.00001 or similarity.min() < -1.00001
                or similarity.max() > 1.00001
                or not np.array_equal(admitted, (round_trip >= .90) & (similarity >= .95))
                or not np.array_equal(dest, np.where(admitted, best, -1))):
            raise ValueError('Invalid recorded association or changed admission rule')
        return {k: a.reshape(shape[0], -1) for k, a in data.items()}

    arrays = {name: proposals(name+'_eval.npz', (1280, 8, 8)) for name in representations}
    raw_controls = {name: proposals(name+'_controls_raw.npz', (4, 10, 12)) for name in representations}
    pooled = {name: sealed_json(name+'_controls_pooled.json') for name in representations}
    for records in pooled.values():
        if len(records) != 4:
            raise ValueError('Four pooled structural guards required')
        for row in records:
            dest, admitted = np.asarray(row['destination']), np.asarray(row['admitted'])
            if (dest.shape != (10, 12) or admitted.shape != (10, 12)
                    or (dest != -1).any() or admitted.any() or admitted.dtype != np.bool_
                    or row['computed'] is not False
                    or row['reason'] != 'temporal_support_not_point_observation'
                    or any(row[k] is not None for k in ('best_index', 'round_trip_score', 'endpoint_similarity'))):
                raise ValueError('Invalid structural guard archive')
    # These are the first reads of physical targets outside the generation process.
    judges = sealed_json('eval_judge.json')
    control_judges = sealed_json('controls_judge.json')
    if (len(judges) != 1280 or len(control_judges) != 4
            or [row['stratum'] for row in judges] != [s for s in STRATA for _ in range(256)]):
        raise ValueError('The declared scene table is incomplete or reordered')
    expected_controls = [dict(family=family, branch_index=branch, source_pixel=38, target_pixel=None)
                         for family in ('changed_occluder', 'hidden_subject') for branch in (0, 1)]
    if (control_judges != expected_controls
            or any(type(row['branch_index']) is not int or type(row['source_pixel']) is not int
                   for row in control_judges)):
        raise ValueError('Exactly the four ordered family/branch controls and source (3,2) are required')
    rows, details = [], []
    rank_tables = {s: dict(both_correct=0, initial_only=0, final_only=0, neither_correct=0,
                           rgb_reference_correct=0) for s in VISIBLE}
    descriptive = {rep: {s: dict(scenes=0, admitted=0, correct_admitted=0) for s in STRATA}
                   for rep in representations}
    coincidences = {s: dict(scenes_with_visible_palette_collision=0,
                            scenes_with_visible_quantization_only_collision=0) for s in STRATA}

    def query(data, index, pixel):
        return {key: value[index, pixel].item() for key, value in data.items()}

    for index, judge in enumerate(judges):
        check_time(intent)
        stratum, pixel, target = judge['stratum'], judge['source_pixel'], judge['target_pixel']
        if type(pixel) is not int or not 0 <= pixel < 64:
            raise ValueError('Invalid declared query')
        if stratum in VISIBLE:
            if type(target) is not int or not 0 <= target < 64:
                raise ValueError('Visible query requires a target')
        elif target is not None:
            raise ValueError('Unattributable query must not have an observable target')
        answers = {rep: query(arrays[rep], index, pixel) for rep in representations}
        correct = {rep: bool(target is not None and answer['admitted'] and answer['destination'] == target)
                   for rep, answer in answers.items()}
        rows.append(SceneVerdict(judge['scene_id'], stratum, answers['final']['admitted'], correct['final'],
                                 answers['initial']['admitted'], correct['initial']))
        details.append(dict(scene_id=judge['scene_id'], stratum=stratum, source_pixel=pixel,
                            target_pixel=target, answers=answers, correct_admitted=correct))
        for rep, answer in answers.items():
            item = descriptive[rep][stratum]
            item['scenes'] += 1
            item['admitted'] += int(answer['admitted'])
            item['correct_admitted'] += int(correct[rep])
        if stratum in VISIBLE:
            before = answers['initial']['best_index'] == target
            after = answers['final']['best_index'] == target
            key = 'both_correct' if before and after else 'initial_only' if before else 'final_only' if after else 'neither_correct'
            rank_tables[stratum][key] += 1
            rank_tables[stratum]['rgb_reference_correct'] += int(answers['rgb_reference']['best_index'] == target)
        visible_collisions = [other for t, others in enumerate(judge['query_palette_coincidences'])
                              if judge['query_visible'][t] for other in others]
        coincidences[stratum]['scenes_with_visible_palette_collision'] += int(bool(visible_collisions))
        coincidences[stratum]['scenes_with_visible_quantization_only_collision'] += int(any(
            judge['base_colors'][other] != judge['base_colors'][1] for other in visible_collisions))
    branch_rows, witness_rows = {}, {}
    for rep in representations:
        branches = []
        for index, control in enumerate(control_judges):
            for kind in ('raw', 'pooled'):
                if kind == 'raw':
                    answer = query(raw_controls[rep], index, control['source_pixel'])
                else:
                    answer = dict(admitted=False, destination=None, best_index=None,
                                  round_trip_score=None, endpoint_similarity=None,
                                  reason='temporal_support_not_point_observation')
                branches.append(dict(witness_id=control['family']+'_'+kind,
                                     branch_index=control['branch_index'], image_kind=kind,
                                     source_pixel=control['source_pixel'], answer=answer,
                                     structural_guard=(kind == 'pooled')))
        branch_rows[rep] = branches
        witness_rows[rep] = []
        for witness in WITNESSES:
            matching = [row for row in branches if row['witness_id'] == witness]
            if len(matching) != 2 or {row['branch_index'] for row in matching} != {0, 1}:
                raise ValueError('A witness needs both distinct branch responses; missing is not abstention')
            witness_rows[rep].append(WitnessVerdict(witness, any(row['answer']['admitted'] for row in matching)))
    # Durable complete score inputs precede the unique score invocation.
    write_json(OUT / 'score_inputs_v0.json', dict(rows=[asdict(r) for r in rows],
        witnesses=[asdict(r) for r in witness_rows['final']]))
    write_json(OUT / 'judge_details_v0.json', dict(
        scene_answers=details, control_branch_answers=branch_rows,
        control_reductions={rep: [asdict(r) for r in rows_] for rep, rows_ in witness_rows.items()},
        descriptive_counts=descriptive, best_index_paired_tables=rank_tables,
        coincidence_counts=coincidences,
        query_population='one initially visible actor-0 pixel per declared IID scene; not dense-pixel risk',
        pooled_controls='structural temporal guards, no credit for learned abstention'))
    check_time(intent)
    result = score(rows, witness_rows['final'])  # The only score call.
    result.update(completed_utc=utc(), elapsed_seconds=elapsed(intent),
                  score_inputs_sha256=sha(OUT / 'score_inputs_v0.json'),
                  judge_details_sha256=sha(OUT / 'judge_details_v0.json'),
                  fit_result_sha256=intent['fit_result_sha256'], score_calls=1,
                  new_rgb_frames=0, model_calls=0, optimizer_updates=0)
    write_json(OUT / 'score_v0.json', result)
    print(result['verdict'])


if __name__ == '__main__':
    main()
