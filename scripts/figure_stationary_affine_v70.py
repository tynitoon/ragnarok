"""Render the reviewed affine comparison from JSON, without arrays or inference."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figure_response_envelope_fit_v70 import sha, committed

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_affine_fit'
RESULT, REVIEW = FOLDER / 'fit_v0.json', FOLDER / 'fit_review_v0.json'
JOURNAL = FOLDER / 'fit_v0_progress.jsonl'
OUTPUT, METADATA = FOLDER / 'fit_v0.png', FOLDER / 'fit_v0_figure.json'
HELPER = ROOT / 'scripts/figure_response_envelope_fit_v70.py'
ARMS = ('newton_visual', 'newton_constant', 'adam_visual', 'adam_constant')


def main():
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError('Preserve the existing figure and receipt')
    result, review = (json.loads(path.read_text(encoding='utf-8')) for path in (RESULT, REVIEW))
    if (result['kind'] != 'stationary_affine_development_fit' or result['execution_status'] != 'complete'
            or result['completed_updates'] != dict(newton_visual=16, newton_constant=16, adam_visual=256, adam_constant=256)
            or not result['optimizer_update_count_exact'] or not result['operation_count_exact']
            or review.get('kind') != 'stationary_affine_independent_result_review' or review.get('passed') is not True
            or review.get('result_sha256') != sha(RESULT) or review.get('source_sha256') != result['source_sha256']
            or sha(JOURNAL) != result['progress_sha256']
            or sha(FOLDER / 'fit_v0_payload.json') != result['scientific_payload_sha256']):
        raise ValueError('Complete, exactly counted and independently reviewed affine fit required')
    for path in (RESULT, REVIEW, JOURNAL, FOLDER / 'fit_v0_payload.json', Path(__file__).resolve(), HELPER):
        committed(path)
    metrics = result['metrics']
    initial_bce = metrics['initial']['balanced_bce']
    losses = {arm: [initial_bce] for arm in ARMS}
    for line in JOURNAL.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event['event'] == 'update_complete':
            sequence = losses[event['arm']]
            if event['update'] != len(sequence):
                raise ValueError('Missing or reordered update in the loss history')
            sequence.append(event['loss_after'])
    if any(len(losses[arm]) != result['completed_updates'][arm] + 1 or not np.isfinite(losses[arm]).all() for arm in ARMS):
        raise ValueError('Four complete finite loss histories required')
    nonempty = [i for i, row in enumerate(metrics['initial']['cases']) if row['target_pixels']]
    empty = [i for i, row in enumerate(metrics['initial']['cases']) if not row['target_pixels']]
    if nonempty != [0, 1, 2, 3, 4, 5, 8, 9, 10, 11] or empty != [6, 7]:
        raise ValueError('Every one of the twelve cases must be represented')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.4))
    fig.subplots_adjust(left=.065, right=.98, bottom=.23, top=.79, hspace=.53, wspace=.25)
    fig.text(.065, .96, 'Apprendre une association visuelle avec sept paramètres', fontsize=20, weight='bold', color='#142b45')
    status = lambda recipe: 'critère atteint' if result['verdicts'][recipe]['passed'] else 'critère non atteint'
    fig.text(.065, .913, f'Newton (primaire) : {status("newton")}  ·  Adam (témoin) : {status("adam")}', fontsize=14, color='#40536a')
    fig.text(.065, .873, f'Même lot de 12 cas, coefficients initialement nuls et calcul float64 · {result["work_seconds"]:.1f} s CPU · aucun poids du juge', fontsize=11)
    fig.text(.065, .834, 'Les deux recettes ont des budgets différents ; leurs courbes ne comparent pas une efficacité à calcul égal.', fontsize=10, color='#40536a')
    for axis, recipe, total in zip(axes[0], ('newton', 'adam'), (16, 256), strict=True):
        for suffix, color, label in [('visual', '#147d87', 'RGB observés'), ('constant', '#d38b3b', 'Entrée constante')]:
            axis.plot(range(total + 1), losses[recipe + '_' + suffix], color=color, label=label, linewidth=1.7)
        axis.set(xlim=(0, total), ylim=(0, None), xlabel='Mises à jour de chaque bras', ylabel='BCE équilibrée')
        axis.set_title(f'{recipe.capitalize()} · initial puis erreur après chaque mise à jour', fontsize=11)
        axis.legend(frameon=False, fontsize=9)
        axis.grid(alpha=.18)
    overlap, specificity = axes[1]
    labels = ('Newton RGB', 'Newton constant', 'Adam RGB', 'Adam constant', 'Initial commun', 'Sortie zéro')
    colors = ('#147d87', '#b4dbd9', '#7c60a9', '#d2c8e3', '#92a0ae', '#dce1e5')
    all_arms = (*ARMS, 'initial', 'zero')
    for index, (arm, label, color) in enumerate(zip(all_arms, labels, colors, strict=True)):
        offset = (index - 2.5) * .13
        overlap.bar(np.arange(10) + offset, [metrics[arm]['cases'][i]['iou'] for i in nonempty], width=.125, color=color, label=label)
        specificity.bar(np.arange(2) + offset, [100 * metrics[arm]['cases'][i]['false_positive_fraction'] for i in empty], width=.125, color=color)
    overlap.set_xticks(range(10), [f'{i + 1}\n{metrics["initial"]["cases"][i]["target_pixels"]} px' for i in nonempty], fontsize=8)
    overlap.set(ylim=(0, 1.05), xlabel='Cas et taille de sa cible', ylabel='IoU au seuil fixé de 0,5')
    overlap.set_title('Les dix cibles non vides, sans sélection', fontsize=11)
    overlap.grid(axis='y', alpha=.18)
    specificity.axhline(.1, color='#9d552d', linestyle='--', linewidth=1, label='Plafond 0,1 %')
    specificity.set_xticks(range(2), ['Cas 7 · vide', 'Cas 8 · vide'])
    specificity.set_yscale('symlog', linthresh=.01)
    specificity.set(ylabel='Faux positifs (%)', ylim=(0, 150))
    specificity.set_title('Les deux vides restent dans le critère', fontsize=11)
    specificity.legend(frameon=False, fontsize=9)
    specificity.grid(axis='y', alpha=.18)
    handles, legend_labels = overlap.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .173), ncols=6, frameon=False, fontsize=9)
    for y, recipe in ((.152, 'newton'), (.122, 'adam')):
        measured, verdict = metrics[recipe + '_visual'], result['verdicts'][recipe]
        counts = '/'.join(str(measured['cases'][i]['false_positive_pixels']) for i in empty)
        fig.text(.065, y, f'{recipe.capitalize()} : IoU {measured["mean_nonempty_iou"]:.4f} (≥ 0,25) · avance {verdict["advantage_over_best_control"]:.4f} (≥ 0,10) · AP {measured["mean_nonempty_average_precision"]:.4f} · faux pixels des vides {counts} (≤ 33 chacun)', fontsize=10, color='#40536a')
    limits = ['Calibration sur un lot artificiel de développement : elle ne réhabilite pas le CNN ni son ancienne validation conditionnelle.',
              'Aucun jeu maîtrisé, généralisation indépendante, transfert ou apprentissage accéléré démontré par cet essai.']
    for y, line in zip((.075, .04), limits, strict=True):
        fig.text(.065, y, line, fontsize=10, color='#40536a')
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=150, facecolor='white')
        stream.flush()
        os.fsync(stream.fileno())
    plt.close(fig)
    with METADATA.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(kind='reviewed_stationary_affine_figure', result_sha256=sha(RESULT), review_sha256=sha(REVIEW),
             journal_sha256=sha(JOURNAL), figure_sha256=sha(OUTPUT),
             source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__).resolve(), HELPER)},
             arrays_opened=0, model_inferences=0, optimizer_updates=0, limits=limits), stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
