"""Describe the reviewed stationary fit using JSON only, with every case shown."""
import json
import os
from pathlib import Path

from scripts.figure_response_envelope_fit_v70 import sha, committed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_fit'
RESULT, REVIEW = FOLDER / 'fit_v0.json', FOLDER / 'fit_review_v1.json'
JOURNAL = FOLDER / 'fit_v0_progress.jsonl'
OUTPUT, METADATA = FOLDER / 'fit_v0.png', FOLDER / 'fit_v0_figure.json'
HELPER = ROOT / 'scripts/figure_response_envelope_fit_v70.py'


def main():
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError('Preserve the existing figure and its receipt')
    result, review = (json.loads(path.read_text(encoding='utf-8')) for path in (RESULT, REVIEW))
    if (result['kind'] != 'stationary_control_development_fit' or result['execution_status'] != 'complete'
            or result['completed_updates'] != {'visual': 256, 'constant': 256}
            or not result['optimizer_update_count_exact'] or not result['model_forward_count_exact']
            or review.get('kind') != 'stationary_fit_independent_result_review' or review.get('passed') is not True
            or review.get('result_sha256') != sha(RESULT) or review.get('source_sha256') != result['source_sha256']
            or sha(JOURNAL) != result['progress_sha256']
            or sha(FOLDER / 'fit_v0_payload.json') != result['scientific_payload_sha256']):
        raise ValueError('Complete, exactly counted and independently reviewed stationary fit required')
    for path in (RESULT, REVIEW, JOURNAL, FOLDER / 'fit_v0_payload.json', Path(__file__).resolve(), HELPER):
        committed(path)
    losses = {'visual': [], 'constant': []}
    for line in JOURNAL.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event['event'] == 'update_complete':
            sequence = losses[event['arm']]
            if event['update'] != len(sequence) + 1:
                raise ValueError('No missing or reordered update may be plotted')
            sequence.append(event['loss'])
    if any(len(values) != 256 or not np.isfinite(values).all() for values in losses.values()):
        raise ValueError('Two complete finite loss histories required')
    metrics = result['metrics']
    visual = metrics['visual']
    nonempty = [i for i, row in enumerate(visual['cases']) if row['target_pixels']]
    empty = [i for i, row in enumerate(visual['cases']) if not row['target_pixels']]
    if nonempty != [0, 1, 2, 3, 4, 5, 8, 9, 10, 11] or empty != [6, 7]:
        raise ValueError('All twelve declared cases must remain represented')
    best = max(metrics[arm]['mean_nonempty_iou'] for arm in ('constant', 'initial', 'zero'))
    fig = plt.figure(figsize=(14, 9.3))
    grid = fig.add_gridspec(2, 2, left=.07, right=.97, top=.78, bottom=.17, hspace=.6, wspace=.28,
                          height_ratios=[1.2, .9])
    curve, overlap = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    specificity, text_axis = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.text(.07, .95, 'Apprentissage du cas simple depuis des poids aléatoires', fontsize=19, weight='bold', color='#142b45')
    fig.text(.07, .9, 'Critère de faisabilité ' + ('atteint' if result['passed'] else 'non atteint'), fontsize=15,
             color='#16766d' if result['passed'] else '#9d552d')
    fig.text(.07, .85, f'12 cas de développement · 256 mises à jour par bras · {result["work_seconds"] / 60:.1f} min CPU · aucun poids du juge',
             fontsize=11, color='#40536a')
    for arm, color, label in [('visual', '#167f86', 'RGB observés'), ('constant', '#d79245', 'Entrée constante')]:
        curve.plot(np.arange(1, 257), losses[arm], color=color, label=label, linewidth=1.8)
    curve.set(xlim=(1, 256), ylim=(0, None), xlabel='Mise à jour de chaque bras', ylabel='BCE équilibrée')
    curve.set_title('Erreur avant chaque mise à jour', fontsize=11)
    curve.legend(frameon=False, fontsize=9)
    curve.grid(alpha=.18)
    colors = ('#167f86', '#d79245', '#91a0b2', '#d7dee5')
    labels = ('RGB appris', 'Constant appris', 'Initial', 'Sortie zéro')
    for arm, offset, color, label in zip(('visual', 'constant', 'initial', 'zero'), (-.27, -.09, .09, .27), colors, labels, strict=True):
        overlap.bar(np.arange(10) + offset, [metrics[arm]['cases'][i]['iou'] for i in nonempty],
                    width=.17, color=color, label=label)
    overlap.set_xticks(np.arange(10), [f'{i + 1}\n{visual["cases"][i]["target_pixels"]} px' for i in nonempty], fontsize=8)
    overlap.set(ylim=(0, 1.05), xlabel='Cas et taille de sa cible', ylabel='IoU au seuil fixé de 0,5')
    overlap.set_title('Les dix cas non vides, sans sélection', fontsize=11)
    overlap.legend(frameon=False, fontsize=8, ncols=2, loc='upper left')
    overlap.grid(axis='y', alpha=.18)
    for arm, offset, color, label in zip(('visual', 'constant', 'initial', 'zero'), (-.27, -.09, .09, .27), colors, labels, strict=True):
        specificity.bar(np.arange(2) + offset, [100 * metrics[arm]['cases'][i]['false_positive_fraction'] for i in empty],
                        width=.17, color=color, label=label)
    specificity.axhline(.1, color='#9d552d', linestyle='--', linewidth=1, label='Plafond 0,1 %')
    specificity.set_xticks(np.arange(2), ['Cas 7 · cible vide', 'Cas 8 · cible vide'])
    # An initial all-positive output can coexist with tiny final errors: show the exact counts alongside the panel.
    specificity.set(yscale='symlog', ylabel='Faux positifs (%)')
    specificity.set_yscale('symlog', linthresh=.01)
    specificity.set_title('Les deux vides restent dans le test', fontsize=11)
    specificity.grid(axis='y', alpha=.18)
    text_axis.axis('off')
    lines = [f'IoU moyen : {visual["mean_nonempty_iou"]:.4f}  (minimum 0,25)',
             f'Avance sur le meilleur témoin : {visual["mean_nonempty_iou"] - best:.4f}  (minimum 0,10)',
             f'AP moyenne : {visual["mean_nonempty_average_precision"]:.4f}',
             'Faux pixels des vides : ' + ' ; '.join(str(visual['cases'][i]['false_positive_pixels']) for i in empty) + '  (maximum 33 chacun)',
             f'BCE visuelle aux poids finaux : {visual["balanced_bce"]:.6f}',
             'Moyennes par paire, puis entre les cinq paires non vides.']
    for y, line in zip((1., .8, .6, .4, .2, 0.), lines, strict=True):
        text_axis.text(0, y, line, fontsize=10, color='#40536a', transform=text_axis.transAxes, va='top')
    limitations = ['Le certificat prouve une solution possible ; ces poids appris proviennent d’un autre processus, initialisé aléatoirement.',
                   'Un critère moyen peut être atteint en ne réussissant que certains cas. Les douze résultats sont tous représentés.',
                   'Aucun jeu maîtrisé, généralisation indépendante, transfert ou apprentissage accéléré démontré par ce fit.']
    for y, line in zip((.105, .07, .035), limitations, strict=True):
        fig.text(.07, y, line, fontsize=9, color='#40536a')
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=160, facecolor='white')
        stream.flush()
        os.fsync(stream.fileno())
    plt.close(fig)
    with METADATA.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(kind='reviewed_stationary_fit_figure', result_sha256=sha(RESULT), review_sha256=sha(REVIEW),
                       journal_sha256=sha(JOURNAL), figure_sha256=sha(OUTPUT),
                       source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in (Path(__file__).resolve(), HELPER)},
                       arrays_opened=0, model_inferences=0, optimizer_updates=0, limits=limitations), stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
