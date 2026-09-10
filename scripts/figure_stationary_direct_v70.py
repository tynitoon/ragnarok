"""Plot the reviewed direct-path fit and all saved component metrics, JSON only."""

import json
import os
from pathlib import Path

from scripts.figure_response_envelope_fit_v70 import committed, sha
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_direct_fit'
RESULT, REVIEW = FOLDER / 'fit_v0.json', FOLDER / 'fit_review_v0.json'
JOURNAL, PAYLOAD = FOLDER / 'fit_v0_progress.jsonl', FOLDER / 'fit_v0_payload.json'
OUTPUT, METADATA = FOLDER / 'fit_v0.png', FOLDER / 'fit_v0_figure.json'
HELPER = ROOT / 'scripts/figure_response_envelope_fit_v70.py'
COLORS = ('#167f86', '#d79245', '#91a0b2', '#d7dee5')


def case_bars(ax, series, indices, empty=False):
    width = .75 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for (label, row, color), offset in zip(series, offsets, strict=True):
        values = [100 * row['cases'][i]['false_positive_fraction'] if empty
                  else row['cases'][i]['iou'] for i in indices]
        ax.bar(np.arange(len(indices)) + offset, values, width=width * .94, label=label, color=color)
    if empty:
        ax.axhline(.1, color='#9d552d', ls='--', lw=1, label='Plafond 0,1 %')
        ax.set_xticks(np.arange(2), ['Cas 7', 'Cas 8'])
        ax.set_yscale('symlog', linthresh=.01)
        ax.set_ylabel('Faux positifs (%)')
    else:
        ax.set_xticks(np.arange(len(indices)),
                      [f'{i + 1}\n{series[0][1]["cases"][i]["target_pixels"]} px' for i in indices], fontsize=8)
        ax.set(ylim=(0, 1.05), ylabel='IoU au seuil fixé de 0,5', xlabel='Cas et taille de sa cible')
    ax.legend(frameon=False, fontsize=8, ncols=2, loc='upper left', bbox_to_anchor=(0, -.36))
    ax.grid(axis='y', alpha=.18)


def main():
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError('Preserve the existing figure and its receipt')
    result, review = (json.loads(p.read_text(encoding='utf-8')) for p in (RESULT, REVIEW))
    expected = dict(training=dict(visual=256, constant=256), evaluation=dict(visual=2, constant=1))
    if (result['kind'] != 'stationary_direct_path_development_fit'
            or result['execution_status'] != 'complete' or not result['initial_identity_verified']
            or result['completed_updates'] != dict(visual=256, constant=256)
            or not result['optimizer_update_count_exact'] or not result['model_forward_count_exact']
            or result['model_forward_entries'] != expected or result['model_forward_returns'] != expected
            or review.get('kind') != 'stationary_direct_fit_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != sha(RESULT)
            or review.get('source_sha256') != result['source_sha256']
            or sha(JOURNAL) != result['progress_sha256']
            or sha(PAYLOAD) != result['scientific_payload_sha256']):
        raise ValueError('Complete, exactly counted and independently reviewed fit required')
    for p in (RESULT, REVIEW, JOURNAL, PAYLOAD, Path(__file__).resolve(), HELPER):
        committed(p)
    losses = dict(visual=[], constant=[])
    for line in JOURNAL.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event['event'] == 'update_complete':
            values = losses[event['arm']]
            if event['update'] != len(values) + 1:
                raise ValueError('Missing or reordered loss history')
            values.append(event['loss'])
    if any(len(v) != 256 or not np.isfinite(v).all() for v in losses.values()):
        raise ValueError('Two complete, finite loss histories required')
    m = result['metrics']
    parts = result['component_metrics']['visual']
    nonempty = [i for i, r in enumerate(m['visual']['cases']) if r['target_pixels']]
    empty = [i for i, r in enumerate(m['visual']['cases']) if not r['target_pixels']]
    if nonempty != [0, 1, 2, 3, 4, 5, 8, 9, 10, 11] or empty != [6, 7]:
        raise ValueError('All twelve declared cases must remain represented')
    fig = plt.figure(figsize=(18, 11.4))
    grid = fig.add_gridspec(2, 3, left=.055, right=.975, top=.79, bottom=.26,
                          hspace=1.12, wspace=.29, height_ratios=[1, 1])
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    fig.text(.055, .954, 'Le CNN avec une voie RGB directe', fontsize=22, weight='bold', color='#142b45')
    fig.text(.055, .908, 'Critère de développement ' + ('atteint' if result['passed'] else 'non atteint'),
             fontsize=17, color='#16766d' if result['passed'] else '#9d552d')
    fig.text(.055, .86, f'12 cas artificiels · initial CNN identique · 7 paramètres ajoutés à zéro · '
             f'256 mises à jour par bras · {result["work_seconds"] / 60:.1f} min CPU', fontsize=11, color='#40536a')
    for arm, label, color in [('visual', 'RGB observés', COLORS[0]), ('constant', 'Entrée constante', COLORS[1])]:
        axes[0].plot(np.arange(1, 257), losses[arm], label=label, color=color, lw=1.7)
    axes[0].set(xlim=(1, 256), ylim=(0, None), xlabel='Mise à jour par bras', ylabel='BCE équilibrée')
    axes[0].set_title('Erreur avant chaque mise à jour', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=.18)
    summed = [(label, m[arm], color) for arm, label, color in zip(
        ('visual', 'constant', 'initial', 'zero'), ('RGB appris', 'Constant appris', 'Initial', 'Sortie zéro'), COLORS, strict=True)]
    case_bars(axes[1], summed, nonempty)
    axes[1].set_title('Sortie complète : dix cas non vides', fontsize=11)
    case_bars(axes[2], summed, empty, empty=True)
    axes[2].set_title('Sortie complète : les deux cas vides', fontsize=11)
    contributions = [('Somme', m['visual'], COLORS[0]), ('CNN seul', parts['contextual'], '#6d68a8'),
                     ('Voie directe seule', parts['direct'], '#d79245')]
    case_bars(axes[3], contributions, nonempty)
    axes[3].set_title('Contributions finales : diagnostic descriptif', fontsize=11)
    case_bars(axes[4], contributions, empty, empty=True)
    axes[4].set_title('Contributions finales sur les vides', fontsize=11)
    axes[5].axis('off')
    best = max(m[arm]['mean_nonempty_iou'] for arm in ('constant', 'initial', 'zero'))
    lines = [f'IoU de la somme : {m["visual"]["mean_nonempty_iou"]:.4f} / minimum 0,25',
             f'Avance sur le meilleur témoin : {m["visual"]["mean_nonempty_iou"] - best:.4f} / 0,10',
             'Faux pixels des vides : ' + ' ; '.join(str(m['visual']['cases'][i]['false_positive_pixels']) for i in empty),
             'Maximum permis : 33 faux pixels par vide',
             f'IoU CNN seul : {parts["contextual"]["mean_nonempty_iou"]:.4f}',
             f'IoU voie directe seule : {parts["direct"]["mean_nonempty_iou"]:.4f}']
    for y, line in zip(np.linspace(.97, .05, len(lines)), lines, strict=True):
        axes[5].text(0, y, line, fontsize=10, transform=axes[5].transAxes, color='#40536a', va='top')
    limitations = [
        'Le critère porte uniquement sur la somme. Les contributions ont été sauvegardées pendant les mêmes trois évaluations.',
        'Retirer une contribution décrit la fonction finale ; cela ne mesure pas son rôle causal pendant l’apprentissage. Les biais peuvent se compenser.',
        'Moyennes par paire, puis entre les cinq paires non vides. Un score moyen favorable peut masquer des cas entièrement manqués.',
        'Aucun jeu maîtrisé, généralisation indépendante, transfert ou apprentissage accéléré démontré par ce fit.',
    ]
    for y, line in zip((.129, .097, .065, .033), limitations, strict=True):
        fig.text(.055, y, line, fontsize=10, color='#40536a')
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=150, facecolor='white')
        stream.flush()
        os.fsync(stream.fileno())
    plt.close(fig)
    with METADATA.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(kind='reviewed_stationary_direct_fit_figure', result_sha256=sha(RESULT), review_sha256=sha(REVIEW),
                       journal_sha256=sha(JOURNAL), figure_sha256=sha(OUTPUT),
                       source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__).resolve(), HELPER)},
                       arrays_opened=0, model_inferences=0, optimizer_updates=0, limits=limitations), stream,
                  indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
