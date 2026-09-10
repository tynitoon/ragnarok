"""Plot the reviewed fit's JSON and journal only, without opening any arrays."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_response_envelope_fit'
RESULT = FOLDER / 'fit_v0.json'
REVIEW = FOLDER / 'fit_review_v0.json'
JOURNAL = FOLDER / 'fit_v0_progress.jsonl'
OUTPUT = FOLDER / 'fit_v0.png'
METADATA = FOLDER / 'fit_v0_figure.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={rel}', str(path)) != git('rev-parse', f'HEAD:{rel}'):
        raise ValueError('Commit the exact reviewed inputs and figure source first')


def main():
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError('Preserve the existing figure and its receipt')
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if (result['execution_status'] != 'complete' or result['optimizer_update_count_exact'] is not True
            or result['completed_updates'] != {'visual': 256, 'constant': 256}
            or review.get('kind') != 'response_envelope_fit_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != sha(RESULT)
            or review.get('source_sha256') != result['source_sha256']
            or review.get('evidence_sha256', {}).get(RESULT.relative_to(ROOT).as_posix()) != sha(RESULT)
            or sha(JOURNAL) != result['progress_sha256']
            or sha(FOLDER / 'fit_v0_payload.json') != result['scientific_payload_sha256']):
        raise ValueError('An independently reviewed, complete and exactly counted fit is required')
    for path in (RESULT, REVIEW, JOURNAL, Path(__file__).resolve()):
        committed(path)
    losses = {'visual': [], 'constant': []}
    with JOURNAL.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'update_complete':
                sequence = losses[row['arm']]
                if row['update'] != len(sequence) + 1:
                    raise ValueError('No incomplete or reordered update history may be plotted')
                sequence.append(row['loss'])
    if any(len(values) != 256 or not np.isfinite(values).all() for values in losses.values()):
        raise ValueError('Both complete finite loss histories required')
    metrics = result['metrics']
    games = ('pong', 'breakout', 'freeway', 'ms_pacman', 'asteroids')
    names = ('Pong', 'Breakout', 'Freeway', 'Ms. Pac-Man', 'Asteroids')
    fig = plt.figure(figsize=(14.5, 9.0))
    grid = fig.add_gridspec(1, 2, left=.07, right=.97, top=.73, bottom=.32, wspace=.26)
    loss_axis, score_axis = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.text(.06, .94, 'Premier ajustement sur des images Atari', fontsize=22, weight='bold', color='#142b45')
    outcome = 'Critère de faisabilité atteint' if result['passed'] else 'Critère de faisabilité non atteint'
    fig.text(.06, .885, outcome, fontsize=15, color='#16766d' if result['passed'] else '#9d552d')
    fig.text(.06, .835, '12 situations de développement · 256 mises à jour par bras · '
             f'{result["work_seconds"] / 60:.1f} min sur CPU · aucune validation indépendante', fontsize=11, color='#40536a')
    for arm, color, label in [('visual', '#167f86', 'Images réelles'), ('constant', '#d79245', 'Entrée constante')]:
        loss_axis.plot(np.arange(1, 257), losses[arm], color=color, label=label, linewidth=1.8)
    loss_axis.set_title('Erreur sur les douze situations utilisées pour ajuster le réseau', fontsize=10, pad=12)
    loss_axis.set_xlabel('Mise à jour de chaque bras')
    loss_axis.set_ylabel('BCE équilibrée, moyenne par situation')
    loss_axis.set_xlim(1, 256)
    loss_axis.set_ylim(bottom=0)
    loss_axis.legend(frameon=False, fontsize=9)
    loss_axis.grid(alpha=.18)
    offsets = (-.27, -.09, .09, .27)
    for arm, offset, color, label in zip(('visual', 'constant', 'initial', 'zero'), offsets,
            ('#167f86', '#d79245', '#91a0b2', '#d7dee5'),
            ('Images réelles', 'Entrée constante', 'Avant ajustement', 'Sortie zéro'), strict=True):
        values = [metrics[arm]['nonempty_by_game'][game]['iou'] for game in games]
        score_axis.bar(np.arange(5) + offset, values, width=.17, color=color, label=label)
    score_axis.set_xticks(np.arange(5), names, rotation=18, ha='right', fontsize=9)
    score_axis.set_ylim(0, 1)
    score_axis.set_ylabel('IoU au seuil fixé de 0,5')
    score_axis.set_title('Accord final avec les zones de contraste observées', fontsize=10, pad=12)
    score_axis.legend(frameon=False, fontsize=8, ncols=2, loc='upper left')
    score_axis.grid(axis='y', alpha=.18)
    visual = metrics['visual']
    best = max(metrics[arm]['mean_nonempty_iou'] for arm in ('constant', 'initial', 'zero'))
    empty = visual['empty_anchor_false_positive_fractions']
    notes = [
        f'IoU moyen : {visual["mean_nonempty_iou"]:.3f} (minimum 0,25) · avance sur le meilleur témoin : '
        f'{visual["mean_nonempty_iou"] - best:.3f} (minimum 0,10).',
        f'Seaquest, deux cibles vides : {100 * empty[0]:.3f} % et {100 * empty[1]:.3f} % de faux positifs '
        '(maximum 0,1 % pour chacune).',
        'Les mesures regroupent les ancres par jeu ; les cibles vides sont évaluées séparément, sans IoU artificiellement parfait.',
        'La cible réunit les écarts des commandes face à NOOP sur huit images futures. Elle ne désigne pas un objet physique.',
        'Un ajustement sur ces images reste compatible avec leur mémorisation. Aucun jeu maîtrisé, transfert ou apprentissage accéléré établi.'
    ]
    for y, note in zip((.245, .195, .145, .095, .045), notes, strict=True):
        fig.text(.06, y, note, fontsize=10, color='#40536a')
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=160, facecolor='white')
        stream.flush()
        os.fsync(stream.fileno())
    plt.close(fig)
    with METADATA.open('x', encoding='utf-8') as stream:
        json.dump(dict(kind='reviewed_response_envelope_fit_figure', result_sha256=sha(RESULT),
                       review_sha256=sha(REVIEW), journal_sha256=sha(JOURNAL), figure_sha256=sha(OUTPUT),
                       source_sha256=sha(Path(__file__).resolve()), arrays_opened=0, model_inferences=0,
                       optimizer_updates=0, limits=notes), stream, indent=2, ensure_ascii=False)
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
