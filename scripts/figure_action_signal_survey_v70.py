"""Plot reviewed survey JSON only; no RGB archive or emulator is opened."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_action_signal_survey'
RESULT = FOLDER / 'survey_v0.json'
REVIEW = FOLDER / 'survey_review_v0.json'
OUTPUT = FOLDER / 'action_signal_v0.png'
METADATA = FOLDER / 'action_signal_figure_v0.json'
RESULT_SHA = '877dc75eb44141d89395f009f1325cffd87b7ea13f3c3ea247c7d5f6cd45b61d'
GAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
NAMES = ('Pong', 'Breakout', 'Freeway', 'Seaquest', 'Ms. Pac-Man', 'Asteroids')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed(path):
    relative = path.relative_to(ROOT).as_posix()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={relative}', str(path)) != git('rev-parse', f'HEAD:{relative}'):
        raise ValueError('Commit the exact figure source and reviewed inputs first')


def matrix():
    if sha(RESULT) != RESULT_SHA:
        raise ValueError('The completed survey result changed')
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if (review.get('kind') != 'action_signal_survey_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != RESULT_SHA
            or review.get('source_sha256') != result['source_sha256']
            or review.get('evidence_sha256', {}).get(RESULT.relative_to(ROOT).as_posix()) != RESULT_SHA
            or result.get('execution_status') != 'complete' or result.get('acquired_pairs') != 216
            or result.get('accounting_complete') is not True
            or tuple(c['game'] for c in result['cases']) != GAMES):
        raise ValueError('Exact complete survey and independent result review required')
    for path in (RESULT, REVIEW, Path(__file__).resolve()):
        committed(path)
    values, labels = [], []
    for name, case in zip(NAMES, result['cases'], strict=True):
        for target, anchor in zip((10000, 20000), case['anchors'], strict=True):
            if (anchor['target_delivered_frames'] != target or anchor['instrument_valid'] is not True
                    or len(anchor['pairs']) != 18):
                raise ValueError('All fixed anchors and repeated controls must be valid')
            row = []
            for command, pair in enumerate(anchor['pairs']):
                if pair['command'] != command or pair['status'] != 'acquired':
                    raise ValueError('Fixed complete command order required')
                counts = []
                for kind in ('raw', 'pooled'):
                    summary = pair[kind]
                    end = summary['horizons'][-1]
                    if (not summary['complete_eight_frame_pair'] or end['horizon'] != 8
                            or end['observed'] is not True or type(end['cumulative_union_pixels']) is not int
                            or not 0 <= end['cumulative_union_pixels'] <= 33600):
                        raise ValueError('No incomplete or unknown window may be plotted as zero')
                    counts.append(end['cumulative_union_pixels'])
                if counts[0] != counts[1]:
                    raise ValueError('This figure declares equal counts, not equal pixel sets')
                row.append(counts[0])
            values.append(row)
            labels.append(f'{name}  /  {target // 1000}k')
    return np.asarray(values), labels, result


def main():
    if OUTPUT.exists() or METADATA.exists():
        raise ValueError('Preserve every existing figure and receipt')
    values, labels, result = matrix()
    fig, ax = plt.subplots(figsize=(16.4, 9.0))
    fig.subplots_adjust(left=.145, right=.915, top=.79, bottom=.25)
    fig.text(.06, .94, 'Quelles commandes produisent un changement visible ?',
             fontsize=21, weight='bold', color='#142b45')
    fig.text(.06, .89, '216 essais prévus et acquis · 6 jeux · 2 moments fixés · 8 images futures par branche',
             fontsize=12, color='#40536a')
    fig.text(.06, .847, f'{int(np.count_nonzero(values))} paires avec un écart visible · '
             f'{result["known_charged_frames"]:,} images facturées · '
             f'{result["work_seconds"] / 60:.2f} min de collecte'.replace(',', ' '),
             fontsize=11, color='#40536a')
    palette = plt.get_cmap('Blues').copy()
    palette.set_bad('#edf0f4')
    rendered = ax.imshow(np.ma.masked_equal(values, 0), cmap=palette, norm=LogNorm(1, 33600),
                         interpolation='nearest', aspect='auto')
    ax.set_xticks(range(18), [str(i) for i in range(18)], fontsize=10)
    ax.set_yticks(range(12), labels, fontsize=10)
    ax.xaxis.tick_top()
    ax.set_xlabel('Commande locale (0 = NOOP)', fontsize=11, labelpad=12)
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, 18, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', length=0)
    for y, row in enumerate(values):
        for x, count in enumerate(row):
            ax.text(x, y, str(count), ha='center', va='center', fontsize=9,
                    color='#607083' if count == 0 else '#142b45')
    for boundary in (1.5, 3.5, 5.5, 7.5, 9.5):
        ax.axhline(boundary, color='#b4c0ce', linewidth=1.25)
    bar = fig.colorbar(rendered, ax=ax, fraction=.022, pad=.025, ticks=[1, 10, 100, 1000, 33600])
    bar.ax.set_yticklabels(['1', '10', '100', '1 000', '33 600'])
    bar.ax.tick_params(labelsize=8)
    bar.set_label('Pixels modifiés au moins une fois (échelle logarithmique)', fontsize=9)
    notes = [
        'Chaque nombre compte les pixels distincts différents de la branche NOOP pendant les 8 images futures.',
        '10k / 20k : seuils d’images ordinaires reçues ; dépassements de 0 à 3 images conservés dans les résultats.',
        '0 = aucun écart observé dans cette fenêtre. Tous les contrôles NOOP répétés ont passé la vérification.',
        'Les comptages finaux sont identiques en brut et en max-pool ; cela ne démontre pas que leurs masques sont égaux.',
        'Douze situations de développement : aucune preuve d’apprentissage, d’identité physique des objets ou de transfert.'
    ]
    for y, note in zip((.20, .16, .12, .08, .04), notes, strict=True):
        fig.text(.06, y, note, fontsize=10, color='#40536a')
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=160, facecolor='white')
        stream.flush(); os.fsync(stream.fileno())
    plt.close(fig)
    metadata = dict(kind='reviewed_action_signal_survey_figure', result_sha256=RESULT_SHA,
                    review_sha256=sha(REVIEW), source_sha256=sha(Path(__file__).resolve()),
                    figure_sha256=sha(OUTPUT), union_pixel_counts=values.tolist(), labels=labels,
                    archives_opened=0, new_emulated_frames=0, optimizer_updates=0,
                    limits=notes)
    with METADATA.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
