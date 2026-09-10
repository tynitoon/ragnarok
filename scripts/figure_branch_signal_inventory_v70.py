"""Render reviewed descriptive counts only; never opens RGB archives or a model."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_branch_signal_inventory'
RESULT = FOLDER / 'inventory_v0.json'
REVIEW = FOLDER / 'inventory_review_v0.json'
OUTPUT = FOLDER / 'observed_signal_v0.png'
NAMES = ['pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids',
         'pong_prefix_1', 'pong_prefix_2', 'pong_prefix_3']
LABELS = ['Pong', 'Breakout', 'Freeway', 'Seaquest', 'Ms. Pac-Man', 'Asteroids',
          'Pong / préfixe 1', 'Pong / préfixe 2', 'Pong / préfixe 3']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve the existing figure')
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('result_sha256') != sha(RESULT)
            or review.get('source_sha256') != result.get('source_sha256')
            or result.get('kind') != 'paid_branch_signal_inventory'
            or result.get('passed') is not True or result.get('present_pairs') != 6
            or result.get('absent_pairs') != 3 or result.get('optimizer_updates') != 0
            or result.get('new_emulated_frames') != 0 or result.get('gate_result') is not None
            or [row['case'] for row in result['cases']] != NAMES):
        raise ValueError('Complete reviewed inventory required')
    counts = []
    for kind in ('raw', 'pooled'):
        matrix = np.full((9, 8), np.nan)
        for i, case in enumerate(result['cases']):
            if case['pair_present'] != (i < 6):
                raise ValueError('The three absent pairs must remain explicit')
            rows = case[kind]['horizons']
            if len(rows) != 8 or [r['horizon'] for r in rows] != list(range(1, 9)):
                raise ValueError('Eight fixed horizons required')
            for j, row in enumerate(rows):
                if i < 6:
                    n = row['changed_pixels']
                    if (row['observed'] is not True or type(n) is not int or not 0 <= n <= 33600
                            or row['observed_pixel_denominator'] != 33600
                            or row['changed_fraction'] != n / 33600):
                        raise ValueError('Invalid observed pixel count')
                    matrix[i, j] = n
                elif row['observed'] or row['changed_pixels'] is not None:
                    raise ValueError('Absence must not be depicted as zero')
        counts.append(matrix)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.4), sharey=True)
    fig.patch.set_facecolor('#f6f7fb')
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#e1e4ea')
    for ax, values, title in zip(axes, counts, ['Images brutes', 'Images max-poolées'], strict=True):
        shown = ax.imshow(values + 1, cmap=cmap, norm=LogNorm(vmin=1, vmax=33601), aspect='auto')
        for i in range(9):
            for j in range(8):
                value = values[i, j]
                label = '—' if np.isnan(value) else f'{int(value)}\n{100 * value / 33600:.2f} %'
                ax.text(j, i, label, ha='center', va='center', fontsize=8.5,
                        color='white' if value >= 1000 else '#182737')
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xticks(range(8), labels=range(1, 9))
        ax.set_xlabel("Images écoulées depuis l'intervention")
        ax.set_yticks(range(9), labels=LABELS)
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Deux actions, quelles différences visibles ?', x=.08, ha='left', fontsize=21, weight='bold')
    fig.text(.08, .91, 'Commande 5 contre NOOP 0 · une ancre par jeu · six paires déjà enregistrées', fontsize=11)
    fig.text(.08, .12, 'Chaque case : pixels RGB différents / 33 600, puis pourcentage. — : aucune paire disponible.', fontsize=10)
    fig.text(.08, .08, 'Couleur : nombre de pixels + 1, échelle logarithmique fixe de 1 à 33 601. Tous les horizons sont conservés.', fontsize=9)
    fig.text(.08, .035, "Inventaire descriptif : ni apprentissage, ni identification de soi, ni transfert. Zéro différence observée ≠ absence de contrôle physique.", fontsize=10, color='#653b37')
    fig.subplots_adjust(left=.12, right=.98, top=.84, bottom=.22, wspace=.08)
    with OUTPUT.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({'figure': str(OUTPUT), 'sha256': sha(OUTPUT), 'result_sha256': sha(RESULT)}))


if __name__ == '__main__':
    main()
