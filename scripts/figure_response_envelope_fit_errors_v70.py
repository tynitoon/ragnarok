"""Show all twelve reviewed development cases at the already fixed threshold."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'craft_v6_out/v70_response_envelopes'
FIT = ROOT / 'craft_v6_out/v70_response_envelope_fit'
OUT = FIT / 'errors_v0'
GAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
NAMES = ('Pong', 'Breakout', 'Freeway', 'Seaquest', 'Ms. Pac-Man', 'Asteroids')
RESULT_SHA = '9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a'
REVIEW_SHA = '3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={rel}', str(path)) != git('rev-parse', f'HEAD:{rel}'):
        raise ValueError('Commit the exact reviewed inputs and this figure source first')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve every earlier atlas artifact')
    if any(name.split('.')[0] in ('torch', 'ale_py') for name in sys.modules):
        raise ValueError('No model or emulator runtime is used by this atlas')
    result_path, review_path = FIT / 'fit_v0.json', FIT / 'fit_review_v0.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if (sha(result_path) != RESULT_SHA or sha(review_path) != REVIEW_SHA
            or result['execution_status'] != 'complete' or result['protocol']['probability_threshold'] != .5
            or review['kind'] != 'response_envelope_fit_independent_result_review'
            or review['passed'] is not True or review['result_sha256'] != RESULT_SHA):
        raise ValueError('The exact completed and independently checked fit is required')
    paths = (result_path, review_path, TARGET / 'targets_v0.json', TARGET / 'targets_v0.npz',
             FIT / 'final_predictions.npz', Path(__file__).resolve())
    evidence = {}
    for path in paths:
        committed(path)
        rel = path.relative_to(ROOT).as_posix()
        actual = sha(path)
        if path not in (review_path, Path(__file__).resolve()) and review['evidence_sha256'].get(rel) != actual:
            raise ValueError('Atlas input differs from independently reviewed evidence')
        evidence[rel] = actual
    manifest = json.loads((TARGET / 'targets_v0.json').read_text(encoding='utf-8'))
    order = [(game, anchor) for game in GAMES for anchor in range(2)]
    if [(row['game'], row['anchor_index']) for row in manifest['cases']] != order:
        raise ValueError('All twelve original development cases in fixed order are required')
    # An exclusive directory marks this rendering attempt before any array decoding.
    OUT.mkdir()
    with np.load(TARGET / 'targets_v0.npz', allow_pickle=False) as data:
        past, labels = data['past_pairs'], data['observed_response_envelopes']
    with np.load(FIT / 'final_predictions.npz', allow_pickle=False) as data:
        probability = data['visual']
    if (signature(past) != manifest['output']['arrays']['past_pairs']
            or signature(labels) != manifest['output']['arrays']['observed_response_envelopes']
            or probability.shape != labels.shape or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError('Exact reviewed arrays and finite probabilities required')
    positive = probability >= .5
    colors = dict(correct='#178d83', false_positive='#db7655', missed='#4d8de6')
    artifacts = {}
    for game_index, (game, name) in enumerate(zip(GAMES, NAMES, strict=True)):
        fig, axes = plt.subplots(2, 5, figsize=(13, 7.9))
        fig.subplots_adjust(left=.055, right=.98, top=.86, bottom=.16, hspace=.23, wspace=.10)
        fig.suptitle(f'{name} · les deux situations de développement', fontsize=19, weight='bold', y=.98)
        fig.text(.055, .918, 'Réseau final · seuil fixé à 0,5 · mêmes images utilisées pour son ajustement',
                 fontsize=11, color='#40536a')
        for row in range(2):
            index = 2 * game_index + row
            target, prediction = labels[index], positive[index]
            error = np.full((*target.shape, 3), [16, 23, 32], dtype=np.uint8)
            for mask, color in ((target & prediction, colors['correct']),
                                (~target & prediction, colors['false_positive']),
                                (target & ~prediction, colors['missed'])):
                error[mask] = tuple(round(channel * 255) for channel in matplotlib.colors.to_rgb(color))
            panels = (past[index, 0], past[index, 1], target, prediction, error)
            for column, panel in enumerate(panels):
                axes[row, column].imshow(panel, interpolation='nearest', cmap='gray', vmin=0, vmax=1)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_visible(False)
            measure = result['metrics']['visual']['cases'][index]
            if (measure['game'], measure['anchor_index']) != order[index]:
                raise ValueError('Reviewed metric row order changed')
            score = 'IoU indisponible : cible vide' if measure['iou'] is None else f'IoU {measure["iou"]:.3f}'
            axes[row, 0].set_ylabel(f'Ancre {row + 1}', fontsize=10)
            axes[row, 4].set_xlabel(f'{score}\n{measure["false_positive_pixels"]} faux positifs', fontsize=9)
        for axis, title in zip(axes[0], ('Image passée 1', 'Image passée 2', 'Cible observée', 'Prédiction', 'Accords et erreurs'), strict=True):
            axis.set_title(title, fontsize=10)
        fig.legend(handles=[Patch(color=colors['correct'], label='Vrai positif'),
                            Patch(color=colors['false_positive'], label='Faux positif'),
                            Patch(color=colors['missed'], label='Positif manqué')],
                   loc='lower left', bbox_to_anchor=(.045, .065), ncols=3, frameon=False, fontsize=10)
        fig.text(.055, .035, 'La cible couvre des écarts futurs entre commandes et NOOP ; elle ne désigne pas un objet physique.',
                 fontsize=10, color='#40536a')
        path = OUT / f'{game}.png'
        with path.open('xb') as stream:
            fig.savefig(stream, format='png', dpi=150, facecolor='white')
            stream.flush()
            os.fsync(stream.fileno())
        plt.close(fig)
        artifacts[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    receipt = dict(kind='reviewed_response_envelope_fit_error_atlas', evidence_sha256=evidence,
                   artifacts=artifacts, all_development_anchors=12, probability_threshold=.5,
                   arrays_decoded=3, source_branch_archives_decoded=0, model_inferences=0,
                   optimizer_updates=0, new_emulated_frames=0,
                   purpose='post-result descriptive visualization only; no selection or revised criterion')
    with (OUT / 'atlas.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(str(OUT))


if __name__ == '__main__':
    main()
