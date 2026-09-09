"""Render reviewed diagnostic arrays only; never import a model or rerun fitting."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_free_displacement_fit'
RESULT = FOLDER / 'fit_v0.json'
REVIEW = FOLDER / 'fit_review_v0.json'
OUTPUT = FOLDER / 'diagnostic_figure_v0.png'
RECEIPT = FOLDER / 'diagnostic_figure_v0.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render():
    if OUTPUT.exists() or RECEIPT.exists():
        raise ValueError('Preserve the existing diagnostic figure or partial attempt')
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    result_sha = sha(RESULT)
    if (review.get('passed') is not True
            or review['evidence_sha256'].get(RESULT.relative_to(ROOT).as_posix()) != result_sha
            or result.get('kind') != 'free_displacement_fit_result'
            or result.get('execution_complete') is not True
            or result.get('optimizer_updates') != 256):
        raise ValueError('Independently reviewed complete free-field diagnostic required')
    arrays = result['final_arrays']
    rgb = [np.asarray(arrays[key], dtype=np.float32)
           for key in ('source_rgb', 'target_rgb', 'reconstruction_rgb')]
    field = np.asarray(arrays['displacement'], dtype=np.float32)
    if (any(a.shape != (2, 3, 16, 16) or not np.isfinite(a).all()
            or (a < 0).any() or (a > 1).any() for a in rgb)
            or field.shape != (2, 2, 16, 16) or not np.isfinite(field).all()):
        raise ValueError('Finite reviewed arrays with the exact diagnostic geometry required')
    errors = np.abs(rgb[2] - rgb[1]).mean(axis=(1, 2, 3))
    last = result['measurements'][-1]
    if not np.allclose(errors, last['scene_l1'], rtol=0, atol=1e-8):
        raise ValueError('Stored reconstruction does not reproduce the final native errors')
    threshold = .01171875
    passed = last['image_l1'] < threshold
    if passed != result['native_fit_below_declared_threshold']:
        raise ValueError('Native criterion and recorded verdict disagree')

    fig, axes = plt.subplots(2, 5, figsize=(13.2, 6.8))
    fig.patch.set_facecolor('#f6f6f2')
    titles = ('Source commune', 'Futur attendu', 'Reconstruction finale',
              'Erreur RGB moyenne', 'Déplacements finaux')
    yy, xx = np.mgrid[0:16:2, 0:16:2]
    for scene in range(2):
        for column, data in enumerate(rgb):
            axes[scene, column].imshow(data[scene].transpose(1, 2, 0), interpolation='nearest')
        error_image = np.abs(rgb[2][scene] - rgb[1][scene]).mean(axis=0)
        axes[scene, 3].imshow(error_image, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
        axes[scene, 3].set_xlabel(f'L1 native = {float(errors[scene]):.6f}', fontsize=10)
        axes[scene, 4].imshow(rgb[0][scene].transpose(1, 2, 0), interpolation='nearest')
        axes[scene, 4].quiver(xx, yy, field[scene, 0, ::2, ::2], field[scene, 1, ::2, ::2],
                             angles='xy', scale_units='xy', scale=1, color='#24cde0', width=.006)
        axes[scene, 4].set_xlabel('Flèches à l’échelle des pixels\n1 pixel sur 2 affiché', fontsize=9)
        for column, axis in enumerate(axes[scene]):
            if scene == 0:
                axis.set_title(titles[column], fontsize=11, pad=9)
            axis.set_xlim(-.5, 15.5)
            axis.set_ylim(15.5, -.5)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[scene, 0].set_ylabel(f'Scène {scene + 1}', fontsize=11)
    outcome = 'atteint' if passed else 'NON atteint'
    fig.suptitle('Diagnostic synthétique : optimisation directe de deux champs libres',
                 fontsize=15, x=.06, ha='left', y=.985)
    fig.text(.06, .924,
             f'Critère natif {outcome} : erreur moyenne {last["image_l1"]:.6f} ; seuil strict {threshold:.6f}.',
             fontsize=11, color='#21664b' if passed else '#9b332c')
    fig.text(.06, .055,
             '256 mises à jour ; paramètres indépendants par scène. Même paire utilisée pour ajuster et mesurer.\n'
             'Ce témoin ne mesure ni perception apprise, ni généralisation, ni jeu Atari ou transfert.',
             fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.06, right=.98, top=.85, bottom=.16, hspace=.28, wspace=.12)
    fig.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = dict(
        kind='reviewed_free_field_diagnostic_figure', result_sha256=result_sha,
        review_sha256=sha(REVIEW), script_sha256=sha(Path(__file__)), image_sha256=sha(OUTPUT),
        native_scene_l1=errors.tolist(), native_fit_below_declared_threshold=passed,
        model_inferences=0, optimizer_updates=0, new_emulated_frames=0, gpu_used=False,
        figure_only=True,
    )
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


if __name__ == '__main__':
    print(json.dumps(render()))
