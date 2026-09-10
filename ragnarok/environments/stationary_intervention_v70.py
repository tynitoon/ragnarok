"""Synthetic stationary intervention used only to calibrate a learning pipeline."""
import numpy as np

POSITIVE_COUNTS = (1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2)


def new_ledger():
    return dict(initial_screens_generated=0, synthetic_step_entries=0, synthetic_steps=0,
                synthetic_rgb_frames=0, new_emulated_frames=0, optimizer_updates=0)


def generate_screens(seed=774, height=210, width=160, positive_counts=POSITIVE_COUNTS,
                     ledger=None, checkpoint=lambda: None):
    """Draw visible initial conditions; never return an oracle target mask."""
    if height < 1 or width < 1 or any(count < 0 or count > height * width for count in positive_counts):
        raise ValueError('Valid image dimensions and support sizes required')
    rng = np.random.Generator(np.random.PCG64(seed))
    screens = []
    corners = (0, width - 1, (height - 1) * width, height * width - 1)
    for index, count in enumerate(positive_counts):
        checkpoint()
        screen = np.zeros((height, width, 3), dtype=np.uint8)
        screen[..., 1:] = rng.integers(0, 256, size=(height, width, 2), dtype=np.uint8)
        if count:
            forced = corners[index % 4]
            remaining = np.delete(np.arange(height * width), forced)
            selected = np.r_[forced, rng.choice(remaining, size=count - 1, replace=False)]
            screen.reshape(-1, 3)[selected, 0] = 255
            screen.reshape(-1, 3)[selected, 2] = 0
        screens.append(screen)
        if ledger is not None:
            ledger['initial_screens_generated'] += 1
    return np.stack(screens)


def step_rgb(screen, action):
    """World law: action 1 turns blue on at red pixels; NOOP leaves RGB alone."""
    image = np.asarray(screen)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or action not in (0, 1):
        raise ValueError('uint8 RGB and one of the two fixed commands required')
    following = image.copy()
    if action == 1:
        following[..., 2][following[..., 0] == 255] = 255
    return following


def collect_interventions(screens, horizon=8, ledger=None, checkpoint=lambda: None,
                          on_step_return=None):
    """Observe two NOOP steps, then both branches from the same owned RGB state."""
    images = np.asarray(screens)
    if (images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3
            or min(images.shape) < 1 or horizon < 1
            or not np.isin(images[..., 0], [0, 255]).all()
            or (images[..., 2][images[..., 0] == 255] != 0).any()):
        raise ValueError('Admitted stationary initial RGB family required')
    pairs, noop, acted = [], [], []
    if ledger is None:
        ledger = new_ledger()
    def emit(screen, action):
        checkpoint()
        ledger['synthetic_step_entries'] += 1
        value = step_rgb(screen, action)
        ledger['synthetic_steps'] += 1
        ledger['synthetic_rgb_frames'] += 1
        if on_step_return is not None:
            on_step_return(ledger.copy())
        return value
    for screen in images:
        first = emit(screen, 0)
        second = emit(first, 0)
        pairs.append(np.stack((first, second)))
        branches = []
        for action in (0, 1):
            state = second.copy()
            frames = []
            for _ in range(horizon):
                state = emit(state, action)
                frames.append(state)
            branches.append(np.stack(frames))
        noop.append(branches[0])
        acted.append(branches[1])
    arrays = dict(past_pairs=np.stack(pairs), noop_futures=np.stack(noop), action_futures=np.stack(acted))
    # This is the only target-construction path: differences of observed branches.
    arrays['observed_response_envelopes'] = np.any(
        arrays['noop_futures'] != arrays['action_futures'], axis=(1, 4))
    return arrays, ledger
