"""The promise view (docs/ARC4_V83_PROTOCOL.md §1; synthesis §2.2): a 64x64 window on me over the
colour-free foreground (pixels that differ from my own background), me masked out, four channels:
objects, pixels that appeared, pixels that vanished, small blobs (at most twice my size). Logged
unrotated and bit-packed with an unrotated 8x8 whole-screen foreground; `features` rotates the window
so the candidate slot's move points up (4 rotations, diagonals to up-right with a flag; stay and
private slots unrotated and flagged) and returns ~430 numbers. Colours never enter the view.
"""
import numpy as np
from scipy import ndimage

WIN = 64
HALF = WIN // 2
CONE_BINS = (4, 8, 12, 16, 24, 32, 48)
N_FEAT = 4 * (64 + 16 + 4) + 2 * (len(CONE_BINS) + 1) + 8 + 1 + 1 + 64 + 1


def fg_mask(obs, bg):
    return (obs != bg).any(-1)


def _me_mask(lab, fg, me, r=4):
    """My component: the foreground component under my position, else the nearest within r pixels."""
    m = np.zeros(fg.shape, bool)
    if me is None:
        return m
    H, W = fg.shape
    x, y = int(round(me[0])), int(round(me[1]))
    y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
    sub = lab[y0:y1, x0:x1]
    ids = np.unique(sub[sub > 0])
    if len(ids):
        m = np.isin(lab, ids)
    return m


def _window(mask, cx, cy):
    H, W = mask.shape
    pad = np.zeros((H + WIN, W + WIN), bool)
    pad[HALF:HALF + H, HALF:HALF + W] = mask
    x, y = int(round(cx)), int(round(cy))
    return pad[y:y + WIN, x:x + WIN]


def pool(a, n):
    H, W = a.shape
    a = a[: H - H % n, : W - W % n]
    return a.reshape(n, a.shape[0] // n, n, a.shape[1] // n).mean((1, 3))


def log_view(obs, prev_fg, bg, me, my_size):
    """Returns (packed 4x64x64 window bits, uint8 8x8 whole-screen foreground, lost flag, fg mask)."""
    fg = fg_mask(obs, bg)
    lab, n = ndimage.label(fg)
    mem = _me_mask(lab, fg, me)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small_ids = sizes <= 2.0 * (my_size or 32.0)
    small_ids[0] = False
    small = small_ids[lab]
    pf = fg if prev_fg is None else prev_fg
    chans = (fg & ~mem, fg & ~pf & ~mem, pf & ~fg & ~mem, small & ~mem)
    H, W = fg.shape
    lost = me is None
    cx, cy = (W / 2.0, H / 2.0) if lost else me
    win = np.stack([_window(c, cx, cy) for c in chans])
    screen = np.round(pool(fg.astype(np.float32), 8) * 255).astype(np.uint8)
    return np.packbits(win), screen, lost, fg


def unpack(packed):
    return np.unpackbits(packed)[: 4 * WIN * WIN].reshape(4, WIN, WIN).astype(bool)


def orientation(move):
    """(k quarter turns counter-clockwise, diagonal flag, oriented flag) bringing move (sx, sy) to up
    (or up-right for diagonals); image y points down."""
    sx, sy = move
    if sx == 0 and sy == 0:
        return 0, False, False
    k = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3, (1, -1): 0, (1, 1): 1, (-1, 1): 2, (-1, -1): 3}[(sx, sy)]
    return k, sx != 0 and sy != 0, True


def _cone_bins(ch, diag):
    """One-hot distance bin of the nearest set pixel in the forward cone (up, or up-right if diag)."""
    ys, xs = np.nonzero(ch)
    dx, dy = xs - (HALF - 0.5), (HALF - 0.5) - ys          # dy > 0 ahead (up)
    if diag:
        dx, dy = (dx - dy) / np.sqrt(2), (dx + dy) / np.sqrt(2)
    ahead = (dy > 0) & (np.abs(dx) <= dy)
    out = np.zeros(len(CONE_BINS) + 1, np.float32)
    if ahead.any():
        d = float(np.sqrt(dx[ahead] ** 2 + dy[ahead] ** 2).min())
        out[int(np.searchsorted(CONE_BINS, d))] = 1.0
    else:
        out[-1] = 1.0
    return out


def oriented_window(packed, move):
    k, diag, oriented = orientation(move)
    return np.rot90(unpack(packed), k, axes=(1, 2)), k, diag, oriented


def features(packed, screen, lost, me, lives, move, extra, private, shape=(210, 160)):
    """The ~430 promise features for one (view, slot); `move` is the slot's (sign dx, sign dy)."""
    win, k, diag, oriented = oriented_window(packed, move)
    w = win.astype(np.float32)
    f = [pool(w[c], n).ravel() for c in range(4) for n in (8, 4, 2)]
    f.append(_cone_bins(win[0], diag))
    f.append(_cone_bins(win[1], diag))
    rot = np.zeros(4, np.float32)
    if oriented:
        rot[k] = 1.0
    f.append(rot)
    f.append(np.array([diag, not oriented and not private, private, extra], np.float32))
    H, W = shape
    if lost or not oriented:
        edge = 0.0
    else:
        x, y = me
        ahead = {0: y / H, 1: (W - x) / W, 2: (H - y) / H, 3: x / W}
        edge = ahead[k] if not diag else min(ahead[k], ahead[(k + 1) % 4])
    f.append(np.array([edge, lives == 1], np.float32))
    f.append(screen.astype(np.float32).ravel() / 255.0)
    f.append(np.array([lost], np.float32))
    out = np.concatenate(f)
    assert out.shape == (N_FEAT,)
    return out
