"""Frame codes for V2's cells (docs/ARC4_V84_PROTOCOL_EN.md §5, lesson 5): the block-averaged luminance
frame (0.299 R + 0.587 G + 0.114 B; block edges floor(i H / h), floor(j W / w), summed with
np.add.reduceat), quantised to d levels. Four resolutions R = (w, h, d), all stored for every cell; the
archive keys on one of them, chosen adaptively (Go-Explore's downscaling, Ecoffet et al. 2021)."""
import numpy as np

RES = ((8, 6, 4), (11, 8, 8), (16, 12, 8), (22, 16, 8))


def luminance(obs):
    return obs[..., 0] * 0.299 + obs[..., 1] * 0.587 + obs[..., 2] * 0.114


def _edges(n, k):
    return np.array([(i * n) // k for i in range(k)], np.int64)


def block_mean(lum, w, h):
    """Reference implementation (np.add.reduceat), used by the tests."""
    H, W = lum.shape
    ey, ex = _edges(H, h), _edges(W, w)
    s = np.add.reduceat(np.add.reduceat(lum, ey, axis=0), ex, axis=1)
    hy = np.diff(np.append(ey, H))
    wx = np.diff(np.append(ex, W))
    return s / np.outer(hy, wx)


_POOL = {}


def _pool(H, W, w, h):
    """Averaging matrices with the same block edges as block_mean: (h x H) and (W x w)."""
    k = (H, W, w, h)
    if k not in _POOL:
        ey, ex = np.append(_edges(H, h), H), np.append(_edges(W, w), W)
        Ay = np.zeros((h, H))
        for i in range(h):
            Ay[i, ey[i]:ey[i + 1]] = 1.0 / (ey[i + 1] - ey[i])
        Ax = np.zeros((W, w))
        for j in range(w):
            Ax[ex[j]:ex[j + 1], j] = 1.0 / (ex[j + 1] - ex[j])
        _POOL[k] = (Ay, Ax)
    return _POOL[k]


_LUM = np.array([0.299, 0.587, 0.114])


def codes(obs):
    """The four codes (bytes) of a frame, coarse to fine, and the finest quantised blocks (for the share
    of changed blocks between a parent and its child cell)."""
    lum = obs.astype(np.float64) @ _LUM
    H, W = lum.shape
    out, fine = [], None
    for w, h, d in RES:
        Ay, Ax = _pool(H, W, w, h)
        m = Ay @ lum @ Ax
        q = np.minimum((m * d / 256.0).astype(np.int64), d - 1).astype(np.uint8)
        out.append(q.tobytes())
        fine = q
    return tuple(out), fine
