"""Experimenter-side classical reference: causal pixel-mode background and components.

No game name, semantic palette, RAM, score, pretrained model or labels. This
reference is not a teacher for the learned parser. Update it only on ordinary
observations; querying a counterfactual must not alter ordinary background state.
"""
import numpy as np
from scipy import ndimage


class ReferenceExtractor:
    def __init__(self, *, max_colours=256):
        if type(max_colours) is not int or not 1 <= max_colours <= 4096:
            raise ValueError("max_colours must be in 1..4096")
        self.max_colours = max_colours
        self.reset()

    def reset(self):
        self._counts = self._colour_ids = self._pixel_indices = None
        self._palette_size = 0
        self._mode = self._best = None
        self.frames_seen = 0

    @staticmethod
    def _packed(rgb):
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
                rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]):
            raise ValueError("Expected a nonempty uint8 RGB image")
        # Pack once without allocating a uint32 copy of all three channels.
        packed = np.empty(rgb.shape[:2], dtype="<u4")
        channels = packed.view(np.uint8).reshape(*rgb.shape[:2], 4)
        channels[..., :3] = rgb
        channels[..., 3] = 0
        return packed

    def update(self, rgb):
        self._update_packed(self._packed(rgb))

    def _update_packed(self, packed):
        if self._mode is not None and self._mode.shape != packed.shape:
            raise ValueError("Resolution changed; start a new background explicitly")
        if self.frames_seen >= 1_000_000:
            raise ValueError("Background observation limit reached; counters cannot overflow")
        if self._colour_ids is None:
            # 24-bit RGB -> compact ID. Zero means unseen. Lazily allocated;
            # IDs cover every observed colour, without a game-specific palette.
            self._colour_ids = np.zeros(1 << 24, dtype=np.uint16)
        ids = self._colour_ids[packed]
        unseen = ids == 0
        if unseen.any():
            colours = np.unique(packed[unseen])
            needed = self._palette_size + len(colours)
            if needed > self.max_colours:
                raise ValueError("Reference palette capacity exceeded; do not silently discard colours")
            if self._counts is None or needed > self._counts.shape[0]:
                capacity = min(self.max_colours, max(16, 1 << (needed - 1).bit_length()))
                counts = np.zeros((capacity, packed.size), dtype=np.uint32)
                if self._counts is not None:
                    counts[:self._palette_size] = self._counts[:self._palette_size]
                self._counts = counts
            self._colour_ids[colours] = np.arange(self._palette_size + 1, needed + 1, dtype=np.uint16)
            self._palette_size = needed
            ids[unseen] = self._colour_ids[packed[unseen]]
        if self._mode is None:
            self._mode = packed.copy()
            self._best = np.zeros(packed.shape, dtype=np.uint32)
            self._pixel_indices = np.arange(packed.size, dtype=np.intp)
        # Strictly larger wins; ties retain the incumbent, independent of RGB numeric ordering.
        # Each pixel increments exactly one distinct (colour, pixel) counter.
        # Cast before multiplying: uint16 palette IDs cannot hold flattened offsets.
        indices = (ids.ravel().astype(np.intp) - 1) * packed.size + self._pixel_indices
        flat_counts = self._counts.ravel()
        updated = flat_counts[indices] + 1
        flat_counts[indices] = updated
        updated = updated.reshape(packed.shape)
        wins = updated > self._best
        self._mode[wins] = packed[wins]
        self._best[wins] = updated[wins]
        self.frames_seen += 1

    def extract(self, rgb):
        return self._extract_packed(self._packed(rgb))

    def _extract_packed(self, packed):
        if self._mode is None or self._mode.shape != packed.shape:
            raise ValueError("Observe a background at this resolution first")
        # Four-connected foreground: no game-specific filter, HUD mask, colour or size threshold.
        labels, _ = ndimage.label(packed != self._mode)
        boxes = [(int(x.start), int(y.start), int(x.stop-x.start), int(y.stop-y.start))
                 for y, x in ndimage.find_objects(labels)]
        return np.array(boxes, dtype=np.int32).reshape(-1, 4)

    def observe(self, rgb):
        packed = self._packed(rgb)
        self._update_packed(packed)
        return self._extract_packed(packed)
