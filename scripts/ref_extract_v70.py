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
        self._counts = {}
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
        colours = np.unique(packed)
        if len(set(self._counts).union(int(c) for c in colours)) > self.max_colours:
            raise ValueError("Reference palette capacity exceeded; do not silently discard colours")
        if self._mode is None:
            self._mode = packed.copy()
            self._best = np.zeros(packed.shape, dtype=np.uint32)
        # Strictly larger wins; ties retain the incumbent, independent of RGB numeric ordering.
        for colour in colours:
            colour = int(colour)
            count = self._counts.get(colour)
            if count is None:
                count = np.zeros(packed.shape, dtype=np.uint32)
                self._counts[colour] = count
            count += packed == colour
            wins = count > self._best
            self._mode[wins] = colour
            self._best[wins] = count[wins]
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
