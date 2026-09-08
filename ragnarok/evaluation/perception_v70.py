"""Experimenter-only framewise perception measurements with explicit denominators."""
from collections import deque
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_centers(truth, predicted, *, tolerance=4.0):
    """Maximum-cardinality gated assignment, then minimum summed distance.

    Filtering an unconstrained Hungarian assignment afterwards can lose valid
    matches. An invalid edge costs more than every valid distance combined.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive and finite")
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    for points in (truth, predicted):
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Expected finite Nx2 centers")
    if not len(truth) or not len(predicted):
        return []
    distance = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2)
    invalid_cost = (min(len(truth), len(predicted)) + 1) * tolerance
    rows, cols = linear_sum_assignment(np.where(distance <= tolerance, distance, invalid_cost))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if distance[i, j] <= tolerance]


class PerceptionScorer:
    def __init__(self, *, tolerance=4.0, motion_frames=8, roi=(0, 34, 160, 194)):
        if type(motion_frames) is not int or motion_frames < 1:
            raise ValueError("motion_frames must be a positive integer")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Tolerance must be positive and finite")
        if len(roi) != 4 or not roi[0] < roi[2] or not roi[1] < roi[3]:
            raise ValueError("Expected a nonempty xyxy evaluation ROI")
        self.tolerance, self.roi = tolerance, roi
        self._history = deque(maxlen=motion_frames + 1)
        self._episode = self._frame = None
        self._frames = self._predictions = self._matches = self._eligible = self._recalled = 0
        self._per_type = {}

    def add(self, truths, boxes, *, frame, episode):
        if type(frame) is not int or type(episode) is not int or frame < 0 or episode < 0:
            raise ValueError("Frame and episode must be nonnegative integers")
        if self._frame is not None and (frame != self._frame + 1 or episode not in (self._episode, self._episode + 1)):
            raise ValueError("Scorer requires consecutive raw frames and monotone episode ids")
        boxes = np.asarray(boxes, dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
            raise ValueError("Expected finite Nx4 boxes with positive dimensions")
        truths = list(truths)
        if len({obj.kind for obj in truths}) != len(truths):
            raise ValueError("Pong scorer requires one object per declared kind")
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        left, top, right, bottom = self.roi
        centers = centers[(centers[:, 0] >= left) & (centers[:, 0] < right) &
                          (centers[:, 1] >= top) & (centers[:, 1] < bottom)]
        visible = [o for o in truths if left <= o.center[0] < right and top <= o.center[1] < bottom]
        positions = {obj.kind: obj.center for obj in visible}
        if episode != self._episode:
            self._history.clear()
        self._history.append(positions)
        moving = {kind for kind, center in positions.items()
                  if any(kind in previous and previous[kind] != center for previous in self._history)}
        pairs = match_centers(np.array([o.center for o in visible], dtype=float).reshape(-1, 2),
                             centers, tolerance=self.tolerance)
        matched = {visible[i].kind for i, _ in pairs}
        self._frames += 1
        self._predictions += len(centers)
        self._matches += len(pairs)
        self._eligible += len(moving)
        self._recalled += len(moving & matched)
        for kind in positions:
            stat = self._per_type.setdefault(kind, dict(visible=0, eligible=0, recalled=0))
            stat["visible"] += 1
            stat["eligible"] += int(kind in moving)
            stat["recalled"] += int(kind in moving and kind in matched)
        self._frame, self._episode = frame, episode

    def report(self):
        return dict(frames=self._frames, predictions=self._predictions, matched_visible=self._matches,
                    eligible_moving=self._eligible, recalled_moving=self._recalled,
                    recall_moving=self._recalled/self._eligible if self._eligible else None,
                    precision_visible=self._matches/self._predictions if self._predictions else None,
                    per_type={k: dict(v, recall_moving=v["recalled"]/v["eligible"] if v["eligible"] else None)
                              for k, v in self._per_type.items()},
                    gate_result=None)
