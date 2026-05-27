"""Traffic density estimator."""
from __future__ import annotations

from collections import deque
from typing import Sequence

import cv2
import numpy as np


class DensityEstimator:
    def __init__(self, roi_polygon=None, frame_shape=(720, 1280), window: int = 30):
        self.frame_shape = frame_shape
        self.window = window
        self._history = deque(maxlen=window)
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        self._roi_area = None
        if self.roi_polygon is not None:
            mask = np.zeros(frame_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [self.roi_polygon], 1)
            self._roi_area = int(mask.sum())
            self._mask = mask
        else:
            self._mask = None
            self._roi_area = frame_shape[0] * frame_shape[1]

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def update(self, tracks: Sequence[dict]) -> float:
        """Return occupancy ratio (0..1) for the current frame."""
        if not tracks:
            self._history.append(0.0)
            return 0.0
        if self._mask is None:
            occupied = sum(self._bbox_area(t["bbox"]) for t in tracks)
            ratio = occupied / float(self._roi_area)
        else:
            frame_mask = np.zeros(self.frame_shape, dtype=np.uint8)
            for t in tracks:
                x1, y1, x2, y2 = map(int, t["bbox"])
                cv2.rectangle(frame_mask, (x1, y1), (x2, y2), 1, -1)
            ratio = float(np.logical_and(frame_mask, self._mask).sum()) / float(self._roi_area)
        ratio = min(1.0, ratio)
        self._history.append(ratio)
        return ratio

    def rolling_average(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def history(self):
        return list(self._history)
