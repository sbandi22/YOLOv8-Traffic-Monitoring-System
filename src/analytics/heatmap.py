"""Traffic heatmap generator with temporal decay."""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self, frame_shape: Tuple[int, int] = (720, 1280), decay: float = 0.985, blur_kernel: int = 31):
        self.shape = frame_shape
        self.decay = float(decay)
        self.blur_kernel = int(blur_kernel)
        self._heat = np.zeros(frame_shape, dtype=np.float32)

    def update(self, tracks: Sequence[dict]):
        self._heat *= self.decay
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr["bbox"])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.shape[1] - 1, x2); y2 = min(self.shape[0] - 1, y2)
            if x2 > x1 and y2 > y1:
                self._heat[y1:y2, x1:x2] += 1.0
        return self._heat

    def render(self, colormap=cv2.COLORMAP_JET) -> np.ndarray:
        blurred = cv2.GaussianBlur(self._heat, (self.blur_kernel, self.blur_kernel), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    @property
    def raw(self) -> np.ndarray:
        return self._heat

    def reset(self):
        self._heat[:] = 0
