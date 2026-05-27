"""Speed estimation in km/h via pixel-per-meter or homography calibration."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        smoothing_window: int = 5,
        homography: Optional[np.ndarray] = None,
    ):
        self.fps = float(fps)
        self.ppm = float(pixels_per_meter)
        self.window = smoothing_window
        self.H = homography
        self._last_pos: Dict[int, Tuple[float, float]] = {}
        self._speeds: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))

    def _to_world(self, pt):
        if self.H is None:
            return pt
        v = np.array([pt[0], pt[1], 1.0])
        w = self.H @ v
        return (w[0] / w[2], w[1] / w[2])

    def update(self, tracks: Sequence[dict]) -> Dict[int, float]:
        speeds: Dict[int, float] = {}
        for tr in tracks:
            tid = int(tr["id"])
            cur = self._to_world(tr["centroid"])
            if tid in self._last_pos:
                prev = self._last_pos[tid]
                if self.H is None:
                    dist_px = float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
                    dist_m = dist_px / self.ppm
                else:
                    dist_m = float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
                # Speed (m/s -> km/h)
                v = dist_m * self.fps * 3.6
                self._speeds[tid].append(v)
                speeds[tid] = float(np.mean(self._speeds[tid]))
            self._last_pos[tid] = cur
        return speeds

    def reset_for_track(self, tid: int):
        self._last_pos.pop(tid, None)
        self._speeds.pop(tid, None)
