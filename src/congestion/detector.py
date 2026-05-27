"""Congestion detection + hotspot extraction.

Combines density + average speed into a 0..1 congestion index, and uses a
grid-cell binning over the heatmap to identify top-K hotspots."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("congestion")


@dataclass
class Hotspot:
    x: int
    y: int
    w: int
    h: int
    intensity: float

    def as_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "intensity": self.intensity}


class CongestionDetector:
    def __init__(
        self,
        density_threshold: float = 0.55,
        speed_threshold_kmh: float = 8.0,
        window_seconds: int = 10,
        fps: float = 25.0,
        grid: Tuple[int, int] = (8, 8),
        top_k: int = 3,
    ):
        self.density_threshold = float(density_threshold)
        self.speed_threshold = float(speed_threshold_kmh)
        self.grid = grid
        self.top_k = top_k
        self.window = max(1, int(window_seconds * fps))
        self._density_hist = deque(maxlen=self.window)
        self._speed_hist = deque(maxlen=self.window)

    def update(self, density: float, avg_speed: Optional[float]) -> Dict[str, float]:
        self._density_hist.append(float(density))
        if avg_speed is not None:
            self._speed_hist.append(float(avg_speed))
        d = float(np.mean(self._density_hist))
        s = float(np.mean(self._speed_hist)) if self._speed_hist else 60.0
        d_score = min(1.0, d / max(1e-6, self.density_threshold))
        s_score = max(0.0, 1.0 - s / max(1e-6, self.speed_threshold * 3))
        index = float(np.clip(0.6 * d_score + 0.4 * s_score, 0.0, 1.0))
        congested = bool(d >= self.density_threshold and s <= self.speed_threshold)
        return {"index": index, "density": d, "avg_speed": s, "congested": congested}

    def hotspots(self, heatmap: np.ndarray) -> List[Hotspot]:
        h, w = heatmap.shape[:2]
        gh, gw = self.grid
        cell_h, cell_w = h // gh, w // gw
        cells = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                cells[i, j] = heatmap[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w].mean()
        if cells.max() <= 0:
            return []
        flat = cells.flatten()
        top_idx = np.argsort(flat)[::-1][: self.top_k]
        hots: List[Hotspot] = []
        for idx in top_idx:
            i, j = divmod(int(idx), gw)
            hots.append(
                Hotspot(
                    x=j * cell_w,
                    y=i * cell_h,
                    w=cell_w,
                    h=cell_h,
                    intensity=float(cells[i, j] / cells.max()),
                )
            )
        return hots
