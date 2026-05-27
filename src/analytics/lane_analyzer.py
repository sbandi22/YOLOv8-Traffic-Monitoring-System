"""Per-lane analytics: occupancy and average speed."""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np


class LaneAnalyzer:
    def __init__(self, lanes: List[dict]):
        self.lanes = lanes
        self._polys = {l["id"]: np.array(l["polygon"], dtype=np.int32) for l in lanes}

    def _point_in_lane(self, pt, lane_id) -> bool:
        return cv2.pointPolygonTest(self._polys[lane_id], (float(pt[0]), float(pt[1])), False) >= 0

    def analyze(self, tracks: Sequence[dict], speeds: Dict[int, float] | None = None) -> Dict[str, dict]:
        out: Dict[str, dict] = {lid: {"count": 0, "avg_speed": 0.0, "track_ids": []} for lid in self._polys}
        speed_acc: Dict[str, List[float]] = {lid: [] for lid in self._polys}
        for tr in tracks:
            cx, cy = tr["centroid"]
            for lid in self._polys:
                if self._point_in_lane((cx, cy), lid):
                    out[lid]["count"] += 1
                    out[lid]["track_ids"].append(int(tr["id"]))
                    if speeds is not None and tr["id"] in speeds:
                        speed_acc[lid].append(float(speeds[tr["id"]]))
                    break
        for lid in self._polys:
            if speed_acc[lid]:
                out[lid]["avg_speed"] = float(np.mean(speed_acc[lid]))
        return out
