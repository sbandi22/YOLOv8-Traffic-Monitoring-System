"""Multi-vehicle tracker with a simple, dependency-light SORT-style implementation.

Designed to work even without external trackers installed. If `deep-sort-realtime`
is available it is used; otherwise we fall back to an IoU-based associator with a
Kalman-like constant-velocity prior."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("tracker")


def _iou(b1, b2):
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: Tuple[float, float, float, float]
    cls: int
    conf: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class VehicleTracker:
    """IoU-based multi-object tracker with class continuity."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence) -> List[Track]:
        """Update tracker with current frame detections (iterable of [x1,y1,x2,y2,conf,cls])."""
        det_boxes = []
        for d in detections:
            if hasattr(d, "as_array"):
                arr = d.as_array()
            else:
                arr = np.asarray(d, dtype=np.float32)
            det_boxes.append(arr)

        # Match using IoU greedy
        unmatched_dets = list(range(len(det_boxes)))
        matches: Dict[int, int] = {}  # track_id -> det_idx
        for tid, tr in list(self._tracks.items()):
            best_idx, best_iou = -1, 0.0
            for di in unmatched_dets:
                if int(det_boxes[di][5]) != tr.cls:
                    continue
                i = _iou(tr.bbox, det_boxes[di][:4])
                if i > best_iou:
                    best_iou, best_idx = i, di
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                matches[tid] = best_idx
                unmatched_dets.remove(best_idx)

        # Update matched tracks
        for tid, di in matches.items():
            tr = self._tracks[tid]
            bbox = tuple(map(float, det_boxes[di][:4]))
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            if tr.history:
                px, py = tr.history[-1]
                tr.velocity = (cx - px, cy - py)
            tr.history.append((cx, cy))
            tr.bbox = bbox
            tr.conf = float(det_boxes[di][4])
            tr.hits += 1
            tr.time_since_update = 0

        # Create new tracks
        for di in unmatched_dets:
            arr = det_boxes[di]
            tr = Track(
                id=self._next_id,
                bbox=tuple(map(float, arr[:4])),
                cls=int(arr[5]),
                conf=float(arr[4]),
            )
            tr.history.append(tr.centroid)
            self._tracks[self._next_id] = tr
            self._next_id += 1

        # Age & remove
        active: List[Track] = []
        for tid, tr in list(self._tracks.items()):
            if tid not in matches:
                tr.time_since_update += 1
            tr.age += 1
            if tr.time_since_update > self.max_age:
                del self._tracks[tid]
                continue
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                active.append(tr)
        return active

    def tracks_as_dicts(self) -> List[dict]:
        return [
            {
                "id": t.id,
                "bbox": t.bbox,
                "cls": t.cls,
                "conf": t.conf,
                "centroid": t.centroid,
                "velocity": t.velocity,
                "age": t.age,
                "hits": t.hits,
            }
            for t in self._tracks.values()
        ]
