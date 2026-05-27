"""Drawing helpers for detections, tracks, lanes, lines and heatmaps."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

# Colour palette (BGR)
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def color_for_id(idx: int) -> Tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)]


def draw_detections(frame: np.ndarray, detections, class_names=None):
    """Draw raw detections (no tracking IDs)."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4]) if len(det) > 4 else 0.0
        cls = int(det[5]) if len(det) > 5 else 0
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        color = color_for_id(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_tracks(frame: np.ndarray, tracks, class_names=None, speeds=None):
    """Draw tracked vehicles with track ID and optional speed (km/h)."""
    for t in tracks:
        tid = int(t.get("id"))
        x1, y1, x2, y2 = map(int, t.get("bbox"))
        cls = int(t.get("cls", 0))
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        color = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        spd = speeds.get(tid) if speeds else None
        label = f"#{tid} {name}" + (f" {spd:.0f} km/h" if spd is not None else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_lanes(frame: np.ndarray, lanes: Sequence[dict], occupancy=None):
    overlay = frame.copy()
    for i, lane in enumerate(lanes):
        pts = np.array(lane["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color_for_id(i + 7))
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    for i, lane in enumerate(lanes):
        pts = np.array(lane["polygon"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color_for_id(i + 7), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        label = lane.get("id", f"lane_{i}")
        if occupancy is not None and label in occupancy:
            label = f"{label}: {occupancy[label]}"
        cv2.putText(frame, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_lines(frame: np.ndarray, lines: Sequence[dict], counts=None):
    for i, line in enumerate(lines):
        p1, p2 = line["points"]
        color = color_for_id(i)
        cv2.line(frame, tuple(p1), tuple(p2), color, 3)
        text = line.get("id", f"line_{i}")
        if counts is not None and text in counts:
            c = counts[text]
            text = f"{text}  in={c.get("in",0)} out={c.get("out",0)}"
        cv2.putText(frame, text, tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET):
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, colormap)
    return cv2.addWeighted(frame, 1 - alpha, coloured, alpha, 0)
