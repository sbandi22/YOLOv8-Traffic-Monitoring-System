"""Unit tests for analytics + congestion modules."""
import numpy as np

from src.analytics import (
    DensityEstimator,
    HeatmapGenerator,
    LaneAnalyzer,
    MultiLineCounter,
    SpeedEstimator,
)
from src.congestion import CongestionDetector


def _track(tid, bbox, cls=2):
    x1, y1, x2, y2 = bbox
    return {
        "id": tid, "bbox": bbox, "cls": cls, "conf": 0.9,
        "centroid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    }


def test_line_counter_counts_crossings():
    counter = MultiLineCounter([{"id": "l", "points": [[0, 100], [200, 100]]}])
    counter.update([_track(1, (95, 50, 105, 60))])
    counter.update([_track(1, (95, 150, 105, 160))])
    s = counter.summary()["l"]
    assert s["out"] + s["in"] == 1


def test_density_estimator_in_range():
    est = DensityEstimator(frame_shape=(720, 1280))
    ratio = est.update([_track(1, (0, 0, 640, 360))])
    assert 0.0 < ratio <= 1.0


def test_lane_analyzer_assigns():
    lanes = [{"id": "L1", "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]]}]
    la = LaneAnalyzer(lanes)
    stats = la.analyze([_track(1, (50, 50, 80, 80))])
    assert stats["L1"]["count"] == 1


def test_speed_estimator_returns_kmh():
    se = SpeedEstimator(fps=25, pixels_per_meter=10.0)
    se.update([_track(1, (0, 0, 10, 10))])
    speeds = se.update([_track(1, (10, 0, 20, 10))])
    assert 1 in speeds and speeds[1] > 0


def test_heatmap_updates():
    hm = HeatmapGenerator(frame_shape=(100, 100), decay=1.0, blur_kernel=3)
    hm.update([_track(1, (10, 10, 30, 30))])
    assert hm.raw[20, 20] > 0


def test_congestion_index_range():
    cd = CongestionDetector()
    out = cd.update(density=0.8, avg_speed=5.0)
    assert 0.0 <= out["index"] <= 1.0
