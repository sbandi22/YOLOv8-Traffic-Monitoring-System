"""Unit tests for the VehicleTracker IoU associator."""
import numpy as np

from src.tracking import VehicleTracker


def _det(x1, y1, x2, y2, cls=2, conf=0.9):
    return np.array([x1, y1, x2, y2, conf, cls], dtype=np.float32)


def test_tracker_assigns_ids():
    tr = VehicleTracker(max_age=5, min_hits=1, iou_threshold=0.2)
    tracks = tr.update([_det(0, 0, 50, 50), _det(100, 100, 160, 160)])
    assert {t.id for t in tracks} == {1, 2}


def test_tracker_keeps_id_across_frames():
    tr = VehicleTracker(max_age=5, min_hits=1, iou_threshold=0.2)
    tr.update([_det(0, 0, 50, 50)])
    tracks2 = tr.update([_det(5, 5, 55, 55)])
    assert tracks2[0].id == 1
    assert tracks2[0].hits == 2


def test_tracker_removes_stale():
    tr = VehicleTracker(max_age=1, min_hits=1, iou_threshold=0.2)
    tr.update([_det(0, 0, 50, 50)])
    tr.update([])
    final = tr.update([])
    assert all(t.id != 1 for t in final)
