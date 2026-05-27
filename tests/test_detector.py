"""Unit tests for the YOLOv8 detector wrapper.

These tests stub out Ultralytics so they can run without the heavy model
being installed (CI-friendly)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_yolo():
    with patch("src.detection.yolov8_detector.YOLO") as YOLO_mock:
        instance = MagicMock()
        instance.model.names = {2: "car", 7: "truck"}

        boxes = MagicMock()
        boxes.__len__ = MagicMock(return_value=2)
        boxes.xyxy = MagicMock()
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 60, 90], [100, 100, 200, 250]], dtype=np.float32)
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92, 0.81], dtype=np.float32)
        boxes.cls = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = np.array([2, 7], dtype=np.float32)

        result = MagicMock(); result.boxes = boxes
        instance.predict.return_value = [result]
        YOLO_mock.return_value = instance
        yield YOLO_mock


def test_detector_predict_returns_detections(fake_yolo):
    from src.detection import YOLOv8Detector

    det = YOLOv8Detector(weights="yolov8n.pt", device="cpu", half=False, classes=[2, 7])
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.predict(img)
    assert len(out) == 2
    assert out[0].cls == 2 and out[1].cls == 7
    assert out[0].conf > 0.9


def test_detection_as_array(fake_yolo):
    from src.detection import YOLOv8Detector
    det = YOLOv8Detector(weights="yolov8n.pt", device="cpu", half=False)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.predict(img)
    arr = out[0].as_array()
    assert arr.shape == (6,)
