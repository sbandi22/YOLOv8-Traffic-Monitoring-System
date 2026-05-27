"""YOLOv8 detection wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

from src.utils.logger import get_logger

log = get_logger("detector")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    def as_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2, self.conf, self.cls], dtype=np.float32)


class YOLOv8Detector:
    """Thin wrapper around Ultralytics YOLO with configurable filtering."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        half: bool = True,
        classes: Optional[Sequence[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. `pip install ultralytics`")

        log.info("Loading YOLOv8 weights from {}", weights)
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.classes = list(classes) if classes else None
        self.class_names = class_names or self.model.model.names

    def warmup(self, shape=(640, 640, 3)):
        dummy = np.zeros(shape, dtype=np.uint8)
        self.predict(dummy)
        log.debug("Detector warmup complete")

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR image and return a list of Detection."""
        results = self.model.predict(
            source=frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            classes=self.classes,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            detections.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        return detections

    def predict_batch(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        return [self.predict(f) for f in frames]
