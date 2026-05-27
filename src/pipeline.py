"""End-to-end real-time pipeline tying detection, tracking, analytics and congestion."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from src.analytics import (
    DensityEstimator,
    HeatmapGenerator,
    LaneAnalyzer,
    MultiLineCounter,
    SpeedEstimator,
)
from src.congestion import CongestionDetector
from src.detection import YOLOv8Detector
from src.tracking import VehicleTracker
from src.utils.logger import get_logger, configure_logger
from src.utils.visualization import (
    draw_lanes,
    draw_lines,
    draw_tracks,
    overlay_heatmap,
)
from src.video import VideoStream

log = get_logger("pipeline")


@dataclass
class FrameAnalytics:
    frame_idx: int
    timestamp: float
    fps: float
    detections: int
    tracks: int
    density: float
    avg_speed: float
    congestion: Dict[str, float]
    counts: Dict[str, Any]
    lanes: Dict[str, Any]
    hotspots: List[Dict[str, Any]] = field(default_factory=list)


class TrafficPipeline:
    def __init__(self, config: dict):
        self.cfg = config
        cfg_log = config.get("logging", {})
        configure_logger(
            log_file=cfg_log.get("file", "logs/traffic.log"),
            level=cfg_log.get("level", "INFO"),
            rotation=cfg_log.get("rotation", "20 MB"),
            retention=cfg_log.get("retention", "14 days"),
            json_logs=cfg_log.get("json", True),
        )

        d = config["detector"]
        self.detector = YOLOv8Detector(
            weights=d["weights"], device=d["device"], imgsz=d["imgsz"],
            conf=d["conf"], iou=d["iou"], half=d.get("half", False),
            classes=d.get("classes"), class_names=d.get("class_names"),
        )

        t = config["tracker"]
        self.tracker = VehicleTracker(
            max_age=t["max_age"], min_hits=t["min_hits"], iou_threshold=t["iou_threshold"],
        )

        v = config["video"]
        self.video = VideoStream(
            v["source"],
            buffer_size=v.get("buffer_size", 8),
            resize=(v["resize_width"], v["resize_height"]),
            target_fps=v.get("target_fps"),
            loop=v.get("loop", False),
        )
        self.frame_shape = (v["resize_height"], v["resize_width"])

        roi = config.get("roi", {})
        self.density = DensityEstimator(
            roi_polygon=roi.get("polygon") if roi.get("enabled") else None,
            frame_shape=self.frame_shape,
        )

        self.counter = MultiLineCounter(config.get("counting_lines", []))
        self.lane = LaneAnalyzer(config.get("lanes", []))

        sp = config.get("speed", {})
        H = None
        if sp.get("homography", {}).get("enabled"):
            ip = np.array(sp["homography"]["image_points"], dtype=np.float32)
            wp = np.array(sp["homography"]["world_points"], dtype=np.float32)
            H, _ = cv2.findHomography(ip, wp)
        self.speed = SpeedEstimator(
            fps=v.get("target_fps", 25),
            pixels_per_meter=sp.get("pixels_per_meter", 8.0),
            smoothing_window=sp.get("smoothing_window", 5),
            homography=H,
        )

        hm = config.get("heatmap", {})
        self.heatmap = HeatmapGenerator(
            frame_shape=self.frame_shape,
            decay=hm.get("decay", 0.985),
            blur_kernel=hm.get("blur_kernel", 31),
        )

        cg = config.get("congestion", {})
        self.congestion = CongestionDetector(
            density_threshold=cg.get("density_threshold", 0.55),
            speed_threshold_kmh=cg.get("avg_speed_threshold", 8.0),
            window_seconds=cg.get("window_seconds", 10),
            fps=v.get("target_fps", 25),
            grid=tuple(cg.get("hotspot_grid", [8, 8])),
            top_k=cg.get("hotspot_top_k", 3),
        )

        self.frame_idx = 0
        self._start_time = None
        self._frame_analytics: List[FrameAnalytics] = []

    def _annotate(self, frame, tracks, counts, speeds, heat):
        frame = overlay_heatmap(frame, heat, alpha=self.cfg.get("heatmap", {}).get("alpha", 0.5))
        frame = draw_lanes(frame, self.cfg.get("lanes", []))
        frame = draw_lines(frame, self.cfg.get("counting_lines", []), counts=counts)
        frame = draw_tracks(
            frame, tracks,
            class_names=self.cfg["detector"].get("class_names"),
            speeds=speeds,
        )
        return frame

    def step(self, frame: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        dets = self.detector.predict(frame)
        det_arrays = [d.as_array() for d in dets]
        tracks = self.tracker.update(det_arrays)
        track_dicts = [
            {"id": tr.id, "bbox": tr.bbox, "cls": tr.cls, "conf": tr.conf, "centroid": tr.centroid}
            for tr in tracks
        ]

        density = self.density.update(track_dicts)
        self.counter.update(track_dicts)
        speeds = self.speed.update(track_dicts)
        avg_speed = float(np.mean(list(speeds.values()))) if speeds else 0.0
        lane_stats = self.lane.analyze(track_dicts, speeds)
        self.heatmap.update(track_dicts)
        cong = self.congestion.update(density, avg_speed if speeds else None)
        hotspots = [h.as_dict() for h in self.congestion.hotspots(self.heatmap.raw)]

        annotated = self._annotate(
            frame.copy(), track_dicts, self.counter.summary(), speeds, self.heatmap.raw,
        )

        elapsed = time.time() - t0
        fa = FrameAnalytics(
            frame_idx=self.frame_idx,
            timestamp=time.time(),
            fps=1.0 / elapsed if elapsed > 0 else 0.0,
            detections=len(det_arrays),
            tracks=len(track_dicts),
            density=density,
            avg_speed=avg_speed,
            congestion=cong,
            counts=self.counter.summary(),
            lanes=lane_stats,
            hotspots=hotspots,
        )
        self._frame_analytics.append(fa)
        self.frame_idx += 1
        return {"frame": annotated, "analytics": asdict(fa)}

    def run(self, output_path: Optional[str] = None, max_frames: Optional[int] = None):
        writer = None
        with self.video as stream:
            self._start_time = time.time()
            while True:
                if max_frames and self.frame_idx >= max_frames:
                    break
                frame = stream.read(timeout=2.0)
                if frame is None:
                    break
                result = self.step(frame)
                if output_path:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        h, w = result["frame"].shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, self.video.fps, (w, h))
                    writer.write(result["frame"])
        if writer is not None:
            writer.release()
        log.info("Pipeline finished after {} frames in {:.1f}s", self.frame_idx, time.time() - self._start_time)
        return self.report()

    def report(self) -> Dict[str, Any]:
        if not self._frame_analytics:
            return {}
        last = self._frame_analytics[-1]
        return {
            "frames": self.frame_idx,
            "final_counts": last.counts,
            "avg_density": float(np.mean([f.density for f in self._frame_analytics])),
            "avg_speed": float(np.mean([f.avg_speed for f in self._frame_analytics if f.avg_speed > 0]) or 0.0),
            "peak_congestion": float(max((f.congestion["index"] for f in self._frame_analytics), default=0.0)),
            "hotspots": last.hotspots,
            "lanes": last.lanes,
        }


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
