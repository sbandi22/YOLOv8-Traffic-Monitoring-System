"""REST endpoints for the YOLOv8 Traffic Monitoring API."""
from __future__ import annotations

import io
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src.detection import YOLOv8Detector
from src.pipeline import TrafficPipeline
from src.utils.logger import get_logger

from .schemas import (
    AnalyticsSummary,
    BoundingBox,
    DetectionResponse,
    HealthResponse,
    HotspotItem,
)

log = get_logger("api.routes")
router = APIRouter()

_detector_cache: dict = {}


def _get_pipeline(request: Request) -> TrafficPipeline:
    if request.app.state.pipeline is None:
        log.info("Initialising shared pipeline")
        request.app.state.pipeline = TrafficPipeline(request.app.state.config)
    return request.app.state.pipeline


def _get_detector(request: Request) -> YOLOv8Detector:
    if "d" not in _detector_cache:
        cfg = request.app.state.config["detector"]
        _detector_cache["d"] = YOLOv8Detector(
            weights=cfg["weights"], device=cfg["device"], imgsz=cfg["imgsz"],
            conf=cfg["conf"], iou=cfg["iou"], half=cfg.get("half", False),
            classes=cfg.get("classes"), class_names=cfg.get("class_names"),
        )
    return _detector_cache["d"]


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    cfg = request.app.state.config
    return HealthResponse(
        status="ok",
        version=cfg["app"]["version"],
        device=cfg["detector"]["device"],
    )


@router.post("/detect/image", response_model=DetectionResponse, tags=["detection"])
async def detect_image(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    detector = _get_detector(request)
    t0 = time.time()
    dets = detector.predict(img)
    elapsed_ms = (time.time() - t0) * 1000.0

    names = request.app.state.config["detector"].get("class_names", {})
    bbs = [
        BoundingBox(
            x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
            conf=d.conf, cls=d.cls, name=names.get(d.cls),
        )
        for d in dets
    ]
    return DetectionResponse(detections=bbs, inference_ms=elapsed_ms, image_shape=list(img.shape))


@router.post("/process/video", tags=["processing"])
async def process_video(request: Request, file: UploadFile = File(...), max_frames: int = 600):
    tmp_path = f"/tmp/{int(time.time())}_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    cfg = dict(request.app.state.config)
    cfg["video"] = dict(cfg["video"])
    cfg["video"]["source"] = tmp_path
    pipe = TrafficPipeline(cfg)
    report = pipe.run(max_frames=max_frames)
    return JSONResponse(report)


@router.get("/analytics/summary", response_model=AnalyticsSummary, tags=["analytics"])
async def analytics_summary(request: Request) -> AnalyticsSummary:
    pipe = _get_pipeline(request)
    return AnalyticsSummary(**pipe.report())


@router.get("/analytics/heatmap", tags=["analytics"])
async def analytics_heatmap(request: Request):
    pipe = _get_pipeline(request)
    img = pipe.heatmap.render()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Could not encode heatmap")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/congestion/hotspots", tags=["analytics"])
async def congestion_hotspots(request: Request):
    pipe = _get_pipeline(request)
    return [HotspotItem(**h.as_dict()).model_dump() for h in pipe.congestion.hotspots(pipe.heatmap.raw)]
