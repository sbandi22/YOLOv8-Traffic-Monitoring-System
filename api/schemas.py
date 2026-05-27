"""Pydantic schemas for API responses."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    device: Optional[str] = None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    name: Optional[str] = None


class DetectionResponse(BaseModel):
    detections: List[BoundingBox]
    inference_ms: float
    image_shape: List[int]


class CountSummary(BaseModel):
    in_: int = Field(..., alias="in")
    out: int
    per_class: Dict[int, Dict[str, int]] = {}


class AnalyticsSummary(BaseModel):
    frames: int
    final_counts: Dict[str, Any]
    avg_density: float
    avg_speed: float
    peak_congestion: float
    hotspots: List[Dict[str, Any]]
    lanes: Dict[str, Any]


class HotspotItem(BaseModel):
    x: int
    y: int
    w: int
    h: int
    intensity: float
