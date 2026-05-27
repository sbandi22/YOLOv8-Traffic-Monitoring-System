"""WebSocket streaming of annotated frames + analytics."""
from __future__ import annotations

import asyncio
import base64
import json

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.pipeline import TrafficPipeline
from src.utils.logger import get_logger
from src.video import VideoStream

log = get_logger("api.ws")
router = APIRouter()


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    cfg = websocket.app.state.config
    pipe = websocket.app.state.pipeline or TrafficPipeline(cfg)
    websocket.app.state.pipeline = pipe
    jpeg_q = int(cfg.get("api", {}).get("websocket_jpeg_quality", 70))

    log.info("WebSocket client connected")
    try:
        with VideoStream(
            cfg["video"]["source"],
            resize=(cfg["video"]["resize_width"], cfg["video"]["resize_height"]),
            target_fps=cfg["video"].get("target_fps"),
            loop=True,
        ) as stream_obj:
            while True:
                frame = stream_obj.read(timeout=2.0)
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue
                result = pipe.step(frame)
                ok, buf = cv2.imencode(".jpg", result["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                if not ok:
                    continue
                payload = {
                    "frame_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "analytics": result["analytics"],
                }
                await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:  # pragma: no cover
        log.exception("WebSocket error: {}", e)
        await websocket.close()
