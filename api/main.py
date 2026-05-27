"""FastAPI entrypoint for YOLOv8 Traffic Monitoring System."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import TrafficPipeline, load_config
from src.utils.logger import get_logger

from . import routes, websocket

log = get_logger("api")

CONFIG_PATH = os.getenv("APP_CONFIG", "config/config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading config from {}", CONFIG_PATH)
    cfg = load_config(CONFIG_PATH)
    app.state.config = cfg
    app.state.pipeline = None  # Lazy initialised
    log.info("FastAPI app starting")
    yield
    log.info("FastAPI app shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="YOLOv8 Traffic Monitoring System",
        description="Real-time vehicle detection, tracking, and traffic analytics.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)
    app.include_router(websocket.router)
    return app


app = create_app()
