"""FastAPI backend for the YOLOv8 Traffic Monitoring System."""

from .main import app, create_app

__all__ = ["app", "create_app"]
