"""Video stream handler with threaded buffering.

Accepts file paths, RTSP/HTTP URLs, or integer webcam indices."""
from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Optional, Tuple

import cv2

from src.utils.logger import get_logger

log = get_logger("video.stream")


class VideoStream:
    def __init__(
        self,
        source,
        buffer_size: int = 8,
        resize: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
        loop: bool = False,
    ):
        self.source = source
        self.resize = resize
        self.target_fps = float(target_fps) if target_fps else None
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Queue = Queue(maxsize=buffer_size)
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._native_fps = 0.0

    def _open(self):
        src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self._native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        log.info("Opened source={} native_fps={:.2f}", self.source, self._native_fps)
        return cap

    def start(self) -> "VideoStream":
        self._cap = self._open()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        delay = 1.0 / self.target_fps if self.target_fps else 0.0
        while not self._stopped.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._stopped.set()
                break
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            if self._queue.full():
                try: self._queue.get_nowait()
                except Empty: pass
            self._queue.put(frame)
            if delay:
                time.sleep(delay)

    def read(self, timeout: float = 1.0):
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def fps(self) -> float:
        return self.target_fps or self._native_fps

    def stop(self):
        self._stopped.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
