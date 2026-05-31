from __future__ import annotations
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from loguru import logger

class VideoCapture:
    def __init__(self, source=0, width=1280, height=720, fps_target=30):
        self.source  = source
        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = None
        self._open(width, height, fps_target)

    @property
    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_size(self):
        if not self.is_opened:
            return (0, 0)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def fps(self):
        if not self.is_opened:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def _open(self, width, height, fps_target):
        src = self.source
        if isinstance(src, str) and Path(src).exists():
            self._cap = cv2.VideoCapture(src)
        else:
            self._cap = cv2.VideoCapture(int(src))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps_target)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

    def _reader_loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame