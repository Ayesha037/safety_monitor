from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

@dataclass
class PPEResult:
    has_helmet: bool
    has_vest: bool
    helmet_confidence: float = 0.0
    vest_confidence: float   = 0.0

    @property
    def is_compliant(self):
        return self.has_helmet and self.has_vest

    @property
    def violations(self):
        v = []
        if not self.has_helmet:
            v.append("No Helmet")
        if not self.has_vest:
            v.append("No Safety Vest")
        return v


class PPEClassifier:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.50):
        self.conf_threshold = conf_threshold
        self._model = None
        self._mode  = "heuristic"

        if model_path and _YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_path)
                self._mode  = "model"
                logger.info("PPE model loaded: {}", model_path)
            except Exception as exc:
                logger.warning("PPE model load failed ({}); using heuristics.", exc)

        logger.info("PPE Classifier mode: {}", self._mode)

    def check(self, frame: np.ndarray, x1, y1, x2, y2) -> PPEResult:
        h_frame, w_frame = frame.shape[:2]
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, w_frame - 1); y2 = min(y2, h_frame - 1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PPEResult(has_helmet=False, has_vest=False)
        if self._mode == "model":
            return self._check_with_model(crop)
        return self._check_heuristic(crop)

    def _check_with_model(self, crop):
        results = self._model.predict(crop, conf=self.conf_threshold, verbose=False)
        has_helmet = has_vest = False
        hc = vc = 0.0
        for result in results:
            for box in (result.boxes or []):
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0:
                    has_helmet = True; hc = max(hc, conf)
                elif cls == 1:
                    has_vest = True;   vc = max(vc, conf)
        return PPEResult(has_helmet=has_helmet, has_vest=has_vest,
                         helmet_confidence=hc, vest_confidence=vc)

    def _check_heuristic(self, crop):
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return PPEResult(has_helmet=False, has_vest=False)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        helmet_roi = hsv[:h // 4, :]
        hc, has_helmet = self._detect_hardhat_color(helmet_roi)
        vest_roi = hsv[h // 4: 3 * h // 4, :]
        vc, has_vest = self._detect_vest_color(vest_roi)
        return PPEResult(has_helmet=has_helmet, has_vest=has_vest,
                         helmet_confidence=hc, vest_confidence=vc)

    @staticmethod
    def _detect_hardhat_color(roi_hsv):
        total = roi_hsv.shape[0] * roi_hsv.shape[1]
        if total == 0:
            return 0.0, False
        m_yellow = cv2.inRange(roi_hsv, (20, 100, 100), (35, 255, 255))
        m_orange = cv2.inRange(roi_hsv, (5,  100, 100), (20, 255, 255))
        m_white  = cv2.inRange(roi_hsv, (0,   0,  200), (180, 40, 255))
        m_red1   = cv2.inRange(roi_hsv, (0,  100, 100), (5,  255, 255))
        m_red2   = cv2.inRange(roi_hsv, (175,100, 100), (180,255, 255))
        combined = m_yellow | m_orange | m_white | m_red1 | m_red2
        ratio = cv2.countNonZero(combined) / total
        return float(ratio), ratio > 0.20

    @staticmethod
    def _detect_vest_color(roi_hsv):
        total = roi_hsv.shape[0] * roi_hsv.shape[1]
        if total == 0:
            return 0.0, False
        m_yellow = cv2.inRange(roi_hsv, (25, 150, 150), (40, 255, 255))
        m_orange = cv2.inRange(roi_hsv, (10, 150, 150), (25, 255, 255))
        combined = m_yellow | m_orange
        ratio = cv2.countNonZero(combined) / total
        return float(ratio), ratio > 0.15