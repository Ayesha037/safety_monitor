from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from loguru import logger

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed.")

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        return self.width / max(self.height, 1)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def feet(self):
        return (self.center[0], self.y2)

    def is_person(self):
        return self.class_id == 0


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes=None,
        max_detections: int = 50,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.max_detections = max_detections
        self.model = None
        self.device = self._resolve_device(device)

        if _YOLO_AVAILABLE:
            self._load_model(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
            classes=self.target_classes,
        )
        return self._parse_results(results)

    def _load_model(self, model_path: str):
        logger.info("Loading detector: {} → device={}", model_path, self.device)
        try:
            self.model = YOLO(model_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=self.device)
            logger.info("Detector ready.")
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)
            self.model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].cpu().numpy().astype(int)
                name   = result.names.get(cls_id, str(cls_id))
                detections.append(Detection(
                    class_id=cls_id, class_name=name, confidence=conf,
                    x1=int(xyxy[0]), y1=int(xyxy[1]),
                    x2=int(xyxy[2]), y2=int(xyxy[3]),
                ))
        return detections