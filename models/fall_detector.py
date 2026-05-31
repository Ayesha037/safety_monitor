from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict
from loguru import logger

@dataclass
class _PersonState:
    centroid_y_history:    deque = field(default_factory=lambda: deque(maxlen=10))
    aspect_ratio_history:  deque = field(default_factory=lambda: deque(maxlen=5))

    def push(self, cy: int, aspect_ratio: float):
        self.centroid_y_history.append(cy)
        self.aspect_ratio_history.append(aspect_ratio)

    @property
    def mean_aspect(self):
        if not self.aspect_ratio_history:
            return 0.0
        return sum(self.aspect_ratio_history) / len(self.aspect_ratio_history)

    def vertical_drop(self, window: int) -> int:
        hist = list(self.centroid_y_history)
        if len(hist) < window + 1:
            return 0
        return hist[-1] - hist[-window - 1]


class FallDetector:
    def __init__(
        self,
        aspect_ratio_threshold: float = 1.4,
        velocity_frames: int = 5,
        velocity_threshold: int = 80,
        min_area_fraction: float = 0.005,
    ):
        self.ar_threshold       = aspect_ratio_threshold
        self.velocity_frames    = velocity_frames
        self.velocity_threshold = velocity_threshold
        self.min_area_fraction  = min_area_fraction
        self._state: Dict[int, _PersonState] = {}

    def update(self, person_id, x1, y1, x2, y2, frame_w, frame_h) -> bool:
        bbox_area  = (x2 - x1) * (y2 - y1)
        frame_area = frame_w * frame_h
        if bbox_area / max(frame_area, 1) < self.min_area_fraction:
            return False

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w  = x2 - x1
        h  = max(y2 - y1, 1)
        ar = w / h

        if person_id not in self._state:
            self._state[person_id] = _PersonState()

        state = self._state[person_id]
        state.push(cy, ar)

        return self._check_aspect_ratio(state) or self._check_velocity(state)

    def remove_person(self, person_id):
        self._state.pop(person_id, None)

    def clear(self):
        self._state.clear()

    def _check_aspect_ratio(self, state) -> bool:
        if len(state.aspect_ratio_history) < 3:
            return False
        recent = list(state.aspect_ratio_history)[-3:]
        return all(ar > self.ar_threshold for ar in recent)

    def _check_velocity(self, state) -> bool:
        return state.vertical_drop(self.velocity_frames) > self.velocity_threshold