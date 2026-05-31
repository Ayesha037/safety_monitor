from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

@dataclass
class RestrictedZone:
    name: str
    color: Tuple[int, int, int]
    points_norm: List[List[float]]
    _points_px: np.ndarray = field(default=None, init=False, repr=False)

    def resolve(self, frame_w, frame_h):
        self._points_px = np.array(
            [[int(x * frame_w), int(y * frame_h)] for x, y in self.points_norm],
            dtype=np.int32,
        )

    @property
    def points_px(self):
        if self._points_px is None:
            raise RuntimeError("Call resolve(w, h) first.")
        return self._points_px

    def contains_point(self, x, y):
        return cv2.pointPolygonTest(self.points_px, (float(x), float(y)), False) >= 0

    def bbox_intrudes(self, x1, y1, x2, y2):
        feet_x = (x1 + x2) // 2
        feet_y = y2
        return self.contains_point(feet_x, feet_y)


class ZoneManager:
    def __init__(self, zone_cfg):
        self.enabled = zone_cfg.get("enabled", True)
        self.zones: List[RestrictedZone] = []
        self._resolved = False
        if not self.enabled:
            return
        for z in zone_cfg.get("restricted", []):
            self.zones.append(RestrictedZone(
                name=z["name"], color=tuple(z["color"]), points_norm=z["points"]
            ))

    def resolve_all(self, frame_w, frame_h):
        for zone in self.zones:
            zone.resolve(frame_w, frame_h)
        self._resolved = True

    def check_person(self, x1, y1, x2, y2):
        if not self.enabled or not self._resolved:
            return []
        return [z for z in self.zones if z.bbox_intrudes(x1, y1, x2, y2)]