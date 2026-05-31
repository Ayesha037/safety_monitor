from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from models.detector       import ObjectDetector, Detection
from models.ppe_classifier import PPEClassifier, PPEResult
from models.fall_detector  import FallDetector
from utils.zone_manager    import ZoneManager, RestrictedZone
from utils.alert_manager   import AlertManager, Severity

@dataclass
class PersonAnalysis:
    person_id: int
    detection: Detection
    ppe: PPEResult
    is_falling: bool
    intruding_zones: List[RestrictedZone] = field(default_factory=list)

    @property
    def status(self):
        if self.is_falling or self.intruding_zones:
            return "critical"
        if not self.ppe.is_compliant:
            return "warning"
        return "safe"

    @property
    def violation_labels(self):
        labels = []
        labels.extend(self.ppe.violations)
        for z in self.intruding_zones:
            labels.append(f"Zone: {z.name}")
        if self.is_falling:
            labels.append("FALL DETECTED")
        return labels

@dataclass
class FrameResult:
    frame_id: int
    persons: List[PersonAnalysis] = field(default_factory=list)
    active_alerts: List[str]      = field(default_factory=list)

    @property
    def n_persons(self):
        return len(self.persons)

    @property
    def n_violations(self):
        return sum(1 for p in self.persons if not p.ppe.is_compliant)

    @property
    def n_zone_intrusions(self):
        return sum(1 for p in self.persons if p.intruding_zones)


class SafetyPipeline:
    def __init__(self, detector, ppe, fall, zone_mgr, alert_mgr,
                 resize_factor=0.75, frame_skip=2):
        self._detector    = detector
        self._ppe         = ppe
        self._fall        = fall
        self._zone_mgr    = zone_mgr
        self._alert_mgr   = alert_mgr
        self._resize_factor = resize_factor
        self._frame_skip  = max(1, frame_skip)
        self._frame_id    = 0
        self._skip_counter = 0
        self._last_result: Optional[FrameResult] = None
        self._person_counter = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        self._frame_id += 1
        self._skip_counter += 1

        if self._skip_counter < self._frame_skip and self._last_result is not None:
            return self._last_result
        self._skip_counter = 0

        h_orig, w_orig = frame.shape[:2]
        if self._resize_factor < 1.0:
            inf_frame = cv2.resize(
                frame,
                (int(w_orig * self._resize_factor), int(h_orig * self._resize_factor))
            )
            scale = 1.0 / self._resize_factor
        else:
            inf_frame = frame
            scale = 1.0

        if not self._zone_mgr._resolved:
            self._zone_mgr.resolve_all(w_orig, h_orig)

        detections = self._detector.detect(inf_frame)

        persons = []
        self._person_counter = 0

        for det in detections:
            if not det.is_person():
                continue
            self._person_counter += 1
            pid = self._person_counter

            x1 = int(det.x1 * scale); y1 = int(det.y1 * scale)
            x2 = int(det.x2 * scale); y2 = int(det.y2 * scale)
            det.x1, det.y1, det.x2, det.y2 = x1, y1, x2, y2

            ppe_result = self._ppe.check(frame, x1, y1, x2, y2)
            is_falling = self._fall.update(pid, x1, y1, x2, y2, w_orig, h_orig)
            intruding  = self._zone_mgr.check_person(x1, y1, x2, y2)

            pa = PersonAnalysis(person_id=pid, detection=det, ppe=ppe_result,
                                is_falling=is_falling, intruding_zones=intruding)
            persons.append(pa)
            self._fire_alerts(pa)

        active = [a.message for a in self._alert_mgr.get_active(max_age=8.0)]
        result = FrameResult(frame_id=self._frame_id, persons=persons, active_alerts=active)
        self._last_result = result
        return result

    def _fire_alerts(self, pa):
        fid = self._frame_id
        if pa.is_falling:
            self._alert_mgr.trigger(
                f"FALL_P{pa.person_id}",
                f"Person {pa.person_id} may have fallen!",
                Severity.CRITICAL, fid)
        for zone in pa.intruding_zones:
            self._alert_mgr.trigger(
                f"ZONE_{zone.name}_P{pa.person_id}",
                f"Person {pa.person_id} entered: {zone.name}",
                Severity.CRITICAL, fid)
        if not pa.ppe.is_compliant:
            for v in pa.ppe.violations:
                self._alert_mgr.trigger(
                    f"PPE_{v}_P{pa.person_id}",
                    f"Person {pa.person_id} — PPE violation: {v}",
                    Severity.WARNING, fid)