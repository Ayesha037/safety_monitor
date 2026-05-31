from __future__ import annotations
import time
from collections import deque
from typing import List, Tuple
import cv2
import numpy as np

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
COLOR_GREEN  = (0, 200, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_GRAY   = (120, 120, 120)

class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(elapsed, 1e-6)

    @property
    def fps(self):
        return self.tick()

def draw_bbox(frame, x1, y1, x2, y2, label, confidence, color=COLOR_GREEN, thickness=2, font_scale=0.55):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    chip_y1 = max(y1 - th - baseline - 6, 0)
    chip_y2 = max(y1, th + baseline + 6)
    cv2.rectangle(frame, (x1, chip_y1), (x1 + tw + 8, chip_y2), color, -1)
    cv2.putText(frame, text, (x1 + 4, chip_y2 - baseline - 2),
                FONT, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_person(frame, x1, y1, x2, y2, person_id, status, details, font_scale=0.55):
    color_map = {"safe": COLOR_GREEN, "warning": COLOR_ORANGE, "critical": COLOR_RED}
    color = color_map.get(status, COLOR_GRAY)
    thickness = 3 if status == "critical" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    id_text = f"P{person_id:02d}"
    (tw, th), bl = cv2.getTextSize(id_text, FONT_BOLD, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, id_text, (x1 + 4, y1 - bl - 2),
                FONT_BOLD, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)
    for i, detail in enumerate(details):
        dy = y2 + 4 + i * int(th * 1.8)
        cv2.putText(frame, f"! {detail}", (x1, dy + th),
                    FONT, font_scale * 0.9, color, 1, cv2.LINE_AA)

def draw_restricted_zone(frame, points, label, color=(0,0,220), alpha=0.18, triggered=False):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    n = len(points)
    for i in range(n):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % n])
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    cx = int(points[:, 0].mean())
    cy = int(points[:, 1].mean())
    label_text = f"RESTRICTED: {label}"
    (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.55, 1)
    cv2.putText(frame, label_text, (cx - tw//2, cy + th//2),
                FONT, 0.55, color, 1, cv2.LINE_AA)

def draw_fps(frame, fps, pos=(10, 28)):
    text  = f"FPS: {fps:.1f}"
    color = COLOR_GREEN if fps >= 20 else (COLOR_ORANGE if fps >= 10 else COLOR_RED)
    cv2.putText(frame, text, pos, FONT_BOLD, 0.7, COLOR_BLACK, 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT_BOLD, 0.7, color, 1, cv2.LINE_AA)

def draw_frame_counter(frame, frame_id):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Frame #{frame_id:06d}", (w - 160, 22),
                FONT, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

def draw_alert_banner(frame, alerts, max_alerts=3):
    if not alerts:
        return
    h, w = frame.shape[:2]
    banner_h = 28 + min(len(alerts), max_alerts) * 22
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "SAFETY ALERT", (10, 20),
                FONT_BOLD, 0.65, COLOR_YELLOW, 1, cv2.LINE_AA)
    for i, msg in enumerate(alerts[:max_alerts]):
        cv2.putText(frame, f"  {msg}", (10, 42 + i * 22),
                    FONT, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_stats_panel(frame, n_persons, n_violations, n_zones_triggered):
    h, w = frame.shape[:2]
    panel_w, panel_h = 220, 78
    x0, y0 = 8, h - panel_h - 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_GRAY, 1)
    lines = [
        (f"Persons detected : {n_persons}",         COLOR_WHITE),
        (f"PPE violations   : {n_violations}",      COLOR_ORANGE if n_violations else COLOR_GREEN),
        (f"Zone intrusions  : {n_zones_triggered}", COLOR_RED    if n_zones_triggered else COLOR_GREEN),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x0 + 6, y0 + 18 + i * 22),
                    FONT, 0.46, color, 1, cv2.LINE_AA)