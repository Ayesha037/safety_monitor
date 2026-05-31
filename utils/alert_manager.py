from __future__ import annotations
import csv
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Deque, List

from loguru import logger

class Severity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    timestamp: float
    severity: str
    event_type: str
    message: str
    frame_id: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def age_seconds(self):
        return time.time() - self.timestamp

    def to_dict(self):
        d = asdict(self)
        d["timestamp_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return d


class AlertManager:
    def __init__(self, cooldown=5.0, log_path="logs/alerts.csv", buffer_size=200):
        self.cooldown = cooldown
        self.log_path = Path(log_path)
        self._last_trigger: dict = {}
        self._buffer: Deque[Alert] = deque(maxlen=buffer_size)
        self._init_log_file()

    def trigger(self, event_type, message, severity=Severity.WARNING, frame_id=0, extra=None):
        now  = time.time()
        last = self._last_trigger.get(event_type, 0.0)
        if now - last < self.cooldown:
            return None

        alert = Alert(timestamp=now, severity=severity.value,
                      event_type=event_type, message=message,
                      frame_id=frame_id, extra=extra or {})
        self._last_trigger[event_type] = now
        self._buffer.append(alert)
        self._log_alert(alert)

        log_fn = {
            Severity.INFO.value:     logger.info,
            Severity.WARNING.value:  logger.warning,
            Severity.CRITICAL.value: logger.critical,
        }.get(alert.severity, logger.warning)
        log_fn("[ALERT][{}] {}", event_type, message)
        return alert

    def get_recent(self, n=20):
        return list(reversed(list(self._buffer)))[:n]

    def get_active(self, max_age=10.0):
        cutoff = time.time() - max_age
        return [a for a in self._buffer if a.timestamp >= cutoff]

    def clear(self):
        self._buffer.clear()
        self._last_trigger.clear()

    def _init_log_file(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp_str", "severity", "event_type", "message", "frame_id"])
                writer.writeheader()

    def _log_alert(self, alert):
        try:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp_str", "severity", "event_type", "message", "frame_id"])
                row = alert.to_dict()
                writer.writerow({k: row[k] for k in writer.fieldnames})
        except Exception as exc:
            logger.error("Failed to write alert: {}", exc)