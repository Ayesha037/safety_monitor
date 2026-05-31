from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config              import cfg
from utils.logger        import setup_logger
from utils.alert_manager import AlertManager
from utils.zone_manager  import ZoneManager
from utils.video_capture import VideoCapture
from utils.drawing       import (
    FPSCounter, draw_person, draw_restricted_zone,
    draw_fps, draw_frame_counter, draw_alert_banner, draw_stats_panel,
)
from models.detector        import ObjectDetector
from models.ppe_classifier  import PPEClassifier
from models.fall_detector   import FallDetector
from models.safety_pipeline import SafetyPipeline
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="Industrial Safety Monitor")
    p.add_argument("--source",     default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--record",     action="store_true")
    return p.parse_args()


def render_frame(frame, result, zone_mgr, fps_counter, show_fps=True):
    annotated = frame.copy()
    if zone_mgr.enabled:
        triggered_names = {z.name for p in result.persons for z in p.intruding_zones}
        for zone in zone_mgr.zones:
            draw_restricted_zone(
                annotated, zone.points_px, zone.name,
                color=zone.color, triggered=(zone.name in triggered_names),
            )
    for pa in result.persons:
        det = pa.detection
        draw_person(annotated, det.x1, det.y1, det.x2, det.y2,
                    person_id=pa.person_id, status=pa.status,
                    details=pa.violation_labels)
    if result.active_alerts:
        draw_alert_banner(annotated, result.active_alerts)
    draw_stats_panel(annotated, result.n_persons,
                     result.n_violations, result.n_zone_intrusions)
    if show_fps:
        draw_fps(annotated, fps_counter.tick())
    draw_frame_counter(annotated, result.frame_id)
    return annotated


def main():
    args = parse_args()
    setup_logger(log_level=cfg.system.log_level, log_file="logs/safety_monitor.log")
    logger.info("=== {} v{} starting ===", cfg.system.name, cfg.system.version)

    source = args.source if args.source is not None else cfg.video.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    detector = ObjectDetector(
        model_path=cfg.model.detector, device=cfg.model.device,
        conf_threshold=cfg.model.confidence_threshold,
        iou_threshold=cfg.model.nms_iou_threshold,
        max_detections=cfg.model.max_detections,
        target_classes=[cfg.ppe.person_class_id],
    )
    ppe      = PPEClassifier(model_path=cfg.ppe.ppe_model_path,
                             conf_threshold=cfg.model.confidence_threshold)
    fall     = FallDetector(
        aspect_ratio_threshold=cfg.fall_detection.aspect_ratio_threshold,
        velocity_frames=cfg.fall_detection.velocity_frames,
        velocity_threshold=cfg.fall_detection.velocity_threshold,
        min_area_fraction=cfg.fall_detection.min_area_fraction,
    )
    zone_mgr  = ZoneManager(cfg.zones)
    alert_mgr = AlertManager(cooldown=cfg.alerts.cooldown_seconds,
                              log_path=cfg.alerts.log_path)
    pipeline  = SafetyPipeline(
        detector=detector, ppe=ppe, fall=fall,
        zone_mgr=zone_mgr, alert_mgr=alert_mgr,
        resize_factor=cfg.video.resize_factor,
        frame_skip=cfg.video.frame_skip,
    )

    cap = VideoCapture(source=source, width=cfg.video.width,
                       height=cfg.video.height, fps_target=cfg.video.fps_target)
    cap.start()

    w, h = cap.frame_size
    if w > 0 and h > 0:
        zone_mgr.resolve_all(w, h)

    writer = None
    if args.record or cfg.recording.enabled:
        out_path = Path(cfg.recording.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*cfg.recording.codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, 20, (w, h))

    fps_counter = FPSCounter(window=30)
    logger.info("System ready. Press Q to quit, S to screenshot.")

    try:
        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            result    = pipeline.process(frame)
            annotated = render_frame(frame, result, zone_mgr, fps_counter,
                                     show_fps=cfg.display.show_fps)
            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Safety Monitor — Press Q to quit", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    path = f"logs/screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(path, annotated)
                    logger.info("Screenshot saved: {}", path)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("System shut down.")


if __name__ == "__main__":
    main()