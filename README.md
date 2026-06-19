# Safety Monitor v1.3.0 🏭

Real-time industrial safety monitoring system using computer vision. Detects safety violations, PPE compliance, falls, and restricted zone intrusions in live video streams.

## Features

- 🎥 **Real-time Video Processing** - Webcam, IP cameras, RTSP streams, video files
- 👤 **Person Detection** - YOLOv8-based detection with confidence scoring
- 🦺 **PPE Detection** - Helmet, vest, gloves, masks, goggles
- 📍 **Zone Management** - Custom restricted areas with intrusion alerts
- 👥 **Fall Detection** - Automatic fall detection using motion & aspect ratio analysis
- 🔔 **Smart Alerts** - Configurable alerts with cooldown periods & logging
- 📹 **Video Recording** - Auto-record violations
- 📊 **Live Statistics** - FPS counter, frame count, violation tracking
- 🎨 **Visual Annotations** - Real-time frame annotations with status indicators

## Tech Stack

- **Python** 3.8+
- **OpenCV** - Video processing
- **PyTorch** - Deep learning
- **YOLOv8** - Object detection
- **Streamlit** - Dashboard
- **Loguru** - Logging

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Ayesha037/safety_monitor.git
cd safety_monitor
```

### 2. Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. GPU Support (Optional)
```bash
# For NVIDIA GPU (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage
```bash
# Webcam (default)
python main.py

# IP Camera / RTSP Stream
python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"

# Video File
python main.py --source "video.mp4"

# Headless Mode (no display)
python main.py --no-display

# Enable Recording
python main.py --record
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| **Q** | Quit |
| **S** | Screenshot |

## Configuration

Edit `config/config.yaml`:

```yaml
system:
  name: "Safety Monitor"
  version: "1.3.0"
  log_level: "INFO"

video:
  source: "0"                    # 0=webcam, URL, or video path
  width: 1280
  height: 720
  fps_target: 30
  resize_factor: 1.0             # 0.5 = 50% resolution (faster)
  frame_skip: 0                  # Skip frames for speed (0=process all)

model:
  detector: "models/yolov8n.pt"
  device: "cuda"                 # cuda or cpu
  confidence_threshold: 0.5
  nms_iou_threshold: 0.45
  max_detections: 100

ppe:
  enabled: true
  person_class_id: 0
  ppe_model_path: "models/ppe_classifier.pt"
  required_ppe: ["helmet", "vest"]
  confidence_threshold: 0.7

fall_detection:
  enabled: true
  aspect_ratio_threshold: 0.5
  velocity_frames: 5
  velocity_threshold: 50
  min_area_fraction: 0.01

zones:
  enabled: true
  zones:
    - name: "Restricted Area"
      type: "polygon"
      points: [[100, 100], [500, 100], [500, 400], [100, 400]]
      color: [0, 0, 255]
      alert_on_intrusion: true

alerts:
  enabled: true
  cooldown_seconds: 5
  log_path: "logs/alerts.log"

recording:
  enabled: true
  output_path: "logs/videos/violations.mp4"
  codec: "mp4v"

display:
  show_fps: true
```

## Project Structure

```
safety_monitor/
├── config/                 # Configuration management
├── models/                 # Detection & classification models
│   ├── detector.py            # YOLOv8 object detector
│   ├── ppe_classifier.py      # PPE detection
│   ├── fall_detector.py       # Fall detection engine
│   └── safety_pipeline.py     # Main processing pipeline
├── utils/                  # Utility modules
│   ├── video_capture.py       # Video input handling
│   ├── zone_manager.py        # Zone management & intrusion
│   ├── alert_manager.py       # Alert system
│   ├── drawing.py             # Frame annotation utilities
│   └── logger.py              # Logging setup
├── dashboard/              # Streamlit dashboard (optional)
├── logs/                   # Output: alerts, screenshots, videos
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── yolov8n.pt             # Pre-trained YOLOv8 model
```

## Core Components

### ObjectDetector
Detects persons in frames using YOLOv8.
```python
detector = ObjectDetector(
    model_path="models/yolov8n.pt",
    device="cuda",
    conf_threshold=0.5
)
detections = detector.detect(frame)
```

### PPEClassifier
Classifies PPE items for each detected person.
```python
ppe = PPEClassifier(
    model_path="models/ppe_classifier.pt",
    conf_threshold=0.7
)
ppe_status = ppe.classify(frame, detections)
```

### FallDetector
Detects falls using motion analysis.
```python
fall_detector = FallDetector(
    aspect_ratio_threshold=0.5,
    velocity_frames=5
)
is_falling = fall_detector.detect(detection, prev_detection)
```

### SafetyPipeline
Orchestrates all detectors and returns comprehensive results.
```python
pipeline = SafetyPipeline(
    detector=detector,
    ppe=ppe,
    fall=fall_detector,
    zone_mgr=zone_mgr,
    alert_mgr=alert_mgr
)
result = pipeline.process(frame)
```

## Performance Tips

### Optimize for Speed
```yaml
video:
  resize_factor: 0.5      # Process at 50% resolution
  frame_skip: 2           # Process every 3rd frame
model:
  confidence_threshold: 0.4  # Lower = more detections (slower)
```

### Optimize for Accuracy
```yaml
video:
  resize_factor: 1.0      # Full resolution
  frame_skip: 0           # Process all frames
model:
  confidence_threshold: 0.7  # Higher = fewer false positives
```

## Troubleshooting

**CUDA Out of Memory**
```bash
python main.py --resize-factor 0.5
# Or switch to CPU
export TORCH_DEVICE=cpu
python main.py
```

**No Detections**
- Lower confidence threshold in `config.yaml`
- Ensure camera/video source is working
- Check lighting conditions

**Low FPS**
- Reduce resolution: `resize_factor: 0.5`
- Skip frames: `frame_skip: 2`
- Use CPU-only if GPU memory is limited

**Camera Not Found**
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

## Output

### Console
```
=== Safety Monitor v1.3.0 starting ===
System ready. Press Q to quit, S to screenshot.
[FRAME 0001] Persons: 3 | Violations: 1 | Zone Intrusions: 0 | FPS: 28.5
[ALERT] PPE Violation: Person #2 missing helmet
[ALERT] Zone intrusion in Restricted Area
Screenshot saved: logs/screenshot_1686754323.jpg
```

### Logs
- `logs/safety_monitor.log` - Main system log
- `logs/alerts.log` - Alert events
- `logs/screenshots/` - Captured frames
- `logs/videos/` - Recorded violations

## Deployment

### Docker
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

```bash
docker build -t safety-monitor .
docker run --gpus all -it safety-monitor
```

### Linux Service
```bash
# Copy service file
sudo cp safety_monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable safety_monitor
sudo systemctl start safety_monitor

# Check status
sudo systemctl status safety_monitor
journalctl -u safety_monitor -f
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Open Pull Request

## Roadmap

- [ ] Web-based dashboard
- [ ] REST API
- [ ] Multi-camera support
- [ ] Custom model training
- [ ] Mobile app

## License

MIT License - See LICENSE file for details

## Support

- 📋 **Issues**: [GitHub Issues](https://github.com/Ayesha037/safety_monitor/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Ayesha037/safety_monitor/discussions)

## Acknowledgements

- **OpenCV** - Computer vision library
- **Ultralytics** - YOLOv8 framework
- **PyTorch** - Deep learning framework

---

**⭐ If this project helped you, please star the repository!**

Made with ❤️ for workplace safety
