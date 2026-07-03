# ⚽ Football AI Analyzer

A full-stack real-time football tactical analysis system using computer vision. Upload a match video and get live player tracking, team classification, speed metrics, and tactical Voronoi analysis — all streamed to a React dashboard via WebSocket.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![React](https://img.shields.io/badge/React-18.2-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Custom%20Trained-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Demo

> Upload a match video → real-time detection, tracking, speed overlay, and tactical radar

<video src="https://github.com/LazyTacoz/football-ai-analyzer/assets/demo.mp4" controls width="100%"></video>

---

## 🧠 Technical Pipeline

```
Video Input
    │
    ▼
YOLOv8m (custom fine-tuned)          ← ball, player, goalkeeper, referee
    │
    ▼
ByteTrack Multi-Object Tracker       ← persistent player IDs across frames
    │
    ▼
Pitch Boundary Detector              ← HSV grass segmentation
    │
    ▼
Automatic Homography Estimation      ← Hough line detection + RANSAC
    │
    ▼
Player Kinematics                    ← real-world speed (km/h) + distance (m)
    │
    ▼
Voronoi Spatial Analysis             ← team possession + pitch control zones
    │
    ▼
WebSocket Stream → React Dashboard   ← live annotated frames + radar board
```

---

## 🔬 Model Training

Custom YOLOv8m fine-tuned on the [Powerfoot Dataset](https://universe.roboflow.com/esprit-po5qf/powerfoot) (9,831 images, ~29k augmented):

| Class | mAP50 |
|-------|-------|
| Player | 0.950 |
| Goalkeeper | 0.950 |
| Referee | 0.924 |
| Ball | 0.717 |
| **Overall** | **0.885** |

Training: 50 epochs, YOLOv8m backbone, 640px resolution, Tesla T4 GPU

---

## 🏗️ Architecture

```
football-ai-analyzer/
├── main.py                      # FastAPI server + WebSocket endpoints
├── processor.py                 # CV/ML pipeline orchestrator
├── pitch_keypoint_detector.py   # Auto homography via Hough transforms
├── player_kinematics.py         # Real-world speed + distance computation
├── src/
│   ├── App.jsx                  # React dashboard
│   └── index.css                # Tailwind styles
├── requirements.txt
└── package.json
```

---

## ⚙️ Features

- **Custom YOLOv8m** — fine-tuned on 29k football broadcast images (4 classes)
- **ByteTrack** — persistent player IDs with Kalman filter smoothing
- **Auto Pitch Homography** — Hough line detection + RANSAC for camera-to-pitch mapping
- **Player Kinematics** — real-world speed (km/h) and distance (m) from pixel displacement
- **Voronoi Spatial Control** — pitch territory analysis and possession calculation
- **Referee Filtering** — separate detection class for referees
- **Individual Player Heatmaps** — position frequency visualization per player
- **WebSocket Streaming** — live annotated frames streamed to React frontend
- **Sprint Detection** — alerts when players exceed 25 km/h

---

## 🚀 Setup

### Model Weights
Download `football.pt` from https://drive.google.com/file/d/1GqS2SUG3tHUQUl-TS4iNA-BW2Qmh0Jz6/view?usp=sharing
and place in the project root.

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA (recommended)

### Backend

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (check your CUDA version first)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Download model weights (place in project root)
# Get football.pt from releases or train your own (see Training section)

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
npm install
npm run dev
```

Open `http://localhost:5173`

---

## 🎓 Training Your Own Model

```bash
# Download dataset
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('esprit-po5qf').project('powerfoot')
project.version(2).download('yolov8', location='./powerfoot')
"

# Train
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data='powerfoot/data.yaml', epochs=50, imgsz=640, batch=16, device=0)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload video file |
| `POST` | `/process/{video_id}` | Start processing |
| `GET` | `/status/{video_id}` | Processing status |
| `GET` | `/output/{video_id}` | Download processed video |
| `WS` | `/ws/{video_id}` | Real-time frame stream |
| `GET` | `/heatmap/{video_id}/{player_id}` | Player heatmap |
| `GET` | `/kinematics/{video_id}` | Full kinematics report |

---

## 🔧 Tech Stack

**Backend:** Python, FastAPI, OpenCV, Ultralytics YOLOv8, Supervision, ByteTrack, Scipy, Scikit-learn

**Frontend:** React 18, Tailwind CSS, Vite, Lucide Icons

**ML:** YOLOv8m (custom fine-tuned), K-Means clustering, Kalman filtering, Voronoi diagrams, RANSAC homography

---

## 📄 License

MIT License — free to use in your own projects.

---

## 🙏 Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Supervision](https://github.com/roboflow/supervision)
- [Powerfoot Dataset](https://universe.roboflow.com/esprit-po5qf/powerfoot) by esprit (CC BY 4.0)
