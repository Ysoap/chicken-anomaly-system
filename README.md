# ?? Chicken Farm Anomaly Detection System

[![GitHub stars](https://img.shields.io/github/stars/Ysoap/chicken-anomaly-system?style=social)](https://github.com/Ysoap/chicken-anomaly-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Real-time chicken farm anomaly detection using 3-stage computer vision pipeline: RT-DETRv2-L + VideoMAE + LSTM Autoencoder

## ?? Overview

This project implements an end-to-end anomaly detection system for chicken farm monitoring. The system processes RTSP video streams through a 3-stage deep learning pipeline and sends real-time alerts to a mobile app via Firebase.

**Architecture:**
```
RTSP Stream ? [Stage 1: Detection] ? [Stage 2: Behavior] ? [Stage 3: Anomaly] ? Firebase ? Mobile App
```

**Timeline:** 4-week sprint (Nov 4 - Nov 29, 2026) | Presentation: Week 5 (Dec 2, 2026)

---

## ??? Architecture

```
+-------------------------------------------------------------+
¦                      RTSP Video Stream                      ¦
¦                   (MediaMTX + OBS Studio)                   ¦
+-------------------------------------------------------------+
                                  ¦
                    +-------------?------------+
                    ¦  Stage 1: RT-DETR +      ¦
                    ¦  ByteTrack Detection     ¦ 30 FPS
                    ¦  (Track chickens)        ¦
                    +--------------------------+
                                  ¦ bounding boxes + track IDs
                    +-------------?------------+
                    ¦  Stage 2: VideoMAE       ¦
                    ¦  Behavior Classification ¦ 10 clips/s
                    ¦  (eating/drinking/etc)   ¦
                    +--------------------------+
                                  ¦ behavior labels
                    +-------------?------------+
                    ¦  Stage 3: LSTM-AE        ¦
                    ¦  Anomaly Detection       ¦ 1/sec
                    ¦  (unsupervised)          ¦
                    +--------------------------+
                                  ¦ anomaly score
                    +-------------?------------+
                    ¦  FastAPI REST API        ¦
                    ¦  + Firebase Integration  ¦
                    +--------------------------+
                                  ¦
                    +-------------?------------+
                    ¦  Mobile App (Kotlin)     ¦
                    ¦  Real-time Dashboard +   ¦
                    ¦  Push Notifications ??   ¦
                    +--------------------------+
```

---

## ?? Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Ysoap/chicken-anomaly-system.git
cd chicken-anomaly-system
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Setup Firebase
1. Create Firebase project at https://console.firebase.google.com
2. Download google-services.json and place in mobile-app/app/
3. Download Firebase Admin SDK JSON and place in ackend/api/

### 4. Run Backend Pipeline
```bash
python backend/integration/pipeline.py
```

### 5. Run Mobile App
1. Open mobile-app/ in Android Studio
2. Sync Gradle
3. Run on device/emulator

---

## ??? Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Backend** | Python, PyTorch, Transformers (Hugging Face), FastAPI, OpenCV |
| **Stage 1** | RT-DETRv2-L (NMS-free detection), ByteTrack (multi-object tracking) |
| **Stage 2** | VideoMAE-Small (video transformer with 90% masking) |
| **Stage 3** | LSTM Autoencoder (unsupervised anomaly detection) |
| **Mobile** | Kotlin, Jetpack Compose, Firebase (Realtime DB, Cloud Messaging, Storage) |
| **Streaming** | MediaMTX (RTSP server), OBS Studio |
| **Dataset** | ChickenVerse (ChickenDet for detection, ChickenAct for behavior) |
| **Hardware** | NVIDIA RTX 3060/4060 (12-16GB VRAM) |

---

## ?? Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Ysoap** | PM / Tech Lead | Stage 3 (LSTM-AE), FastAPI, Firebase backend, Integration |
| **Pandu** | Mobile Developer | Kotlin Android app, Firebase SDK, UI/UX |
| **Azis** | CV Engineer | Stage 1 (RT-DETR + ByteTrack), Dataset preparation |
| **Malik** | CV Engineer | Stage 2 (VideoMAE), Behavior classification |

---

## ?? Timeline

| Week | Milestone |
|------|-----------|
| **Week 1** (Nov 4-8) | Dataset preparation, individual stage training, infrastructure setup |
| **Week 2** (Nov 11-15) | Feature engineering, LSTM-AE training, mobile app scaffold |
| **Week 3** (Nov 18-22) | Full integration, FastAPI deployment, mobile Firebase integration |
| **Week 4** (Nov 25-29) | Testing, documentation, demo video, presentation prep |
| **Week 5** (Dec 2) | ?? Final Presentation |

---

## ?? Documentation

- **[?? Whitepaper](docs/buku-putih-spesifikasi-sistem.md)** - Complete system specification
- **[?? Pipeline Analogies](docs/ANALOGI_PIPELINE.md)** ? **MUST READ** - Understand 3 stages with clear analogies
- **[?? Architecture](docs/ARSITEKTUR_LENGKAP.md)** - Technical deep dive
- **[?? Git Workflow](docs/GIT_WORKFLOW.md)** - Branch strategy and PR guidelines
- **[?? Weekly Progress](docs/WEEKLY_PROGRESS.md)** - Sprint tracking checklist
- **[?? API Docs](docs/API_DOCUMENTATION.md)** - FastAPI endpoints specification
- **[?? Mobile Guide](docs/MOBILE_FIREBASE_GUIDE.md)** - Android + Firebase integration

---

## ?? Performance Targets

| Metric | Target |
|--------|--------|
| Stage 1 FPS | 30 FPS |
| Stage 2 Throughput | 10 clips/sec |
| End-to-end Latency | <2 seconds |
| False Positive Rate | <5% |
| Anomaly Detection Recall | >90% |

---

## ?? Dataset

**ChickenVerse Dataset:**
- **ChickenDet** (~5GB): Object detection dataset with chicken bounding boxes
- **ChickenAct** (~10GB): Behavior classification dataset with 15 action classes

Download instructions: data/download_dataset.sh

---

## ?? Testing

```bash
pytest tests/
```

---

## ?? License

MIT License - see [LICENSE](LICENSE) file for details.

---

## ?? Acknowledgments

- ChickenVerse dataset authors
- Hugging Face Transformers team
- ByteTrack and RT-DETR authors

---

**For questions or issues, open a GitHub Issue or contact the team lead.**