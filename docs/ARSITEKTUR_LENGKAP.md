# ??? Arsitektur Teknis Lengkap

> Dokumen ini adalah spesifikasi teknis mendalam untuk semua komponen sistem.

---

## ?? System Components Overview

```
+-------------------------------------------------------------+
¦                    Input Layer (RTSP)                       ¦
¦  MediaMTX Server (Go) + OBS Studio                          ¦
¦  Video: H.264, 1280x720, 10 FPS                             ¦
+-------------------------------------------------------------+
                         ¦ RTSP Stream (rtsp://localhost:8554)
           +-------------?-------------+
           ¦  OpenCV RTSP Capture      ¦
           ¦  (Python, cv2.VideoCapture)¦
           +---------------------------+
                         ¦
    +--------------------?--------------------+
    ¦        Stage 1: Detection Thread        ¦
    ¦  RT-DETRv2-L + ByteTrack (GPU)          ¦
    ¦  Input: RGB 640x640                     ¦
    ¦  Output: Boxes + Track IDs (30 FPS)     ¦
    +-----------------------------------------+
                         ¦ Queue 1 (thread-safe)
    +--------------------?--------------------+
    ¦     Stage 2: Behavior Thread            ¦
    ¦  VideoMAE-Small (GPU)                   ¦
    ¦  Input: 16 frames per track             ¦
    ¦  Output: Behavior labels (10 clips/s)   ¦
    +-----------------------------------------+
                         ¦ Queue 2 (thread-safe)
    +--------------------?--------------------+
    ¦      Stage 3: Anomaly Thread            ¦
    ¦  LSTM Autoencoder (CPU)                 ¦
    ¦  Input: 12 features, 30-60s window      ¦
    ¦  Output: Anomaly score (1/sec)          ¦
    +-----------------------------------------+
                         ¦
    +--------------------?--------------------+
    ¦         FastAPI REST API                ¦
    ¦  + Firebase Admin SDK                   ¦
    +-----------------------------------------+
                         ¦ HTTPS
    +--------------------?--------------------+
    ¦      Firebase Realtime Database         ¦
    ¦      + Cloud Messaging (FCM)            ¦
    +-----------------------------------------+
                         ¦ WebSocket / FCM
    +--------------------?--------------------+
    ¦     Mobile App (Kotlin Android)         ¦
    ¦  Jetpack Compose + MVVM                 ¦
    +-----------------------------------------+
```

---

## ?? Input Layer: RTSP Streaming

### MediaMTX Configuration

**File:** streaming-setup/mediamtx/mediamtx.yml

```yaml
rtspAddress: :8554
protocols: [tcp]
runOnInit: ffmpeg -re -i /path/to/video.mp4 -c copy -f rtsp rtsp://localhost:8554/stream
```

**Start MediaMTX:**
```bash
./mediamtx mediamtx.yml
```

### OBS Studio Setup

1. Add **Media Source**: Select chicken farm video file
2. **Settings** ? **Stream**:
   - Service: Custom
   - Server: tsp://localhost:8554/stream
   - Stream Key: (leave empty)
3. Click **Start Streaming**

### Python RTSP Capture

```python
import cv2

cap = cv2.VideoCapture("rtsp://localhost:8554/stream")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
```

---

## ?? Stage 1: RT-DETRv2-L + ByteTrack

### Model Architecture

**RT-DETRv2-L (Real-Time Detection Transformer v2 Large):**
- Backbone: ResNet-50 + FPN
- Neck: Hybrid Encoder (CNN + Transformer)
- Head: DETR-style decoder (6 layers)
- **Key Feature:** NMS-free (no post-processing needed)
- Parameters: ~110M
- Pretrained: COCO (80 classes)

**Fine-tuning on ChickenDet:**
```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

model = RTDetrForObjectDetection.from_pretrained("facebook/rt-detr-v2-l")
model.config.num_labels = 1  # Only "chicken" class

# Fine-tune on ChickenDet
# ... training loop ...
```

---

### ByteTrack Algorithm

**Purpose:** Associate detections across frames (multi-object tracking)

**Algorithm:**
1. For each frame, RT-DETR outputs N detections with confidence scores
2. Separate into high-confidence (>0.5) and low-confidence (0.1-0.5)
3. **First association:** Match high-confidence detections to existing tracks (IoU matching)
4. **Second association:** Match low-confidence detections to unmatched tracks (handles occlusion)
5. **Kalman filter:** Predict track position for next frame

**Why ByteTrack?**
- Handles occlusion better than traditional trackers (e.g., DeepSORT)
- Tracks low-confidence detections temporarily (chickens behind feeders)

**Output:**
- Bounding boxes: [(x1, y1, x2, y2), ...]
- Track IDs: [0, 1, 2, ..., N] (persistent across frames)
- Confidence scores: [0.87, 0.92, ...]

---

### Performance

**Hardware:** NVIDIA RTX 3060 (12GB VRAM)

| Metric | Value |
|--------|-------|
| Inference FPS | 30 |
| Latency per frame | ~33ms |
| GPU memory usage | ~4GB |
| mAP (ChickenDet test set) | >0.85 |

---

## ?? Stage 2: VideoMAE-Small

### Model Architecture

**VideoMAE (Video Masked Autoencoder):**
- Based on Vision Transformer (ViT)
- Encoder: 12 layers, 384 hidden dim
- **Key Feature:** 90% tube masking during pretraining
- Parameters: ~22M
- Pretrained: Kinetics-400 (400 action classes)

**Why 90% Masking Matters:**
- Forces model to learn temporal dependencies (motion flow)
- Ignores static appearance (white feathers, cream bedding)
- Critical for low-contrast scenarios (chicken farms)

---

### Fine-tuning on ChickenAct

**Label Mapping (15 ? 4 classes):**

| ChickenAct Classes | Our Label |
|--------------------|-----------|
| pecking, foraging, scratching, feeding | eating |
| standing, sitting, sleeping, resting | resting |
| walking, running, moving | walking |
| drinking | drinking |

**Training Configuration:**
```python
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-small")
model.config.num_labels = 4

# Training hyperparameters
batch_size = 8
learning_rate = 1e-4
num_epochs = 20
clip_length = 16  # frames
frame_rate = 10   # FPS
```

---

### Clip Extraction

**Per Track ID:**
1. Stage 1 outputs bounding box + track ID
2. Crop frame using bounding box ? 224x224
3. Buffer 16 consecutive frames for same track ID
4. When buffer full ? run VideoMAE inference
5. Clear buffer, repeat

**Buffer Management:**
- FIFO queue per track ID
- If track disappears (occlusion), keep buffer for 2 seconds
- Discard buffer if track does not reappear

**Output:**
- Behavior label: {eating: 0.7, drinking: 0.1, walking: 0.1, resting: 0.1}
- Assigned to track ID

---

### Performance

| Metric | Value |
|--------|-------|
| Inference throughput | 10 clips/sec |
| Latency per clip | ~100ms |
| GPU memory usage | ~3GB |
| Accuracy (4-class) | >85% |

---

## ?? Stage 3: LSTM Autoencoder

### Architecture

**Encoder:**
```
Input (12 features) ? LSTM(64) ? LSTM(32) ? Latent
```

**Decoder:**
```
Latent ? LSTM(32) ? LSTM(64) ? Output (12 features)
```

**Training:**
- Loss: MSE (Mean Squared Error) between input and reconstruction
- Optimizer: Adam, lr=1e-3
- Epochs: 100
- Early stopping: patience=10

**Training Data:**
- 10 hours normal behavior video
- Extract 12 features every 1 second ? 36,000 samples
- Window size: 60 samples (60 seconds)
- Sliding window: stride=1

---

### 12 Custom Features

| # | Feature | Extraction Method |
|---|---------|-------------------|
| 1 | Mean Velocity | 
p.mean([track.velocity for track in tracks]) |
| 2 | Velocity Std | 
p.std([track.velocity for track in tracks]) |
| 3 | Mean NND | 
p.mean([nearest_neighbor_distance(track) for track in tracks]) |
| 4 | NND Std | 
p.std([...]) |
| 5 | Overlap Ratio | count(iou(box1, box2) > 0.3) / total_pairs |
| 6 | Entropy | scipy.stats.entropy(behavior_distribution) |
| 7 | Directional Coherence | 
p.mean([cosine_similarity(v1, v2) for v1, v2 in velocity_pairs]) |
| 8 | Eating % | count(behavior == 'eating') / total |
| 9 | Resting % | count(behavior == 'resting') / total |
| 10 | Active Track Count | len(tracks) |
| 11 | Track Turnover Rate | (new_tracks + lost_tracks) / 2 |
| 12 | Motion Intensity | 
p.sum(optical_flow_magnitudes) |

---

### Anomaly Detection Logic

```python
# Training phase
errors = []
for window in training_data:
    reconstruction = model(window)
    error = mse(window, reconstruction)
    errors.append(error)

threshold = np.mean(errors) + 3 * np.std(errors)  # 3-sigma

# Inference phase
def detect_anomaly(window):
    reconstruction = model(window)
    error = mse(window, reconstruction)
    
    if error > threshold:
        return {
            "is_anomaly": True,
            "anomaly_score": error / threshold,
            "diagnosis": explain_anomaly(window, reconstruction)
        }
    return {"is_anomaly": False, "anomaly_score": error / threshold}
```

---

### Feature Importance (XAI)

To explain which features contributed most to anomaly:

```python
def explain_anomaly(input_window, reconstruction):
    feature_errors = (input_window - reconstruction) ** 2  # Per-feature MSE
    contributions = feature_errors / feature_errors.sum()  # Normalize
    
    top_features = sorted(enumerate(contributions[-1]), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "top_features": [
            {"name": FEATURE_NAMES[idx], "contribution": contrib}
            for idx, contrib in top_features
        ]
    }
```

---

## ?? FastAPI Backend

### Endpoints Implementation

**File:** ackend/api/main.py

```python
from fastapi import FastAPI
from firebase_admin import credentials, initialize_app, db, messaging

# Initialize Firebase
cred = credentials.Certificate("firebase-adminsdk.json")
initialize_app(cred, {"databaseURL": "https://chicken-anomaly-detection.firebaseio.com"})

app = FastAPI()

@app.get("/api/v1/status")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {"stage1": True, "stage2": True, "stage3": True}
    }

@app.get("/api/v1/anomaly/current")
async def get_current_anomaly():
    ref = db.reference("anomalies/latest")
    data = ref.get()
    return data

# Push to Firebase
def push_anomaly(anomaly_data):
    ref = db.reference(f"anomalies/{anomaly_data['date']}/{anomaly_data['time']}")
    ref.set(anomaly_data)
    
    # Send FCM notification if anomaly
    if anomaly_data["is_anomaly"]:
        message = messaging.Message(
            notification=messaging.Notification(
                title="?? Anomaly Detected",
                body=anomaly_data["diagnosis"]
            ),
            topic="anomaly_alerts"
        )
        messaging.send(message)
```

---

## ?? Mobile App Architecture

### MVVM Pattern

```
View (Composables) ? ViewModel ? Repository ? Data Source (Firebase)
```

**ViewModel Example:**
```kotlin
class DashboardViewModel(private val repository: AnomalyRepository) : ViewModel() {
    private val _currentAnomaly = MutableLiveData<AnomalyData>()
    val currentAnomaly: LiveData<AnomalyData> = _currentAnomaly

    init {
        repository.listenToLiveData { data ->
            _currentAnomaly.postValue(data)
        }
    }
}
```

---

## ?? Configuration Files

### Pipeline Configuration

**File:** ackend/integration/config.yaml

```yaml
rtsp:
  url: rtsp://localhost:8554/stream
  reconnect_attempts: 5

stage1:
  model_path: backend/stage1_detection/models/rt_detr_checkpoint.pth
  confidence_threshold: 0.5
  target_fps: 30

stage2:
  model_path: backend/stage2_behavior/models/videomae_checkpoint.pth
  clip_length: 16
  buffer_timeout_sec: 2

stage3:
  model_path: backend/stage3_anomaly/models/lstm_ae_checkpoint.pth
  window_size_sec: 60
  threshold_multiplier: 3  # 3-sigma

firebase:
  credentials_path: backend/api/firebase-adminsdk.json
  database_url: https://chicken-anomaly-detection.firebaseio.com

logging:
  level: INFO
  file: logs/pipeline.log
```

---

## ?? Testing Strategy

### Unit Tests

```python
# tests/test_stage1.py
def test_rt_detr_inference():
    model = load_model("stage1")
    image = load_sample_image()
    boxes, scores, labels = model(image)
    assert len(boxes) > 0
    assert all(score > 0.5 for score in scores)

# tests/test_stage3.py
def test_feature_extraction():
    tracks = load_sample_tracks()
    features = extract_features(tracks)
    assert len(features) == 12
    assert 0 <= features[7] <= 1  # Eating % between 0-1
```

---

## ?? Deployment

### Backend Deployment (Docker)

```dockerfile
FROM python:3.10

WORKDIR /app
COPY backend/ /app/
RUN pip install -r requirements.txt

CMD ["python", "integration/pipeline.py"]
```

### Mobile Deployment

1. Generate signed APK:
   - Build ? Generate Signed Bundle / APK
   - Create keystore (save securely!)
2. Upload to Google Play Console (internal testing track)

---

## ?? Performance Benchmarks

| Component | Latency | Throughput | GPU Memory |
|-----------|---------|------------|------------|
| Stage 1 | 33ms | 30 FPS | 4GB |
| Stage 2 | 100ms | 10 clips/s | 3GB |
| Stage 3 | 50ms | 1/sec | CPU only |
| **End-to-end** | **<2s** | - | **7GB** |

---

**For technical questions, open GitHub Issue with label rchitecture.**