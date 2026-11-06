# ?? FastAPI REST API Documentation

**Base URL:** http://localhost:8000  
**Version:** 1.0.0

---

## Authentication

Currently no authentication (internal network use). For production deployment, add API key authentication.

---

## Endpoints

### 1. Health Check

**Endpoint:** GET /api/v1/status

**Description:** Check if API server and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-11-06T10:00:00Z",
  "models_loaded": {
    "stage1": true,
    "stage2": true,
    "stage3": true
  },
  "rtsp_connected": true,
  "firebase_connected": true
}
```

**Status Codes:**
- 200 OK - Service is healthy
- 503 Service Unavailable - One or more models failed to load

---

### 2. Current Anomaly Status

**Endpoint:** GET /api/v1/anomaly/current

**Description:** Get the most recent anomaly detection result.

**Response:**
```json
{
  "anomaly_score": 0.87,
  "threshold": 0.75,
  "is_anomaly": true,
  "confidence": 0.92,
  "diagnosis": {
    "top_features": [
      {
        "name": "mean_velocity",
        "value": 4.2,
        "zscore": 3.2,
        "contribution": 0.35,
        "normal_range": [0.5, 2.0]
      },
      {
        "name": "mean_nnd",
        "value": 0.15,
        "zscore": 2.8,
        "contribution": 0.28,
        "normal_range": [0.3, 0.8]
      },
      {
        "name": "overlap_ratio",
        "value": 0.42,
        "zscore": 2.1,
        "contribution": 0.18,
        "normal_range": [0.05, 0.15]
      }
    ],
    "summary": "Abnormally high movement speed + crowding detected. Possible stampede behavior.",
    "severity": "high"
  },
  "timestamp": "2026-11-06T10:05:32Z",
  "active_tracks": 47,
  "behavior_distribution": {
    "eating": 0.15,
    "drinking": 0.05,
    "walking": 0.65,
    "resting": 0.15
  }
}
```

**Status Codes:**
- 200 OK - Success
- 404 Not Found - No recent data (pipeline not running)

---

### 3. Historical Anomaly Data

**Endpoint:** GET /api/v1/anomaly/history

**Query Parameters:**
- hours (int, default: 1) - Time window in hours (max: 24)
- interval (string, default: "minute") - Aggregation interval: minute, 5min, 15min, hour
- ormat (string, default: "json") - Response format: json, csv

**Example Request:**
```
GET /api/v1/anomaly/history?hours=2&interval=5min
```

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2026-11-06T08:00:00Z",
      "anomaly_score": 0.23,
      "is_anomaly": false
    },
    {
      "timestamp": "2026-11-06T08:05:00Z",
      "anomaly_score": 0.31,
      "is_anomaly": false
    },
    {
      "timestamp": "2026-11-06T08:10:00Z",
      "anomaly_score": 0.87,
      "is_anomaly": true
    }
  ],
  "stats": {
    "mean": 0.42,
    "std": 0.15,
    "max": 0.87,
    "min": 0.12,
    "anomaly_count": 3,
    "total_samples": 24
  },
  "query": {
    "hours": 2,
    "interval": "5min",
    "start_time": "2026-11-06T08:00:00Z",
    "end_time": "2026-11-06T10:00:00Z"
  }
}
```

**CSV Format Example:**
```csv
timestamp,anomaly_score,is_anomaly
2026-11-06T08:00:00Z,0.23,false
2026-11-06T08:05:00Z,0.31,false
2026-11-06T08:10:00Z,0.87,true
```

**Status Codes:**
- 200 OK - Success
- 400 Bad Request - Invalid query parameters

---

### 4. Manual Inference Trigger

**Endpoint:** POST /api/v1/inference

**Description:** Trigger inference on a pre-recorded video file (for testing).

**Request Body:**
```json
{
  "video_path": "/path/to/test_video.mp4",
  "duration_sec": 60,
  "output_format": "json"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "estimated_time_sec": 30,
  "progress_url": "/api/v1/inference/550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- 202 Accepted - Job created
- 400 Bad Request - Invalid video path

---

### 5. Check Inference Job Status

**Endpoint:** GET /api/v1/inference/{job_id}

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result": {
    "anomaly_detections": 5,
    "max_anomaly_score": 0.92,
    "duration_processed_sec": 60,
    "output_file": "/results/550e8400.json"
  }
}
```

**Status Values:**
- queued - Job is waiting
- processing - Job is running
- completed - Job finished successfully
- ailed - Job failed (see error field)

---

### 6. WebSocket Live Stream (Optional)

**Endpoint:** WS /ws/live

**Description:** Real-time anomaly score streaming via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/live");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

**Message Format:**
```json
{
  "type": "anomaly_update",
  "anomaly_score": 0.42,
  "is_anomaly": false,
  "timestamp": "2026-11-06T10:05:32Z",
  "active_tracks": 35
}
```

---

## Firebase Integration

### Realtime Database Schema

**Path:** /anomalies/{date}/{time}

**Example:**
```
/anomalies
  /2026-11-06
    /10:00:00
      anomaly_score: 0.87
      is_anomaly: true
      diagnosis: "High velocity + crowding"
      timestamp: 1730887500000
      top_features: [...]
    /10:00:01
      anomaly_score: 0.45
      is_anomaly: false
      ...
```

### Cloud Messaging Payload

**Notification Format:**
```json
{
  "notification": {
    "title": "?? Anomaly Detected",
    "body": "Abnormal movement detected in Farm A - Severity: High"
  },
  "data": {
    "anomaly_score": "0.87",
    "diagnosis": "High velocity + crowding",
    "timestamp": "2026-11-06T10:05:00Z",
    "severity": "high"
  },
  "priority": "high",
  "android": {
    "notification": {
      "channel_id": "anomaly_alerts",
      "sound": "default",
      "color": "#FF0000"
    }
  }
}
```

---

## Error Responses

**General Error Format:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Video path does not exist",
    "details": "/path/to/test_video.mp4 not found"
  },
  "timestamp": "2026-11-06T10:05:32Z"
}
```

**Error Codes:**
- INVALID_REQUEST - Bad request parameters
- MODEL_NOT_LOADED - One or more models failed to load
- RTSP_CONNECTION_FAILED - Cannot connect to RTSP stream
- FIREBASE_ERROR - Firebase operation failed
- INTERNAL_ERROR - Unexpected server error

---

## Rate Limiting

**Current:** No rate limiting (internal use)  
**Production:** Implement rate limiting (100 requests/minute per IP)

---

## Testing with cURL

### Health Check
```bash
curl -X GET http://localhost:8000/api/v1/status
```

### Current Anomaly
```bash
curl -X GET http://localhost:8000/api/v1/anomaly/current
```

### Historical Data
```bash
curl -X GET "http://localhost:8000/api/v1/anomaly/history?hours=1&interval=minute"
```

### Manual Inference
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/test_video.mp4",
    "duration_sec": 60
  }'
```

---

## Performance

**Expected Latency:**
- Health check: <10ms
- Current anomaly: <50ms
- Historical data (1 hour): <200ms
- Manual inference: 30-60s (depends on video length)

---

**For questions, open a GitHub Issue with label pi.**