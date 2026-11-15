# Backend API

FastAPI service that exposes the chicken anomaly detection pipeline to
the mobile app and to anything else that needs the current anomaly
status over HTTP. The integration pipeline writes scored windows into
this service's state; the mobile app reads them back, either over HTTP
or through Firebase (see the integration notes below).

## Run it

From the repo root:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs are then at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness probe. Returns `{"status": "ok", "time": ...}`. Works even without PyTorch or a checkpoint. |
| GET | `/api/v1/anomaly/current` | Latest scored window: `score`, `threshold`, `is_anomaly`, `track_id`, `start_frame`, `end_frame`, `timestamp`. Returns 404 with `current: null` before any window is scored. |
| GET | `/api/v1/anomaly/history` | Recent scored windows, newest last. Optional `?limit=N`. Stored in a bounded in-memory ring buffer (200 entries by default), so history resets when the API restarts. |
| POST | `/api/v1/pipeline/score` | Score raw tracking records with Stage 3. Body: a JSON list of records, or `{"records": [...]}`. Returns the scored windows. Returns 503 with an explanation when no Stage 3 checkpoint is configured, 400 for a malformed body, 422 when the records cannot be scored. |
| GET | `/api/v1/status` | Pipeline status: `stage3_checkpoint_loaded`, `stage3_error`, `firebase` (enabled or not, plus any init error), `windows_scored`, `anomalies_detected`, `history_size`. |

### Tracking record format

The POST body records use the same shape the Stage 3 feature extractor
accepts:

```json
{
  "frame": 128,
  "track_id": "track-7",
  "x": 210.0,
  "y": 145.0,
  "w": 46.0,
  "h": 40.0,
  "behavior": "eating"
}
```

`behavior` is optional and comes from Stage 2. A track needs at least
one full window of records (32 frames by default) before it produces a
score.

Example:

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/score \
  -H "Content-Type: application/json" \
  -d '{"records": [{"frame": 0, "track_id": "a", "x": 10, "y": 10, "w": 40, "h": 40}]}'
```

## Environment variables

| Variable | Used by | Meaning |
| --- | --- | --- |
| `STAGE3_CHECKPOINT` | `main.py` | Path to a trained Stage 3 checkpoint (`.pt` written by `backend/stage3_anomaly/train.py`). Without it the API still starts and serves `/health` and `/api/v1/status`, but scoring endpoints report that the checkpoint is not configured. |
| `FIREBASE_CREDENTIALS` | `firebase_client.py` | Path to a Firebase service account JSON. Without it the Firebase client runs in a no-op mode that logs instead of sending. Never commit this file. |
| `FIREBASE_DATABASE_URL` | `firebase_client.py` | Realtime Database URL, for example `https://your-project-default-rtdb.firebaseio.com`. Needed for the database writes when the service account does not carry it. |

## Firebase behavior

`firebase_client.py` wraps `firebase-admin` defensively:

- It initialises only when a credentials path is provided. Missing
  credentials or a missing `firebase-admin` package put it in a
  disabled no-op mode; every method logs and returns `False` instead
  of raising.
- `PipelineState.record_result` calls it after every scored window:
  the current status is written to `/status/current`, and anomalous
  windows are also appended to `/anomalies/events` and pushed as an
  FCM alert to the `anomalies` topic.
- No credentials or secrets live in this codebase. The service account
  JSON stays on the server and is referenced by path only.

## Mobile app integration notes

The Kotlin app has two ways to follow the pipeline:

1. **Firebase Realtime Database (recommended for the dashboard).**
   - `/status/current`: the latest status object. Fields: `current`
     (the newest scored window with `score`, `threshold`, `is_anomaly`,
     `track_id`, `start_frame`, `end_frame`, `timestamp`),
     `windows_scored`, `anomalies_detected`, `updated_at`.
   - `/anomalies/events`: an append-only list of anomalous windows,
     pushed with generated keys, newest at the end. Same window fields
     as above.
2. **HTTP.** Poll `GET /api/v1/anomaly/current` for the dashboard and
   `GET /api/v1/anomaly/history` for a recent activity list.

For push notifications, the app should subscribe to the FCM topic
`anomalies`. Alerts are sent only for windows whose score is above the
Stage 3 threshold, with the score, track id and frame range in the
message body.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI app factory (`create_app`), the endpoints, and `PipelineState`, the shared thread-safe state object the pipeline writes into. Importable and testable without starting a server. |
| `firebase_client.py` | `FirebaseClient`, the defensive firebase-admin wrapper described above. |

## Tests

From the repo root:

```bash
pytest tests/test_api_pipeline.py
```

The API tests use FastAPI's TestClient and need neither PyTorch nor
Firebase credentials.
