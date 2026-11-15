# Integration Pipeline

The integration pipeline wires the three stages together and feeds the
results to the API and Firebase:

```
RTSP stream or video file (OpenCV)
  -> Stage 1  detector callable: frame -> tracking records
  -> Stage 2  behavior callable: records -> records with behavior labels
  -> BufferManager: records grouped by track -> fixed windows
  -> Stage 3  Stage3Scorer: windows -> anomaly scores (about 1 pass/sec)
  -> PipelineState (backend/api): current status + history ring buffer
  -> FirebaseClient: /status/current, /anomalies/events, FCM alerts
  -> Kotlin mobile app: realtime dashboard + push notifications
```

## How the stages connect

Stage 1 and Stage 2 are **pluggable callables**, because their models
live in `backend/stage1_detection` and `backend/stage2_behavior` and
are developed by different team members:

- `detector(frame, frame_index) -> list of records`: one frame in,
  tracking records out. Each record is a dict with `frame`, `track_id`,
  `x`, `y`, `w`, `h` (the Stage 3 extractor record shape).
- `behavior(frame, records) -> list of records`: the same records with
  a `behavior` label added where Stage 2 has one.

Pass them to `Pipeline(detector=..., behavior=...)`. Both default to
`None`, and the pipeline logs clearly what is missing:

- No detector: no frames are tracked. Records can still be injected
  with `Pipeline.feed_records(records)`, which is how tests and the
  demo drive the chain.
- No behavior callable: records carry no labels, and Stage 3 features
  10 and 11 (behavior change rate and stability) stay 0.0. Scoring
  still works.
- No Stage 3 checkpoint: windows are drained and counted but not
  scored, and the API reports the checkpoint as not configured.

## BufferManager

`buffer_manager.py` collects per-frame records, groups them by track,
and emits completed windows for Stage 3:

- Default window 32 frames, stride 16 (matching the Stage 3 defaults).
- A window is emitted when all of its frame slots are observed, or
  when later frames prove the missing slots are detection gaps and at
  least 80 percent of the window was observed. The Stage 3 extractor
  interpolates small gaps itself.
- Memory is bounded twice: records older than the next window start
  are dropped per track, and a global cap (20000 records by default)
  drops the oldest records first and counts them in `dropped_records`.
- Thread-safe: a capture thread can add records while the scoring
  loop pops windows.

API: `add_frame_records(records)`, `pop_ready_windows()`,
`pending_counts()`.

## Running it

From the repo root (this is the README quick start command):

```bash
python backend/integration/pipeline.py \
    --source rtsp://camera.local:8554/chicken \
    --stage3-checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
    --firebase-credentials path/to/service-account.json
```

| Argument | Meaning |
| --- | --- |
| `--source` | Video file path or RTSP URL, read with OpenCV. |
| `--stage3-checkpoint` | Trained Stage 3 checkpoint. Falls back to the `STAGE3_CHECKPOINT` environment variable. |
| `--firebase-credentials` | Firebase service account JSON path. Falls back to `FIREBASE_CREDENTIALS`. Omit it to run Firebase in no-op mode. |
| `--demo` | Run on synthetic data instead of a source (see below). |
| `--max-seconds` | Stop reading the source after this many seconds. |

Frames are consumed as fast as they arrive, but scoring runs on a
cadence (default one pass per second), matching the Stage 3 design
target. Each scored window is recorded into the API's `PipelineState`,
which writes the Firebase status, appends anomaly events, and sends
push alerts for windows above the threshold.

## Demo mode

No camera, no model weights, no Firebase needed:

```bash
python backend/integration/pipeline.py --demo
```

The demo generates synthetic tracking records (two normally moving
tracks plus one almost perfectly still track, the classic sick-bird
pattern), feeds them through the buffer, and prints the windows that
come out. Without a checkpoint the windows are counted but not scored;
add `--stage3-checkpoint` to see real scores and anomaly events:

```bash
python backend/integration/pipeline.py --demo \
    --stage3-checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt
```

Use the demo to smoke test the integration on any machine, including
CI, before touching the real camera setup.

## Running the API next to the pipeline

Two deployment shapes are supported:

1. **Same process.** Build one `PipelineState` (from
   `backend.api.main`), hand it to both the `Pipeline` and
   `create_app(state)`, and serve the app with uvicorn in a thread.
   Scored windows appear in the API immediately.
2. **Separate processes.** Run the pipeline on the camera machine and
   the API on a server. The pipeline side posts records to
   `POST /api/v1/pipeline/score`, or writes Firebase directly through
   its own `FirebaseClient`.

## Mobile app integration notes

The mobile app reads what this pipeline writes to Firebase:

- `/status/current` in the Realtime Database: latest scored window
  plus `windows_scored`, `anomalies_detected` and `updated_at`. This
  is the dashboard's data source.
- `/anomalies/events`: append-only anomaly feed for the alert list.
- FCM topic `anomalies`: subscribe to it for push notifications when
  a window scores above the Stage 3 threshold.

These are the paths the mobile app should read. If a path ever needs
to change, change it in `backend/api/firebase_client.py` (the
`STATUS_PATH` and `EVENTS_PATH` constants) and in the app at the same
time.

## Files

| File | Purpose |
| --- | --- |
| `pipeline.py` | `Pipeline` orchestrator, synthetic data generator, `--demo` mode, and the command line entry point. |
| `buffer_manager.py` | `BufferManager`, the thread-safe per-track windowing buffer described above. |

## Tests

From the repo root:

```bash
pytest tests/test_api_pipeline.py
```

Covers the buffer windowing behavior, the Firebase no-op mode, and the
API endpoints. No PyTorch, camera or Firebase account required.
