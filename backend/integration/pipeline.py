"""Integration pipeline: RTSP or video file to Stage 3 anomaly events.

Wiring::

    frames (OpenCV, file path or RTSP URL)
      -> detector callable   (frame -> tracking records)      [Stage 1]
      -> behavior callable   (clip -> behavior labels)        [Stage 2]
      -> BufferManager       (records -> fixed windows)
      -> Stage3Scorer        (windows -> anomaly scores)      [Stage 3]
      -> PipelineState       (API current status and history)
      -> FirebaseClient      (status, anomaly events, alerts)

Stage 1 and Stage 2 are pluggable callables because their models live
in backend/stage1_detection and backend/stage2_behavior and evolve
independently. Until they are wired in, pass ``detector=None`` (the
default) and the pipeline logs that Stage 1 is not connected; tracking
records can still be injected directly with :meth:`Pipeline.feed_records`,
which is also how the demo mode works.

Quick start (from the repo README)::

    python backend/integration/pipeline.py \
        --source rtsp://camera.local:8554/chicken \
        --stage3-checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
        --firebase-credentials path/to/service-account.json

Smoke test anywhere, no camera, weights or Firebase needed::

    python backend/integration/pipeline.py --demo

The pipeline scores about once per second: frames are consumed as fast
as they arrive, but ready windows are drained and scored on a
``score_interval`` cadence (default 1.0 s), matching the Stage 3 design
target of one scoring pass per second.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# Make the repo root and this folder importable in both layouts:
# run as a script (python backend/integration/pipeline.py) or imported
# as backend.integration.pipeline.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

logger = logging.getLogger("pipeline")

DetectorFn = Callable[[Any, int], List[Dict[str, Any]]]
BehaviorFn = Callable[[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]


def _import_sibling(name: str, package_path: str):
    """Import a module that lives in a sibling backend package."""
    try:
        return __import__(package_path, fromlist=[name])
    except Exception:
        return __import__(name)


class Pipeline:
    """Orchestrates Stage 1, Stage 2, Stage 3, the API state, Firebase.

    Args:
        scorer: A Stage3Scorer instance (or None: windows are counted
            and logged but not scored, useful while Stage 3 has no
            trained checkpoint yet).
        state: PipelineState from backend.api.main. When None, a local
            one is created so the pipeline still tracks current status
            and history.
        firebase_client: FirebaseClient from backend.api.firebase_client.
            When None, a default one is created (no-op without
            credentials).
        detector: Optional callable (frame, frame_index) returning a
            list of tracking records for that frame.
        behavior: Optional callable (frame, records) returning the same
            records with behavior labels attached where available.
        score_interval: Seconds between scoring passes.
    """

    def __init__(
        self,
        scorer: Any = None,
        state: Any = None,
        firebase_client: Any = None,
        detector: Optional[DetectorFn] = None,
        behavior: Optional[BehaviorFn] = None,
        score_interval: float = 1.0,
        window: int = 32,
        stride: int = 16,
    ) -> None:
        buffer_module = _import_sibling(
            "buffer_manager", "backend.integration.buffer_manager"
        )
        if state is None or firebase_client is None:
            api_main = _import_sibling("main", "backend.api.main")
            api_firebase = _import_sibling(
                "firebase_client", "backend.api.firebase_client"
            )
            if firebase_client is None:
                firebase_client = api_firebase.FirebaseClient()
            if state is None:
                state = api_main.PipelineState(
                    scorer=scorer, firebase_client=firebase_client
                )
        self.scorer = scorer
        self.state = state
        self.firebase = firebase_client
        self.detector = detector
        self.behavior = behavior
        self.score_interval = score_interval
        self.buffer = buffer_module.BufferManager(window=window, stride=stride)
        self.windows_scored = 0
        self.anomalies_detected = 0
        self._last_score_at = 0.0

        if detector is None:
            logger.warning(
                "Stage 1 detector is not connected: no frames will be "
                "tracked until a detector callable is provided. Records "
                "can still be injected with feed_records()."
            )
        if behavior is None:
            logger.info(
                "Stage 2 behavior callable is not connected: records "
                "will carry no behavior labels (Stage 3 features 10 and "
                "11 stay 0.0)."
            )
        if scorer is None:
            logger.warning(
                "Stage 3 scorer is not loaded: windows will be counted "
                "but not scored. Pass --stage3-checkpoint to enable "
                "anomaly scoring."
            )

    # ------------------------------------------------------------------
    # Feeding and scoring
    # ------------------------------------------------------------------
    def feed_records(self, records: List[Dict[str, Any]]) -> int:
        """Inject tracking records directly (demo mode, tests, Stage 1)."""
        return self.buffer.add_frame_records(records)

    def process_frame(self, frame: Any, frame_index: int) -> int:
        """Run Stage 1 and Stage 2 on one frame and buffer the records."""
        if self.detector is None:
            return 0
        records = self.detector(frame, frame_index) or []
        if self.behavior is not None and records:
            records = self.behavior(frame, records) or records
        return self.buffer.add_frame_records(records)

    def score_ready_windows(self, force: bool = False) -> List[Dict[str, Any]]:
        """Score every ready window, respecting the scoring cadence.

        Returns the list of scored window entries (empty when the
        cadence has not elapsed yet, unless ``force`` is set). Results
        are recorded into the API state, which handles the Firebase
        status write, anomaly events and push alerts.
        """
        now = time.monotonic()
        if not force and now - self._last_score_at < self.score_interval:
            return []
        self._last_score_at = now
        windows = self.buffer.pop_ready_windows()
        if not windows:
            return []
        if self.scorer is None:
            self.windows_scored += len(windows)
            logger.info(
                "Drained %d window(s) without a scorer (counted only).",
                len(windows),
            )
            return []
        entries: List[Dict[str, Any]] = []
        for window_records in windows:
            for result in self.scorer.score_frames(window_records):
                entry = self.state.record_result(result)
                entries.append(entry)
                self.windows_scored += 1
                if entry.get("is_anomaly"):
                    self.anomalies_detected += 1
                    logger.warning(
                        "ANOMALY track=%s frames=%s-%s score=%.4f "
                        "threshold=%.4f",
                        entry.get("track_id"),
                        entry.get("start_frame"),
                        entry.get("end_frame"),
                        entry.get("score", 0.0),
                        entry.get("threshold", 0.0),
                    )
        return entries

    def status(self) -> Dict[str, Any]:
        return {
            "windows_scored": self.windows_scored,
            "anomalies_detected": self.anomalies_detected,
            "buffer": self.buffer.pending_counts(),
            "stage3_loaded": self.scorer is not None,
            "firebase_enabled": bool(getattr(self.firebase, "enabled", False)),
        }

    # ------------------------------------------------------------------
    # Frame sources
    # ------------------------------------------------------------------
    def run(self, source: str, max_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Read frames from a video file or RTSP URL and process them.

        Requires OpenCV (opencv-python, already in requirements.txt).
        Runs until the source ends, ``max_seconds`` elapses, or Ctrl-C.
        """
        try:
            import cv2
        except ImportError:
            logger.error(
                "OpenCV is not installed: cannot read video sources. "
                "Use --demo to smoke test without a camera."
            )
            return self.status()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            logger.error("Could not open video source: %s", source)
            return self.status()
        logger.info("Reading frames from %s", source)
        started = time.monotonic()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    logger.info("Video source ended after %d frames.", frame_index)
                    break
                self.process_frame(frame, frame_index)
                self.score_ready_windows()
                frame_index += 1
                if max_seconds is not None and time.monotonic() - started > max_seconds:
                    logger.info("Reached --max-seconds=%s, stopping.", max_seconds)
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping pipeline.")
        finally:
            capture.release()
            self.score_ready_windows(force=True)
        return self.status()


# ----------------------------------------------------------------------
# Demo mode
# ----------------------------------------------------------------------
def synthetic_records(
    num_frames: int = 96,
    normal_tracks: int = 2,
    anomaly_track: bool = True,
    frame_size: tuple = (640, 480),
) -> List[Dict[str, Any]]:
    """Build synthetic Stage 1 plus Stage 2 output for the demo.

    Normal tracks wander slowly with steady behavior labels. When
    ``anomaly_track`` is set, one extra track sits almost perfectly
    still for the whole clip, the classic sick-bird pattern the Stage 3
    stillness feature is built to catch.
    """
    rng = np.random.default_rng(42)
    records: List[Dict[str, Any]] = []
    behaviors = ["eating", "drinking", "walking", "standing"]
    for t in range(normal_tracks):
        track_id = "normal-%d" % (t + 1)
        x = 80.0 + 140.0 * t
        y = 120.0 + 60.0 * t
        behavior = behaviors[t % len(behaviors)]
        for frame in range(num_frames):
            x += float(rng.normal(0.0, 1.2))
            y += float(rng.normal(0.0, 1.2))
            x = min(max(x, 0.0), frame_size[0] - 50.0)
            y = min(max(y, 0.0), frame_size[1] - 50.0)
            records.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "w": 46.0,
                    "h": 40.0,
                    "behavior": behavior if frame % 24 else "standing",
                }
            )
    if anomaly_track:
        for frame in range(num_frames):
            records.append(
                {
                    "frame": frame,
                    "track_id": "suspicious-1",
                    "x": 420.0,
                    "y": 300.0,
                    "w": 52.0,
                    "h": 34.0,
                    "behavior": "standing",
                }
            )
    return records


def run_demo(scorer: Any = None) -> Dict[str, Any]:
    """Run the full chain on synthetic data and print what happens.

    Without a scorer the demo still exercises the buffer and prints the
    windows that would be scored. With ``--stage3-checkpoint`` the
    windows are really scored and anomaly events are printed.
    """
    api_main = _import_sibling("main", "backend.api.main")
    api_firebase = _import_sibling("firebase_client", "backend.api.firebase_client")
    firebase = api_firebase.FirebaseClient()  # no credentials: no-op mode
    state = api_main.PipelineState(scorer=scorer, firebase_client=firebase)
    pipeline = Pipeline(scorer=scorer, state=state, firebase_client=firebase)

    records = synthetic_records()
    print("Demo: generated %d synthetic tracking records" % len(records))
    pipeline.feed_records(records)
    print("Buffer after feeding: %s" % pipeline.buffer.pending_counts())

    entries = pipeline.score_ready_windows(force=True)
    status = pipeline.status()
    print("Windows drained and handled: %d" % status["windows_scored"])
    if scorer is None:
        print(
            "No Stage 3 checkpoint given, so windows were counted but "
            "not scored. Re-run with --stage3-checkpoint to see scores "
            "and anomaly events."
        )
    else:
        for entry in entries:
            flag = "ANOMALY" if entry.get("is_anomaly") else "ok"
            print(
                "  [%s] track=%s frames=%s-%s score=%.4f threshold=%.4f"
                % (
                    flag,
                    entry.get("track_id"),
                    entry.get("start_frame"),
                    entry.get("end_frame"),
                    entry.get("score", 0.0),
                    entry.get("threshold", 0.0),
                )
            )
        print("Anomaly events: %d" % status["anomalies_detected"])
        current = state.current()
        if current is not None:
            print("Latest window now visible at GET /api/v1/anomaly/current")
    print("Firebase enabled: %s (no-op mode is expected in the demo)" % firebase.enabled)
    return status


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _load_scorer_from_checkpoint(checkpoint: Optional[str]) -> Any:
    if not checkpoint:
        return None
    stage3_dir = os.path.join(_REPO_ROOT, "backend", "stage3_anomaly")
    if stage3_dir not in sys.path:
        sys.path.insert(0, stage3_dir)
    try:
        from inference import Stage3Scorer  # type: ignore
    except Exception as exc:
        logger.error("Could not import Stage3Scorer (%s). Is torch installed?", exc)
        return None
    try:
        scorer = Stage3Scorer(checkpoint)
        logger.info("Stage 3 scorer loaded from %s", checkpoint)
        return scorer
    except Exception as exc:
        logger.error("Could not load Stage 3 checkpoint %s: %s", checkpoint, exc)
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chicken anomaly detection integration pipeline."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="video file path or RTSP URL to read frames from",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        default=os.environ.get("STAGE3_CHECKPOINT"),
        help="path to a trained Stage 3 checkpoint (or set STAGE3_CHECKPOINT)",
    )
    parser.add_argument(
        "--firebase-credentials",
        default=None,
        help="path to a Firebase service account JSON (or set FIREBASE_CREDENTIALS)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run on synthetic tracking data, no camera or weights needed",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop reading the source after this many seconds",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    scorer = _load_scorer_from_checkpoint(args.stage3_checkpoint)

    if args.demo:
        run_demo(scorer=scorer)
        return 0

    if not args.source:
        logger.error("Nothing to do: pass --source or --demo.")
        return 2

    api_main = _import_sibling("main", "backend.api.main")
    api_firebase = _import_sibling("firebase_client", "backend.api.firebase_client")
    firebase = api_firebase.FirebaseClient(
        credentials_path=args.firebase_credentials
    )
    state = api_main.PipelineState(scorer=scorer, firebase_client=firebase)
    pipeline = Pipeline(scorer=scorer, state=state, firebase_client=firebase)
    status = pipeline.run(args.source, max_seconds=args.max_seconds)
    logger.info("Pipeline finished: %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
